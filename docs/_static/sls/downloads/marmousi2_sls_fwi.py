"""Marmousi2 SLS FWI using the independently built V13 public StarWave API.
Vp: phase speed at 6 Hz. Q: modulus Q at 6 Hz (single SLS, not broadband constant Q).
Fixed constant density rho=2000 kg/m3; density is not optimized.
Rock Q=3.516e-6*Vp[m/s]**2.2; water Q=1000. Source: unit Pa/m2 Ricker forcing.
Explicit sampling dt=0.0015, nt=4000, no internal resampling.
"""
from pathlib import Path
import argparse,json,os,sys,time,math,hashlib,traceback,signal
import numpy as np
from scipy.ndimage import gaussian_filter

def arguments(argv=None):
 p=argparse.ArgumentParser()
 p.add_argument('--source-root',type=Path,required=True)
 p.add_argument('--data',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--gpu-ids',type=int,nargs='+',default=[1,2])
 p.add_argument('--epochs',type=int,default=5)
 p.add_argument('--num-batches',type=int,default=15)
 p.add_argument('--mode',choices=['vp','vpq'],default='vp')
 p.add_argument('--dt',type=float,default=.0015)
 p.add_argument('--nt',type=int,default=4000)
 p.add_argument('--sigma',type=float,default=8.)
 p.add_argument('--lr-vp',type=float,default=10.)
 p.add_argument('--lr-q',type=float,default=.5)
 p.add_argument('--loss-multiplier',type=float,default=1e6)
 p.add_argument('--pilot',action='store_true')
 return p.parse_args(argv)

def atomic_json(path,data):
 tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 tmp.replace(path)

def main(argv=None):
 a=arguments(argv);a.out.mkdir(parents=True,exist_ok=True)
 if (a.out/'summary.json').exists():raise FileExistsError('Choose a fresh output directory; existing results will not be overwritten')
 sys.path.insert(0,str(a.source_root.resolve()))
 import torch,starwave
 from starwave._visco_sls_reference import WEIGHTS,coefficients
 if not Path(starwave.__file__).resolve().is_relative_to(a.source_root.resolve()):raise RuntimeError('Wrong StarWave source import')
 torch.set_num_threads(2);torch.manual_seed(20261009);np.random.seed(20261009)
 torch.backends.cudnn.enabled=True;torch.backends.cudnn.benchmark=True
 gpu_ids=a.gpu_ids
 assert len(set(gpu_ids))==len(gpu_ids) and a.num_batches>0 and 30%a.num_batches==0
 for gpu in gpu_ids:
  if gpu<0 or gpu>=torch.cuda.device_count():raise ValueError('Invalid GPU ID')
 dev=torch.device('cuda',gpu_ids[0]);torch.cuda.set_device(dev)
 for gpu in gpu_ids:starwave.prepare_visco_sls(f"cuda:{gpu}")
 nz,nx=117,567;dx=30.;freq=6.;pml_freq=6.;pml=10;buffer=0;order=8;max_vel=6000.;water=16
 vp_true=np.fromfile(a.data,dtype=np.float32).reshape(nx,nz).T.copy()
 if not (np.isfinite(vp_true).all() and 500<vp_true.min()<2500 and 3000<vp_true.max()<6000):raise ValueError('Unexpected Marmousi2 units/shape')
 if not np.allclose(vp_true[:water],1500.):raise ValueError('Unexpected water layer in Marmousi2 data')
 q_true=(3.516e-6*vp_true.astype(np.float64)**2.2).astype(np.float32)
 q_true[:water]=1000. # Near-lossless water; do not apply rock empirical law to water.
 rho_true=np.full_like(vp_true,2000.)
 vp_init=gaussian_filter(vp_true,sigma=a.sigma).astype(np.float32)
 vp_init[:water]=vp_true[:water]
 q_init=gaussian_filter(q_true,sigma=a.sigma).astype(np.float32) if a.mode=='vpq' else q_true.copy()
 q_init[:water]=q_true[:water]
 mask=np.ones_like(vp_true);mask[:water]=0.;mask[:,[0,-1]]=0.;mask[-1]=0.
 # Preserve the earlier experiment mask for comparison. SLS itself has a complete
 # replicate-padding transpose; freezing the side frame is an experiment choice.
 src=np.zeros((30,1,2),dtype=np.int64);src[:,0,1]=18+18*np.arange(30)
 rec=np.zeros((30,nx,2),dtype=np.int64);rec[:,:,1]=np.arange(nx)[None]
 t=np.arange(a.nt,dtype=np.float64)*a.dt
 u=(np.pi*freq*(t-.45))**2
 wav=((1-2*u)*np.exp(-u)).astype(np.float32)
 tensor=lambda x:torch.as_tensor(x,device=dev,dtype=torch.float32)
 truth=[tensor(vp_true),tensor(q_true)]
 density=tensor(rho_true)
 initial=[tensor(vp_init),tensor(q_init)]
 grad_mask=tensor(mask)
 source=torch.tensor(wav,device=dev)[None,None,:].repeat(30,1,1)
 sources=torch.tensor(src,device=dev);receivers=torch.tensor(rec,device=dev)
 class Propagator(torch.nn.Module):
  def __init__(self,models,trainable):
   super().__init__()
   self.vp=torch.nn.Parameter(models[0].clone(),requires_grad=trainable)
   self.register_buffer('rho',density.clone())
   if a.mode=='vpq' and trainable:self.Q=torch.nn.Parameter(models[1].clone())
   else:self.register_buffer('Q',models[1].clone())
  def forward(self,source_amplitudes,source_locations,receiver_locations):
   return starwave.visco_sls(self.vp,self.Q,self.rho,dx,a.dt,f_ref=freq,
    source_amplitudes=source_amplitudes,source_locations=source_locations,
    receiver_locations=receiver_locations,accuracy=order,pml_width=pml,
    max_vel=max_vel,memory='full',backend='cuda')[0]
 def parallel(m):
  return torch.nn.DataParallel(m,device_ids=gpu_ids,output_device=gpu_ids[0]) if len(gpu_ids)>1 else m
 def sync():
  for g in gpu_ids:torch.cuda.synchronize(g)
 def memory():
  return {str(g):{'allocated_peak_GiB':torch.cuda.max_memory_allocated(g)/2**30,
   'reserved_peak_GiB':torch.cuda.max_memory_reserved(g)/2**30,
   'free_GiB':torch.cuda.mem_get_info(g)[0]/2**30} for g in gpu_ids}
 def plan(models):
  with torch.no_grad():
   c2,tau,_,_=coefficients(models[0],models[1],a.dt,freq)
   unrelaxed=float((c2*(1+tau)).sqrt().max())
   bound=.8/(max_vel*math.sqrt(float(density.max()/density.min()))*sum(abs(x) for x in WEIGHTS[order])*math.sqrt(2)/dx)
  if a.dt>bound or unrelaxed>max_vel:raise ValueError(f'Invalid fixed SLS time setup: dt={a.dt}, limit={bound}, unrelaxed={unrelaxed}')
  return dict(user_dt=a.dt,user_nt=a.nt,internal_dt=a.dt,internal_nt=a.nt,
   step_ratio=1,max_unrelaxed_speed=unrelaxed,max_vel=max_vel,max_dt=bound)
 plans={'true':plan(truth),'initial':plan(initial)}
 indices=[np.arange(i,30,a.num_batches) for i in range(a.num_batches)]
 config=dict(source_root=str(a.source_root.resolve()),source_import=str(starwave.__file__),data_path=str(a.data),
 data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),pid=os.getpid(),mode=a.mode,
 gpu_ids=gpu_ids,gpu_names=[torch.cuda.get_device_name(g) for g in gpu_ids],
 shape_zx=[nz,nx],dx=dx,dt=a.dt,nt=a.nt,frequency_Hz=freq,reference_omega=2*math.pi*freq,
 wavelet_peak_sample=round(.45/a.dt),wavelet_peak_seconds=.45,source_units="Pa/m^2",pml_freq_Hz=pml_freq,pml_width=pml,boundary_buffer=buffer,
 order=order,memory='full',max_vel=max_vel,epochs=a.epochs,num_shots=30,
 num_batches=a.num_batches,shots_per_batch=30//a.num_batches,receivers_per_shot=nx,
 source_start=18,source_spacing=18,source_depth=0,receiver_depth=0,
 gaussian_sigma_cells=a.sigma,water_rows=water,water_Q=1000.,Q_relation='3.516e-6 * Vp[m/s]**2.2; NOT Gardner',
 Q_relation_reference='https://html.rhhz.net/dqwlxb/2016-11-4212.htm',
 fixed_Q=(a.mode=='vp'),fixed_Q_is_true=(a.mode=='vp'),
 mask='water and outermost physical cells fixed at initialization',
 density='fixed constant rho=2000 kg/m3; not optimized',
 density_range_kg_m3=[float(rho_true.min()),float(rho_true.max())],
 equation='independent single SLS: reference phase Vp and modulus Q at 6 Hz',
 lr_vp=a.lr_vp,lr_q=a.lr_q if a.mode=='vpq' else None,loss_multiplier=a.loss_multiplier,
 experiment_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 loss='loss_multiplier * MSE / global observed mean-square (both scalars fixed)',
 adam_betas=[.5,.99],plans=plans,pilot=a.pilot)
 atomic_json(a.out/'config.json',config)
 report=dict(status='STARTING',config=config,history=[],started_unix=time.time(),
 diagnostic_note='SLS single-shot Marmousi variable-Q records passed the separate 12-second stability check. This short FWI tests the workflow, not final image convergence.')
 def status(name,**kw):
  report.update(status=name,**kw);atomic_json(a.out/'summary.json',report)
 def rmse(model,target):
  err=(model.detach()-target).square()
  return {'whole':float(err.mean().sqrt()),'trainable_region':float(err[grad_mask.bool()].mean().sqrt())}
 def printlog(*s):print(time.strftime('%Y-%m-%d %H:%M:%S'),*s,flush=True)
 np.savez_compressed(a.out/'models_initial.npz',vp_true=vp_true,q_true=q_true,vp_init=vp_init,q_init=q_init,
  rho_true=rho_true,gradient_mask=mask,source_locations=src,receiver_locations=rec,source_wavelet=wav)
 # Agg avoids a remote display and writes only to the run's result directory.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 extent=[0,(nx-1)*dx/1000,(nz-1)*dx/1000,0]
 def draw(epoch,owner):
  pairs=[('Vp (m/s)',vp_true,vp_init,owner.vp.detach().cpu().numpy()),('Q',q_true,q_init,owner.Q.detach().cpu().numpy())]
  fig,axes=plt.subplots(2,3,figsize=(17,6),layout='constrained')
  for row,(label,true,init,current) in enumerate(pairs):
   lo=min(x[water:].min() for x in [true,init,current]);hi=max(x[water:].max() for x in [true,init,current])
   for ax,arr,title in zip(axes[row],[true,init,current],['True','Initial',f'Epoch {epoch}']):
    # Keep high-Q fixed water visually distinct without saturating rock contrast.
    img=ax.imshow(arr,extent=extent,cmap='jet',vmin=lo,vmax=hi,aspect='auto')
    ax.set_title(label+' — '+title);ax.set_xlabel('Distance (km)');ax.set_ylabel('Depth (km)')
   fig.colorbar(img,ax=list(axes[row]),label=label,shrink=.9)
  fig.savefig(a.out/f'models_epoch_{epoch:03d}.png',dpi=160);plt.close(fig)
 def checkpoint(epoch,owner,opt):
  tmp=a.out/'checkpoint.tmp'
  torch.save({'epoch':epoch,'vp':owner.vp.detach().cpu(),'Q':owner.Q.detach().cpu(),
   'optimizer':opt.state_dict(),'config':config,'history':report['history']},tmp)
  tmp.replace(a.out/'checkpoint_latest.pt')
  np.savez_compressed(a.out/f'models_epoch_{epoch:03d}.npz',vp=owner.vp.detach().cpu().numpy(),Q=owner.Q.detach().cpu().numpy())
  draw(epoch,owner)
  if report['history']:
   fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
   ax.semilogy([x['epoch'] for x in report['history']],[x['normalized_MSE_before_updates'] for x in report['history']])
   ax.set(xlabel='Epoch',ylabel='Normalized batch MSE',title='Marmousi2 viscoacoustic FWI');ax.grid(alpha=.25)
   fig.savefig(a.out/'loss.png',dpi=160);plt.close(fig)
  atomic_json(a.out/'summary.json',report)
 training=Propagator(initial,True).to(dev);predict=parallel(training)
 groups=[dict(params=[training.vp],lr=a.lr_vp)]
 if a.mode=='vpq':groups.append(dict(params=[training.Q],lr=a.lr_q))
 opt=torch.optim.Adam(groups,betas=(.5,.99),eps=1e-8)
 report['initial_rmse']={'vp':rmse(training.vp,truth[0]),'Q':rmse(training.Q,truth[1])}
 draw(0,training)
 fig,ax=plt.subplots(figsize=(12,3),layout='constrained')
 img=ax.imshow(rho_true,extent=extent,cmap='viridis',aspect='auto')
 ax.set(xlabel='Distance (km)',ylabel='Depth (km)',title='Fixed constant density')
 fig.colorbar(img,ax=ax,label='Density (kg/m3)')
 fig.savefig(a.out/'fixed_density.png',dpi=160);plt.close(fig)
 status('GENERATING_OBSERVATIONS')
 obs_owner=Propagator(truth,False).to(dev);observe=parallel(obs_owner)
 all_obs=np.lib.format.open_memmap(a.out/'observed.npy',mode='w+',dtype=np.float32,shape=(30,nx,a.nt))
 stages=indices[:1] if a.pilot else indices
 start=time.perf_counter()
 for bi,ix in enumerate(stages):
  printlog('Observed batch',bi+1,'/',len(stages),'shots',ix.tolist())
  with torch.no_grad(): y=observe(source[ix],sources[ix],receivers[ix])
  sync()
  if not torch.isfinite(y).all():raise RuntimeError('Nonfinite observation')
  all_obs[ix]=y.cpu().numpy();all_obs.flush();del y
  status('GENERATING_OBSERVATIONS',observation_batches_done=bi+1)
 del observe,obs_owner
 report['observation_seconds']=time.perf_counter()-start
 if a.pilot:scale=float(np.mean(np.square(np.asarray(all_obs[indices[0]],dtype=np.float64))))
 else:scale=float(np.mean(np.square(all_obs,dtype=np.float64)))
 if not np.isfinite(scale) or scale<=0:raise RuntimeError('Invalid observed data energy')
 report['observed_mean_square']=scale
 printlog('Observation seconds',report['observation_seconds'],'scale',scale,'plans',plans)
 # Complete-survey baseline before any update, matching the final evaluation.
 total_initial=0.
 with torch.no_grad():
  for ix in stages:
   yp=predict(source[ix],sources[ix],receivers[ix])
   yo=torch.as_tensor(np.asarray(all_obs[ix]).copy(),device=dev)
   total_initial+=float(torch.nn.functional.mse_loss(yp,yo)/scale)
   del yp,yo
 sync()
 report['initial_normalized_MSE']=total_initial/len(stages)
 report['initial_optimized_loss']=a.loss_multiplier*report['initial_normalized_MSE']
 printlog('Initial complete-survey normalized MSE',report['initial_normalized_MSE'])
 for g in gpu_ids:
  with torch.cuda.device(g):torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats(g)
 stop_requested=False
 def stop(signum,frame):
  nonlocal stop_requested
  stop_requested=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 max_epochs=1 if a.pilot else a.epochs
 start=time.perf_counter()
 for epoch in range(max_epochs):
  tic=time.perf_counter();epoch_loss=0.;epoch_raw=0.
  status('INVERTING',current_epoch=epoch+1,current_batch=0)
  for bi,ix in enumerate(stages):
   batchobs=torch.as_tensor(np.asarray(all_obs[ix]).copy(),device=dev)
   opt.zero_grad(set_to_none=True)
   printlog('Epoch',epoch+1,'batch',bi+1,'/',len(stages),'forward/backward')
   sync();btic=time.perf_counter()
   pred=predict(source[ix],sources[ix],receivers[ix])
   raw=torch.nn.functional.mse_loss(pred,batchobs);normalized=raw/scale;loss=a.loss_multiplier*normalized
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
   loss.backward()
   training.vp.grad.mul_(grad_mask)
   if a.mode=='vpq':training.Q.grad.mul_(grad_mask)
   for param in training.parameters():
    if param.grad is None or not torch.isfinite(param.grad).all():raise RuntimeError('Invalid parameter gradient')
   if epoch in [0,max_epochs//2,max_epochs-1] and bi==0:
    grads={'vp':training.vp.grad.detach().cpu().numpy()}
    if a.mode=='vpq':grads['Q']=training.Q.grad.detach().cpu().numpy()
    np.savez_compressed(a.out/f'gradient_epoch_{epoch+1:03d}.npz',**grads)
   # Bounds are explicit experiment parameters. No solver/CFL changes.
   opt.step()
   with torch.no_grad():
    training.vp.clamp_(800.,5000.)
    if a.mode=='vpq':training.Q.clamp_(5.,1200.)
   sync()
   value=float(normalized);epoch_loss+=value;epoch_raw+=float(raw)
   elapsed=time.perf_counter()-btic
   printlog('Done epoch',epoch+1,'batch',bi+1,'normalized_MSE',value,'optimized_loss',float(loss),'seconds',elapsed)
   status('INVERTING',current_epoch=epoch+1,current_batch=bi+1,last_batch_seconds=elapsed,memory=memory())
   del pred,loss,raw,normalized,batchobs
   if stop_requested:
    report.update(status='STOPPED_AFTER_BATCH',completed_epochs=epoch,current_epoch=epoch+1,current_batch=bi+1)
    checkpoint(epoch,training,opt);return 0
  entry=dict(epoch=epoch+1,normalized_MSE_before_updates=epoch_loss/len(stages),optimized_loss_before_updates=a.loss_multiplier*epoch_loss/len(stages),
   raw_MSE_before_updates=epoch_raw/len(stages),seconds=time.perf_counter()-tic,
   vp_rmse=rmse(training.vp,truth[0]),Q_rmse=rmse(training.Q,truth[1]),
   vp_update_from_initial={'max_abs_m_per_s':float((training.vp.detach()-initial[0]).abs().max()),'rms_m_per_s':float((training.vp.detach()-initial[0]).square().mean().sqrt())},
   time_plan=plan([training.vp,training.Q]),memory=memory())
  report['history'].append(entry);status('INVERTING',completed_epochs=epoch+1)
  printlog('EPOCH COMPLETE',json.dumps(entry,ensure_ascii=False))
  if a.pilot or max_epochs<=5 or (epoch+1)%10==0 or epoch+1==max_epochs:checkpoint(epoch+1,training,opt)
 # Evaluate the final model without optimizer updates, on the same complete survey.
 total=0.
 with torch.no_grad():
  for ix in stages:
   pred=predict(source[ix],sources[ix],receivers[ix]);obs=torch.as_tensor(np.asarray(all_obs[ix]).copy(),device=dev)
   total+=float(torch.nn.functional.mse_loss(pred,obs)/scale)
   del pred,obs
 sync()
 report.update(status='PILOT_COMPLETED' if a.pilot else 'COMPLETED',completed_epochs=max_epochs,
  final_normalized_MSE=total/len(stages),final_optimized_loss=a.loss_multiplier*total/len(stages),final_rmse={'vp':rmse(training.vp,truth[0]),'Q':rmse(training.Q,truth[1])},
  inversion_wall_seconds=time.perf_counter()-start,memory=memory(),finished_unix=time.time())
 report['finite_and_improved']=bool(report['final_normalized_MSE']<report['initial_normalized_MSE']
  and report['final_rmse']['vp']['trainable_region']<report['initial_rmse']['vp']['trainable_region'])
 checkpoint(max_epochs,training,opt)
 printlog('FINISHED',report['status'],'improved',report['finite_and_improved'],'seconds',report['inversion_wall_seconds'])
 return 0

if __name__=='__main__':
 try:sys.exit(main())
 except Exception:
  traceback.print_exc()
  # Preserve the exact failed experiment and report a nonzero process exit.
  args=arguments()
  if args.out.exists():
   path=args.out/'summary.json'
   report=json.loads(path.read_text()) if path.exists() else {}
   report.update(status='FAILED',traceback=traceback.format_exc(),finished_unix=time.time())
   atomic_json(path,report)
  sys.exit(1)
