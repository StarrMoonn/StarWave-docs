#!/usr/bin/env python3
"""CPU-only 2-D top-PML/free-surface examples for three elastic operators.

Install this checkout normally (pip install -e .), then run this script.
No native libraries, custom imports, PDE changes, or backend fallbacks are used.
The figures illustrate forward responses; they are not physical validation,
convergence, stability, GPU/3-D, inversion, or performance benchmarks.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import traceback

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import starwave

SOURCE_COMMIT = '200621cc31c0ebeebaa01ffaa3591c0ebd7e0adb'
NX, NZ, DX, DT, NT = 80, 64, 10., .001, 750
FREQ, PEAK, SOURCE_SCALE, ORDER, PML = 15., .100, 1.e9, 2, 15


def scalar_stats(array):
    a = np.asarray(array, dtype=np.float64)
    return {'finite': bool(np.isfinite(a).all()), 'max_abs': float(np.max(np.abs(a))),
            'rms': float(np.sqrt(np.mean(a*a)))}


def sampling_dict(plan):
    return dataclasses.asdict(plan)


def make_inputs():
    z = torch.arange(NZ, dtype=torch.float32)[None, :].expand(NX, NZ) * DX
    vp = torch.where(z < 300., 2000., 2600.)
    vs = torch.where(z < 300., 1000., 1400.)
    rho = torch.where(z < 300., 2000., 2200.)
    source_time = torch.arange(NT, dtype=torch.float32)*DT
    arg = (math.pi*FREQ*(source_time-PEAK)).square()
    source = ((1-2*arg)*torch.exp(-arg)*SOURCE_SCALE)[None, None, :]
    sloc = torch.tensor([[[40, 8]]], dtype=torch.long)
    rx = torch.arange(10, 71, dtype=torch.long)
    rloc = torch.stack((rx, torch.full_like(rx, 5)), dim=-1)[None]
    return vp, vs, rho, source, sloc, rloc


def run_pair(name):
    vp, vs, rho, src, sloc, rloc = make_inputs()
    common = {'accuracy': ORDER, 'pml_freq': FREQ, 'memory': 'full',
              'backend': 'torch', 'execution': 'eager'}
    meta = {'operator': name, 'source_commit': SOURCE_COMMIT, 'dimension': 2,
            'device': 'cpu', 'backend': 'torch', 'execution': 'eager',
            'dtype': 'float32', 'torch_version': torch.__version__, 'threads': 2,
            'gradient_enabled': False, 'memory': 'full',
            'nx': NX, 'nz': NZ, 'dx_m': DX, 'dz_m': DX, 'dt_s': DT, 'nt': NT,
            'accuracy': ORDER, 'pml_width_cells': PML, 'pml_freq_hz': FREQ,
            'min_s_wave_grid_points_at_peak_frequency': 1000./(FREQ*DX),
            'source_frequency_note': 'Elastic examples use 15 Hz to avoid the package 20 Hz, five-grid-point S-wavelength warning. A peak-frequency sampling check is not a convergence or full-bandwidth accuracy test.',
            'layer_interface_depth_m': 300., 'vp_m_s': [2000., 2600.],
            'vs_m_s': [1000., 1400.], 'rho_kg_m3': [2000., 2200.],
            'source': {'wavelet': 'Ricker', 'frequency_hz': FREQ, 'peak_time_s': PEAK,
                       'peak_amplitude': SOURCE_SCALE, 'units': 'Pa/s',
                       'grid_index_x_z': [40, 8], 'nominal_x_z_m': [400., 80.]},
            'receivers': {'component': 'vertical_particle_velocity', 'units': 'm/s',
                          'count': 61, 'nominal_x_m': list(range(100, 701, 10)),
                          'grid_index_z': 5, 'nominal_depth_m': 50.,
                          'staggering': 'Native staggered component samples retained, not interpolated to normal-stress nodes.'},
            'comparison_scope': 'Within each operator only: identical physical inputs and source/receiver indices; change top boundary. Amplitudes and phases are not a cross-operator equivalence test.',
            'validation_scope': 'Finite CPU 2-D forward illustrative run only; no convergence, physical acceptance, long-time stability, GPU, 3-D, inversion, or performance benchmark.'}
    extra = {'vp_x_z_m_s': vp.numpy(), 'vs_x_z_m_s': vs.numpy(),
             'rho_x_z_kg_m3': rho.numpy(), 'source_amplitudes': src[0, 0].numpy(),
             'source_time_s': np.arange(NT)*DT, 'source_index_x_z': sloc[0].numpy(),
             'receiver_index_x_z': rloc[0].numpy()}
    records, plans = {}, {}
    if name == 'elastic_tti':
        epsilon, delta, theta = torch.full_like(vp, .1), torch.full_like(vp, .05), torch.full_like(vp, math.radians(20.))
        models = (vp, vs, rho, epsilon, delta, theta)
        # Read-only inspection of the package planner. Use one fixed conservative
        # speed bound for both runs, so their internal time sampling is identical.
        from starwave._elastic_tti_setup import plan_time
        free_plan = plan_time(models, (DX, DX), DT, NT, ORDER, None, True, 2)
        fixed_bound = math.ceil(free_plan.max_vel / 100.) * 100.
        meta.update(model_axes=['x', 'z'], coupling_order=2, epsilon=.1, delta=.05,
                    theta_degrees=20., theta_radians=math.radians(20.), max_vel=fixed_bound,
                    surface_convention='Variational traction-free wall z=-dz/2=-5 m, half a cell above normal row z=0; wall shear strain is Schur-condensed and wall velocity carries half-cell mass. Top CPML removed; side/bottom CPML retained.',
                    time_convention='API records aligned to 0, dt, ..., (nt-1)*dt, initial zero record prepended before resampling.')
        meta['source']['component'] = 'equal sxx and szz isotropic stress rates'
        meta['receivers']['api_component'] = 'vz'
        meta['receivers']['actual_depth_m'] = 55.
        for key, free in [('absorbing', False), ('free_surface', True)]:
            print(f'{name}: {key}', flush=True)
            plan = plan_time(models, (DX, DX), DT, NT, ORDER, fixed_bound, free, 2)
            plans[key] = sampling_dict(plan)
            records[key] = starwave.elastic_tti(*models, DX, DT,
                source_amplitudes=src, source_locations=sloc, receiver_locations=rloc,
                receiver_component='vz', coupling_order=2, pml_width=PML,
                boundary_buffer=5, max_vel=fixed_bound, free_surface=free, **common)[0][0].numpy()
        time_s = np.arange(NT)*DT
        extra.update(epsilon_x_z=epsilon.numpy(), delta_x_z=delta.numpy(), theta_x_z_rad=theta.numpy())
    elif name == 'elastic_vrr':
        params = starwave.elastic_vrr_parameters(vp, vs, rho, DX, accuracy=ORDER)
        meta.update(model_axes=['x', 'z'], reflectivity_layout='edge',
                    reflectivity_definition='Physical coefficients from elastic_vrr_parameters: R = 0.5 D log(rho V), positive-edge samples.',
                    surface_convention='Flat improved-vacuum closure: physical row z=0 is vacuum, first solid normal-stress row z=1 (10 m); zero vacuum moduli and half-mass normal face. Top CPML removed; side/bottom CPML retained.',
                    time_convention='API samples completed user steps: dt, 2*dt, ..., nt*dt; native momentum/stress half-step staggering retained.')
        meta['source']['component'] = 'szz stress_rate only (not isotropic pressure source)'
        meta['receivers'].update(api_component='vz', actual_depth_m=55., density_conversion='Requested vz explicitly: pz divided by package geometric face density; pz itself is momentum, not velocity.')
        extra.update({key: value.numpy() for key, value in zip(['rp_x','rp_z','rs_x','rs_z'], params[2:])})
        for key, free in [('absorbing', False), ('free_surface', True)]:
            print(f'{name}: {key}', flush=True)
            result = starwave.elastic_vrr(*params, DX, DT, source_amplitudes=src,
                source_locations=sloc, receiver_locations=rloc, source_component='szz',
                source_type='stress_rate', receiver_components=('vz',), density=rho,
                pml_width=PML, max_vel=2600., free_surface=free,
                return_time_sampling=True, **common)
            records[key] = result[0][0].numpy()
            plans[key] = sampling_dict(result[-1])
        time_s = (np.arange(NT)+1)*DT
    elif name == 'elastic':
        lamb, mu, buoyancy = starwave.common.vpvsrho_to_lambmubuoyancy(vp.T.contiguous(), vs.T.contiguous(), rho.T.contiguous())
        from starwave._deepwave.common import cfl_condition_n
        internal_dt, step_ratio = cfl_condition_n([DX, DX], DT, 2600.)
        meta.update(model_axes=['y (physical vertical z)', 'x'],
                    surface_convention='Flat improved-vacuum closure selected by top pml_width=0: first physical material row y=0 (physical z=0) becomes fixed vacuum, first solid row y=1 (10 m). Not a collocated exact traction constraint at the first solid stress node.',
                    time_convention='Deepwave-compatible velocity samples at (n-0.5)*dt; pressure/stress source samples at n*dt.')
        meta['source']['component'] = 'source_amplitudes_p isotropic pressure rate (added as equal negative normal-stress increments)'
        meta['source']['stress_sign'] = 'The package adds -internal_dt * source_amplitudes_p to each normal stress.'
        meta['source']['api_index_y_x'] = [8, 40]
        meta['receivers']['api_component'] = 'receiver_locations_y, result[-2]'
        meta['receivers']['staggering'] = 'Deepwave native y-velocity stagger retained; geometry figure shows integer-index reference locations, not a co-location with pressure nodes.'
        for key, free in [('absorbing', False), ('free_surface', True)]:
            print(f'{name}: {key}', flush=True)
            widths = [0 if free else PML, PML, PML, PML]
            result = starwave.elastic(lamb, mu, buoyancy, DX, DT,
                source_amplitudes_p=src, source_locations_p=sloc.flip(-1),
                receiver_locations_y=rloc.flip(-1), pml_width=widths, max_vel=2600., **common)
            records[key] = result[-2][0].numpy()
            plans[key] = {'user_dt': DT, 'user_nt': NT, 'internal_dt': internal_dt,
                          'internal_nt': NT*step_ratio, 'step_ratio': step_ratio, 'max_vel': 2600.,
                          'pml_width_top_bottom_left_right': widths}
        time_s = (np.arange(NT)-.5)*DT
    else:
        raise ValueError(name)
    records['difference'] = records['free_surface'] - records['absorbing']
    for key, data in records.items():
        assert data.shape == (61, NT), (name, key, data.shape)
        if not np.isfinite(data).all():
            raise RuntimeError(f'{name}/{key}: non-finite seismogram')
    if np.max(np.abs(records['absorbing'])) == 0 or np.max(np.abs(records['difference'])) == 0:
        raise RuntimeError(f'{name}: zero response or identical top-PML/free-surface records')
    meta.update(time_sampling=plans, statistics={key: scalar_stats(data) for key, data in records.items()})
    meta['statistics']['relative_difference_l2'] = float(np.linalg.norm(records['difference'].astype(np.float64))/np.linalg.norm(records['absorbing'].astype(np.float64)))
    return records, time_s, extra, meta


def draw(name, records, time_s, extra, meta, path):
    # Shared absolute amplitude scale for ALL three images, no AGC or trace normalization.
    color_limit = float(np.quantile(np.abs(np.concatenate([v.ravel() for v in records.values()])), .995))
    if color_limit == 0: color_limit = max(float(np.max(np.abs(v))) for v in records.values())
    meta['plot'] = {'colormap': 'RdBu_r', 'shared_symmetric_limit_m_s': color_limit,
                    'clip_quantile': .995, 'shared_for': ['absorbing', 'free_surface', 'difference'],
                    'normalization': 'No AGC or per-trace normalization; 99.5th-percentile clipping shared across all panels.'}
    fig = plt.figure(figsize=(15.5, 9), layout='constrained')
    grid = fig.add_gridspec(2, 3, height_ratios=[1.65, 1.])
    image_axes = []
    for i, (key, title) in enumerate([('absorbing','Top PML (no free surface)'), ('free_surface','Flat free surface'), ('difference','Difference: free surface - top PML')]):
        ax = fig.add_subplot(grid[0, i]); image_axes.append(ax)
        im = ax.imshow(records[key].T, origin='upper', extent=[95,705,time_s[-1]+DT/2,time_s[0]-DT/2],
                       cmap='RdBu_r', vmin=-color_limit, vmax=color_limit, aspect='auto', interpolation='nearest')
        ax.set(xlabel='Receiver x (m)', ylabel='Record time (s)' if i==0 else '', title=title)
    fig.colorbar(im, ax=image_axes, label='Vertical velocity (m/s); shared clipped scale', shrink=.93)
    ax=fig.add_subplot(grid[1,0])
    ax.imshow(extra['vp_x_z_m_s'].T, extent=[-5,795,635,-5], aspect='auto', cmap='viridis', vmin=2000, vmax=2600)
    ax.scatter([400],[80],marker='*',s=110,color='#ef4444',edgecolor='white',label='Source index')
    ax.plot(np.arange(100,701,10),np.full(61,50.),'.',color='white',markersize=3,label='Receiver indices')
    ax.axhline(300,color='white',linestyle=':',lw=1)
    ax.text(15,210,'Vp 2000 | Vs 1000 m/s',color='white',fontsize=9)
    ax.text(15,480,'Vp 2600 | Vs 1400 m/s',color='white',fontsize=9)
    ax.set(xlabel='x (m)',ylabel='Depth index coordinate (m)',title='Shared model and acquisition')
    ax.legend(loc='lower right',fontsize=8)
    ax=fig.add_subplot(grid[1,1])
    k=40 # x=500 m, 100 m nominal offset
    ax.plot(time_s,records['absorbing'][k],label='Top PML',color='#334155',lw=1.1)
    ax.plot(time_s,records['free_surface'][k],label='Free surface',color='#dc2626',lw=1.)
    ax.set(xlabel='Record time (s)',ylabel='Vertical velocity (m/s)',title='Unnormalized trace: x = 500 m')
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    ax=fig.add_subplot(grid[1,2])
    ax.plot(extra['source_time_s'],extra['source_amplitudes']/SOURCE_SCALE,color='#4338ca')
    ax.set(xlim=(0,.23),xlabel='Source sample time (s)',ylabel='Amplitude / 1e9 Pa/s',title=f'{FREQ:g} Hz Ricker source; peak at {PEAK:.3f} s')
    ax.grid(alpha=.2)
    fig.suptitle(f'{name}: CPU / Torch eager / 2-D / FD order 2\nWithin-operator boundary comparison; native source and sampling conventions retained',fontsize=15)
    fig.savefig(path,dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operators',nargs='+',default=['elastic','elastic_vrr','elastic_tti'],choices=['elastic','elastic_vrr','elastic_tti'])
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'results')
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    failures=[]
    with torch.no_grad():
        for name in args.operators:
            try:
                records,time_s,extra,meta=run_pair(name)
                draw(name,records,time_s,extra,meta,args.output_dir/f'{name}.png')
                np.savez_compressed(args.output_dir/f'{name}.npz',time_s=time_s,
                    receiver_x_m=np.arange(100,701,10),**records,**extra)
                (args.output_dir/f'{name}.json').write_text(json.dumps(meta,indent=2)+'\n')
                print(name+' DONE '+json.dumps(meta['statistics']),flush=True)
            except Exception:
                report=traceback.format_exc(); print(report,flush=True)
                (args.output_dir/f'{name}.failure.log').write_text(report)
                failures.append(name)
    if failures: raise SystemExit('Failed: '+', '.join(failures))

if __name__=='__main__': main()
