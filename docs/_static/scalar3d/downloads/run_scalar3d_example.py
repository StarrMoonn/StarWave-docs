"""Run a complete reproducible example: CUDA FWI, checkpoint gradients, CPU figures."""
from pathlib import Path
import argparse,subprocess,sys,json,os
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--case",choices=("enclosed","surface","layered"),default="enclosed")
 p.add_argument("--source-root",type=Path,required=True)
 p.add_argument("--output-dir",type=Path,required=True)
 p.add_argument("--gpu-ids",default="2,0,1,3")
 p.add_argument("--epochs",type=int)
 a=p.parse_args()
 item=json.loads((ROOT/"case_manifest.json").read_text())["cases"][a.case]
 epochs=a.epochs or item["epochs"]
 if epochs<1:raise ValueError("epochs must be positive")
 env=os.environ.copy();env.update(OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",MKL_THREADING_LAYER="GNU",MPLBACKEND="Agg")
 subprocess.run([sys.executable,"-u",str(ROOT/item["script"]),"--source-root",str(a.source_root),
                 "--output-dir",str(a.output_dir),"--gpu-ids",a.gpu_ids,"--epochs",str(epochs)],check=True,env=env)
 available=[0]+list(range(10,epochs+1,10))
 if epochs not in available:available.append(epochs)
 indices=sorted(set([0,len(available)//4,len(available)//2,3*len(available)//4,len(available)-1]))
 selected=",".join(str(available[i]) for i in indices)
 subprocess.run([sys.executable,"-u",str(ROOT/"record_checkpoint_gradients.py"),str(a.output_dir),
                 "--source-root",str(a.source_root),"--gpu-ids",a.gpu_ids,"--epochs",selected],check=True,env=env)
 subprocess.run([sys.executable,str(ROOT/"render_scalar3d.py"),str(a.output_dir)],check=True,env=env)
 summary=json.loads((a.output_dir/"summary.json").read_text())
 print(json.dumps({"status":summary["status"],"numerical_checks":summary["numerical_checks"],
                   "quality_checks":summary["quality_checks"],"results":str(a.output_dir)},indent=2))
if __name__=="__main__":main()
