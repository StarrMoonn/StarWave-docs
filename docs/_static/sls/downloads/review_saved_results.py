"""Inspect saved SLS results; no solver calls, network, installation or GPU required."""
from pathlib import Path
import argparse, json
import numpy as np

def review(root):
    root=Path(root)
    reports={}
    for label in ['fixed_Q_Vp','joint_Vp_Q']:
        p=root/'data'/label
        summary=json.loads((p/'summary.json').read_text())
        with np.load(p/'models_initial.npz',allow_pickle=False) as initial, np.load(p/'models_epoch_100.npz',allow_pickle=False) as final:
            mask=initial['gradient_mask'].astype(bool)
            metrics={}
            for key,true_key in [('vp','vp_true'),('Q','q_true')]:
                error=final[key].astype(np.float64)-initial[true_key].astype(np.float64)
                metrics[key]={'whole_grid_RMSE':float(np.sqrt(np.mean(error**2))),'trainable_region_RMSE':float(np.sqrt(np.mean(error[mask]**2)))}
        reports[label]={'status':summary['status'],'completed_epochs':summary['completed_epochs'],'full_survey_initial_normalized_MSE':summary['initial_normalized_MSE'],'full_survey_final_normalized_MSE':summary['final_normalized_MSE'],'recomputed_final_RMSE':metrics}
    return reports

def plot_history(root,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for label,title in [('fixed_Q_Vp','Fixed true Q'),('joint_Vp_Q','Joint Vp/Q')]:
        d=json.loads((root/'data'/label/'summary.json').read_text());h=d['history'];x=[e['epoch'] for e in h]
        axes[0].semilogy(x,[e['normalized_MSE_before_updates'] for e in h],label=title)
        axes[1].plot([0]+x,[d['initial_rmse']['vp']['whole']]+[e['vp_rmse']['whole'] for e in h],label=title)
        axes[2].plot([0]+x,[d['initial_rmse']['Q']['whole']]+[e['Q_rmse']['whole'] for e in h],label=title)
    for ax,title,ylab in zip(axes,['Training residual','Vp model error','Q model error'],['Normalized pre-update batch MSE','Whole-grid RMSE (m/s)','Whole-grid RMSE']):
        ax.set(title=title,xlabel='Epoch',ylabel=ylab);ax.grid(alpha=.2);ax.legend()
    fig.savefig(output,dpi=160);plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--plot',type=Path,help='Optional output PNG path')
    args=parser.parse_args()
    print(json.dumps(review(args.root),indent=2))
    if args.plot:plot_history(args.root,args.plot)
