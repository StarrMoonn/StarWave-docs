# -*- coding: utf-8 -*-
# Paired executable script exported from the public notebook.

# %% [markdown]
# # 03 · A small full-waveform inversion
# 
# 这是一套为 StarWave 独立设计的教学实验。只用解析模型、NumPy、Matplotlib、PyTorch 和 StarWave；无需外部地震数据、Deepwave 包或项目 utils。三份 notebook 都可单独从头运行。
# 
# 这是原始教学 notebook 的公开清理版：仅调整说明文字、清除输出与执行元数据，并从环境记录 helper 中删除不必要的开发来源字段。其余代码单元（数值模型、配置、传播、梯度、优化、绘图与结果导出）与原始交付逐字一致。原始与公开文件的 SHA-256 及逐单元核验范围见 [source_manifest.json](source_manifest.json)。
# 
# 原始实验后来在 NVIDIA A30 / PyTorch 2.5.1 / CUDA 11.8 / StarWave 0.1.0.dev9 环境实际运行；整理后的记录见 [results_summary.json](results_summary.json)。这些记录不证明公开 2.0.0 wheel 通过验证。此公开清理版没有重新执行，不能称为这些文件逐字节运行的结果。学习目标是读懂调用与梯度链；微型合成案例不构成现场 FWI、完整伴随、收敛或多 GPU 验收。
# 
# Public edition: both documentation languages download these same original Chinese-language teaching notebooks. Only privacy/editorial changes were made; numerical experiment cells are unchanged. This public edition has not been run byte-for-byte. Measured results came from the original notebooks on StarWave 0.1.0.dev9, not a tested public 2.0.0 wheel. See the linked source manifest and results summary for provenance and limits.

# %% [markdown]
# ## 实验契约与参数
# - 2D 恒密度声学，模型直接存为 `[nx,nz]`，整数位置 `[x_index,z_index]`；绘图时才转置。网格 96×64、间距 10 m，对应 x=0…950 m、z=0…630 m。
# - 平滑速度梯度 + 一条平滑层界面 + 正速度透镜；初始模型没有透镜。不是任何现成基准模型，没有水层。
# - 3 个独立单源炮，源深 50 m；38 个接收点、间距 20 m、深 50 m。四周均为 PML，没有自由表面。
# - 12 Hz 解析 Ricker，峰值时间 0.10 s；480 点、dt=1.5 ms，最后样点 0.7185 s。源是固定 forcing f，库内部执行 `-v_source² * dt_internal² * U(f)`，不可再预乘 dt 或 v²。振幅使用任意归一化单位，不宣称为 Pa。
# - 固定 `max_vel=2600 m/s` 覆盖整个允许速度区间，确保更新时 CFL 和 PML setup 不变。CFL 由安装版本自己的 planner 计算并保存。空间分辨率仍需单独检查；缩小 dt 不能修复空间色散。
# - 固定最外圈 6 格与顶部 8 格，避免把已知 replicate-forward/crop-backward 的外缘梯度缺项用于可训练参数。固定边界不等于证明其余梯度正确；正常分支仍保留其已记录的 PML 限制。`full` 也不会自动补齐外缘链。
# - float32、单 GPU、默认 CUDA stream、一次 forward 对应一次 backward。不用 AMP、DataParallel、INR 或高阶导数。种子固定，但自定义 CUDA 原子运算不承诺逐位复现。
# 
# 三个 notebook 的绘图采用 cividis 速度色标与 PuOr 正负色标；同类模型共享速度范围，观测/预测/残差共享完整振幅范围，不做按图独立归一化或隐藏截幅。

# %% [markdown]
# ## 学习目标与计算量
# 从不含透镜的平滑背景出发，用普通 PyTorch 网格参数、固定三炮观测和一阶梯度做一个小 FWI。默认 25 次 Adam 更新，每次把 3 炮逐炮累积成同一个全炮平均目标后才更新一次，既不是随机抽炮，也不是每炮一个 optimizer step。
# 
# 默认训练 75 次有梯度 forward/backward，加上 3 次观测正演和 6 组 ×3 炮的评估正演，共 96 次 forward。显存中每次只保留一炮的传播图。实际耗时依赖 GPU 和 native 构建，不给未经实测的秒数承诺。先把 `epochs` 改为 3 验证流程也可以；正式出图再用独立 run 运行 25 次。
# 
# 源子波和背景浅层是已知量；真值只用于合成观测和事后 RMSE 诊断，不参与优化更新、选点或选择最佳模型。最终图展示最后一次更新，不悄悄选取看起来最好的 epoch。改善不是预设结果，失败或变差同样写入指标。

# %%
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib, importlib.metadata, inspect, json, math, platform, random, warnings, zipfile
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import starwave
from starwave.time_sampling import make_time_sampling_plan

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

def as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")

def finite(name, value):
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{name} contains nonfinite values; stop and inspect this run.")

def file_sha256(path):
    h=hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_native_status(raw):
    # Whitelist non-secret metadata; never export full environment or machine paths.
    keys=("library_exists", "library_loaded", "torch_version", "torch_cuda_version", "cuda_available",
          "runtime_cuda_verified", "runtime_dataparallel_verified", "cuda_stream_support", "selected_device_ids", "preparation")
    data={k: raw[k] for k in keys if k in raw}
    p=raw.get("library_path")
    if p:
        data["library_filename"]=Path(p).name
        if Path(p).is_file(): data["library_sha256"]=file_sha256(p)
    return data

def resolve_scalar():
    candidate=starwave.scalar
    route="starwave.scalar"
    if not callable(candidate):
        candidate=getattr(candidate, "scalar", None)
        route="starwave.scalar.scalar (module adapter)"
    if not callable(candidate):
        raise RuntimeError("Cannot resolve StarWave scalar callable. Check the selected kernel/package.")
    signature=inspect.signature(candidate)
    required={"v","grid_spacing","dt","source_amplitudes","source_locations","receiver_locations","max_vel","memory"}
    if not required.issubset(signature.parameters):
        raise RuntimeError(f"Incompatible scalar signature: {signature}")
    return candidate, route, str(signature)

scalar_api, scalar_route, scalar_signature = resolve_scalar()

def start_run(example, cfg):
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg["seed"])
    # No deterministic-algorithm promise for custom CUDA; atomic roundoff can vary.
    root=Path(cfg["output_root"]).expanduser()
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir=root/example/stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    export_config=dict(cfg)
    if Path(export_config["output_root"]).is_absolute(): export_config["output_root"]="<absolute output path omitted>"
    write_json(run_dir/"config.json", export_config)
    try: version=importlib.metadata.version("starwave")
    except importlib.metadata.PackageNotFoundError: version="source checkout (no distribution metadata)"
    mod=importlib.import_module("starwave.scalar")
    module_file=Path(mod.__file__)
    env={"created_utc":stamp, "python":platform.python_version(), "system":platform.system(),
         "machine":platform.machine(), "glibc":list(platform.libc_ver()), "torch":torch.__version__,
         "numpy":np.__version__, "matplotlib":importlib.metadata.version("matplotlib"), "starwave_distribution":version,
         "scalar_adapter":scalar_route, "scalar_signature":scalar_signature,
         "scalar_module_filename":module_file.name, "scalar_module_sha256":file_sha256(module_file),
         "native_before":safe_native_status(starwave.native_status()),
         "experiment_status":"environment_check_only",
         "reproducibility":"seeded inputs; CUDA bitwise reproducibility is not guaranteed"}
    write_json(run_dir/"environment.json",env)
    return run_dir,env

def prepare_device(cfg,run_dir,env):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this kernel. Environment metadata was saved. Select a CUDA PyTorch kernel on your GPU server; there is no CPU propagation fallback.")
    gpu=int(cfg["gpu_id"])
    if not 0 <= gpu < torch.cuda.device_count(): raise ValueError("gpu_id is not a visible logical CUDA index.")
    torch.cuda.set_device(gpu)
    device=torch.device("cuda",gpu)
    props=torch.cuda.get_device_properties(device)
    env["gpu"]={"logical_index":gpu,"name":props.name,"compute_capability":list(torch.cuda.get_device_capability(device)),
                "total_memory_bytes":int(props.total_memory)}
    env["native_prepared"]=safe_native_status(starwave.prepare_native([gpu]))
    write_json(run_dir/"environment.json",env)
    print(json.dumps({"starwave":env["starwave_distribution"],"scalar_adapter":scalar_route,
                      "gpu":env["gpu"],"native":env["native_prepared"]},indent=2))
    return device

def save_figure(fig,run_dir,stem):
    fig.savefig(run_dir/f"{stem}.png",bbox_inches="tight")
    fig.savefig(run_dir/f"{stem}.pdf",bbox_inches="tight")
    plt.show()

def finish_run(run_dir,env,metrics):
    env["experiment_status"]="completed_on_user_device"
    write_json(run_dir/"environment.json",env)
    write_json(run_dir/"metrics.json",metrics)
    files=sorted(p for p in run_dir.iterdir() if p.is_file() and p.name!="artifact_manifest.json")
    write_json(run_dir/"artifact_manifest.json",{p.name:{"bytes":p.stat().st_size,"sha256":file_sha256(p)} for p in files})
    zip_path=run_dir.parent/f"{run_dir.name}.zip"
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(run_dir.iterdir()):
            if p.is_file(): z.write(p,arcname=f"{run_dir.parent.name}/{run_dir.name}/{p.name}")
    print(f"Saved results: {run_dir}\nReturn this ZIP: {zip_path}")

# %%
CONFIG = {
    "seed": 20261005, "gpu_id": 0, "output_root": "starwave_tutorial_results",
    "nx": 96, "nz": 64, "dx_m": 10.0, "dt_s": 0.0015, "nt": 480,
    "frequency_hz": 12.0, "source_peak_time_s": 0.10, "source_scale": 1.0,
    "shot_x_indices": [16, 47, 79], "source_z_index": 5,
    "receiver_x_start": 10, "receiver_x_stop": 86, "receiver_x_step": 2, "receiver_z_index": 5,
    "accuracy": 4, "pml_width": 20, "boundary_buffer": 5, "memory": "boundary",
    "max_vel_m_s": 2600.0, "vmin_m_s": 1450.0, "vmax_m_s": 2450.0,
    "fixed_edge_cells": 6, "fixed_top_cells": 8,
    "background_surface_m_s": 1600.0, "background_gradient_s_inv": 0.55,
    "layer_jump_m_s": 220.0, "layer_depth_m": 350.0, "layer_transition_m": 28.0,
    "lens_amplitude_m_s": 180.0, "lens_x_m": 500.0, "lens_z_m": 260.0,
    "lens_sigma_x_m": 110.0, "lens_sigma_z_m": 65.0,
    "smooth_l1_beta": 0.25,
}

EXAMPLE="03_simple_fwi"
CONFIG.update({"epochs":25,"learning_rate":0.025,"evaluate_every":5,"smoothness_weight":1.0e-4,
               "update_scale_m_s":100.0,"latent_gradient_clip_norm":1.0})

# %%
def analytic_problem(cfg,device):
    nx,nz,dx=cfg["nx"],cfg["nz"],cfg["dx_m"]
    if min(nx,nz) < 24: raise ValueError("This tutorial geometry needs at least 24 cells per axis.")
    x=torch.arange(nx,device=device,dtype=torch.float32)*dx
    z=torch.arange(nz,device=device,dtype=torch.float32)*dx
    X,Z=torch.meshgrid(x,z,indexing="ij")  # model[x_index,z_index]
    initial=cfg["background_surface_m_s"]+cfg["background_gradient_s_inv"]*Z
    initial=initial+cfg["layer_jump_m_s"]*torch.sigmoid((Z-cfg["layer_depth_m"])/cfg["layer_transition_m"])
    active=torch.ones((nx,nz),device=device,dtype=torch.bool)
    edge,top=cfg["fixed_edge_cells"],cfg["fixed_top_cells"]
    if edge < 1 or top < 1 or 2*edge>=min(nx,nz) or top>=nz-edge:
        raise ValueError("Invalid fixed-edge/top width.")
    active[:edge,:]=False;active[-edge:,:]=False;active[:,:max(edge,top)]=False;active[:,-edge:]=False
    lens=cfg["lens_amplitude_m_s"]*torch.exp(-0.5*((X-cfg["lens_x_m"])/cfg["lens_sigma_x_m"])**2
                                          -0.5*((Z-cfg["lens_z_m"])/cfg["lens_sigma_z_m"])**2)
    # Smoothly vanish the analytic anomaly before the fixed band; no edge jump.
    window=torch.sin(math.pi*(X-x[edge])/(x[-edge-1]-x[edge])).clamp_min(0).square()
    window=window*torch.sin(math.pi*(Z-z[max(edge,top)])/(z[-edge-1]-z[max(edge,top)])).clamp_min(0).square()
    truth=initial+torch.where(active,lens*window,torch.zeros_like(lens))
    shot_x=torch.tensor(cfg["shot_x_indices"],device=device,dtype=torch.long)
    rx=torch.arange(cfg["receiver_x_start"],cfg["receiver_x_stop"],cfg["receiver_x_step"],device=device,dtype=torch.long)
    if shot_x.numel()<1 or shot_x.unique().numel()!=shot_x.numel(): raise ValueError("Shot x indices must be nonempty and unique.")
    if rx.numel()<2 or rx.unique().numel()!=rx.numel(): raise ValueError("Receiver x indices must be unique, with at least two receivers.")
    sources=torch.stack((shot_x,torch.full_like(shot_x,cfg["source_z_index"])),dim=-1)[:,None,:]
    receivers=torch.stack((rx,torch.full_like(rx,cfg["receiver_z_index"])),dim=-1)[None].repeat(shot_x.numel(),1,1)
    for locations in (sources,receivers):
        if not bool(((locations[...,0]>=0)&(locations[...,0]<nx)&(locations[...,1]>=0)&(locations[...,1]<nz)).all()):
            raise ValueError("Acquisition coordinate outside model.")
    if bool(active[sources[...,0],sources[...,1]].any()):
        raise ValueError("Tutorial sources must lie in the fixed shallow band.")
    t=torch.arange(cfg["nt"],device=device,dtype=torch.float32)*cfg["dt_s"]
    a=(math.pi*cfg["frequency_hz"]*(t-cfg["source_peak_time_s"]))**2
    ricker=cfg["source_scale"]*(1-2*a)*torch.exp(-a)  # analytic Ricker, no dependency
    wavelet=ricker[None,None,:].repeat(shot_x.numel(),1,1).contiguous()
    for name,v in (("truth",truth),("initial",initial)):
        finite(name,v)
        if not bool(((v>cfg["vmin_m_s"])&(v<cfg["vmax_m_s"])).all()): raise ValueError(f"{name} outside strict velocity bounds.")
    if cfg["max_vel_m_s"] < cfg["vmax_m_s"]: raise ValueError("max_vel must cover all allowed velocities.")
    return x,z,t,truth.contiguous(),initial.contiguous(),active,sources,receivers,wavelet

def propagate(v,source,src,rec,cfg):
    # Public tuple contract: final element has [shot,receiver,time].
    result=scalar_api(v,cfg["dx_m"],cfg["dt_s"],source_amplitudes=source,
        source_locations=src,receiver_locations=rec,accuracy=cfg["accuracy"],
        pml_freq=cfg["frequency_hz"],pml_width=cfg["pml_width"],boundary_buffer=cfg["boundary_buffer"],
        memory=cfg["memory"],max_vel=cfg["max_vel_m_s"],
        freq_taper_frac=0.0,time_pad_frac=0.0,time_taper=False)
    if not isinstance(result,tuple) or len(result)!=1: raise RuntimeError("Unexpected scalar return contract.")
    record=result[-1]
    expected=(source.shape[0],rec.shape[1],source.shape[-1])
    if tuple(record.shape)!=expected: raise RuntimeError(f"Unexpected record shape: {record.shape}, expected {expected}")
    finite("receiver amplitudes",record)
    return record

def problem_diagnostics(cfg,v,src,source):
    plan=make_time_sampling_plan(v,cfg["dx_m"],cfg["dt_s"],cfg["nt"],max_vel=cfg["max_vel_m_s"])
    plan_dict={k:getattr(plan,k) for k in ("user_dt","user_nt","internal_dt","internal_nt","step_ratio","max_vel","max_dt")}
    source_v=v[src[...,0],src[...,1]]
    diag={"cfl":plan_dict,"minimum_cells_per_wavelength_at_peak_frequency":float(v.min())/(cfg["dx_m"]*cfg["frequency_hz"]),
          "source_scale":cfg["source_scale"],"source_increment_coefficient_minus_v2_dt_internal2":as_numpy(-source_v.square()*plan.internal_dt**2).tolist(),
          "source_forcing_units":"arbitrary normalized pressure unit / m^2; not calibrated Pa",
          "record_units":"arbitrary normalized pressure unit", "model_axis_order":"x,z", "record_axis_order":"shot,receiver,time",
          "time_origin":"public sample n is n*dt; no manual shift", "all_sides_pml":True,"free_surface":False,
          "memory_mode":cfg["memory"],"fixed_outer_model_band":True,"water_layer":False}
    print(json.dumps(diag,indent=2))
    return diag

def data_loss(pred,obs,scale,cfg):
    return F.smooth_l1_loss(pred/scale,obs/scale,beta=cfg["smooth_l1_beta"],reduction="mean")

def plot_models(models,titles,x,z,run_dir,stem="models"):
    arrays=[as_numpy(v) for v in models]
    lo=min(a.min() for a in arrays);hi=max(a.max() for a in arrays)
    fig,axes=plt.subplots(1,len(arrays),figsize=(4.5*len(arrays),3.6),layout="constrained",squeeze=False)
    extent=[float(x[0]),float(x[-1]),float(z[-1]),float(z[0])]
    for ax,a,title in zip(axes.ravel(),arrays,titles):
        im=ax.imshow(a.T,extent=extent,cmap="cividis",vmin=lo,vmax=hi,aspect="equal")
        ax.set(title=title,xlabel="x (m)",ylabel="Depth z (m)")
    fig.colorbar(im,ax=axes.ravel().tolist(),label="Velocity (m/s)",shrink=.85)
    save_figure(fig,run_dir,stem)

def plot_gathers(obs,pred,rx_m,t,run_dir,stem="gathers",shot=1):
    shot=min(shot,obs.shape[0]-1)
    arrays=[as_numpy(obs[shot]),as_numpy(pred[shot]),as_numpy(pred[shot]-obs[shot])]
    # Same full-amplitude range for observed, predicted, residual: no hidden clipping.
    lim=max(float(np.abs(a).max()) for a in arrays) or 1.0
    fig,axes=plt.subplots(1,3,figsize=(13,4.1),layout="constrained")
    for ax,a,title in zip(axes,arrays,["Observed","Predicted","Residual: predicted - observed"]):
        im=ax.imshow(a.T,extent=[float(rx_m[0]),float(rx_m[-1]),float(t[-1]),float(t[0])],
                     aspect="auto",cmap="PuOr",vmin=-lim,vmax=lim)
        ax.set(title=f"{title}, shot {shot}",xlabel="Receiver x (m)",ylabel="Time (s)")
    fig.colorbar(im,ax=axes,label="Amplitude (a.u.; shared scale)")
    save_figure(fig,run_dir,stem)

# %%
run_dir,environment=start_run(EXAMPLE,CONFIG)
device=prepare_device(CONFIG,run_dir,environment)
x,z,t,v_true,v_initial,active,source_locations,receiver_locations,source_amplitudes=analytic_problem(CONFIG,device)
NSHOTS=source_amplitudes.shape[0]
diagnostics=problem_diagnostics(CONFIG,v_true,source_locations,source_amplitudes)
write_json(run_dir/"diagnostics.json",diagnostics)
np.savez_compressed(run_dir/"inputs.npz",x_m=as_numpy(x),z_m=as_numpy(z),time_s=as_numpy(t),
    true_model_m_s=as_numpy(v_true),initial_model_m_s=as_numpy(v_initial),active_mask=as_numpy(active),
    source_locations=as_numpy(source_locations),receiver_locations=as_numpy(receiver_locations),
    source_amplitudes=as_numpy(source_amplitudes))
print(f"Experiment: {EXAMPLE}; {NSHOTS} shots; output: {run_dir}")

# %% [markdown]
# ## 1. 有界参数化与目标函数
# 每格的潜变量 θ 经 sigmoid 映射到 `(1450,2450) m/s`；固定边带直接取初始背景。这样每次更新都满足物理速度边界，且 2600 m/s 的固定 CFL 上界始终有效。这里没有水层；未来若加入水层，需要把真实水层显式加入固定 mask。
# 
# 数据项是预测/观测共同除以一次性全观测 RMS 后的 SmoothL1 均值（beta=0.25）；显式小权重平滑项仅惩罚相对初始模型的速度改变量的一阶网格差分。它是教学正则项，不是传播器的隐式处理。对于归一化残差 r，逐样点损失是 |r|<beta 时 0.5*r²/beta，否则 |r|-0.5*beta。它等于 Huber(delta=beta)/beta，与未缩放的 HuberLoss 数值不同。
# 
# 梯度裁剪作用于 θ 的全局范数，阈值 1.0，裁剪前范数会保存；不自动修改学习率，也不替换 NaN。

# %%
class BoundedVelocity(torch.nn.Module):
    def __init__(self,initial,mask,cfg):
        super().__init__()
        self.register_buffer("initial",initial.detach().clone())
        self.register_buffer("active",mask.clone())
        self.low=float(cfg["vmin_m_s"]);self.high=float(cfg["vmax_m_s"])
        fraction=(initial-self.low)/(self.high-self.low)
        if not bool(((fraction>0)&(fraction<1)).all()): raise ValueError("Initial model must lie strictly inside sigmoid bounds.")
        self.theta=torch.nn.Parameter(torch.logit(fraction).detach().clone())
    def forward(self):
        physical=self.low+(self.high-self.low)*torch.sigmoid(self.theta)
        return torch.where(self.active,physical,self.initial)

def smoothness(model,initial,cfg):
    update=(model-initial)/cfg["update_scale_m_s"]
    return (update[1:,:]-update[:-1,:]).square().mean()+(update[:,1:]-update[:,:-1]).square().mean()

velocity=BoundedVelocity(v_initial,active,CONFIG).to(device)
optimizer=torch.optim.Adam(velocity.parameters(),lr=CONFIG["learning_rate"])
with torch.no_grad():
    observed=torch.cat([propagate(v_true,source_amplitudes[s:s+1],source_locations[s:s+1],receiver_locations[s:s+1],CONFIG)
                       for s in range(NSHOTS)],dim=0)
scale=observed.square().mean().sqrt().detach()
if not float(scale)>0: raise RuntimeError("Observed RMS must be nonzero.")
history=[];training_history=[];snapshots=[]

@torch.no_grad()
def evaluate(iteration):
    model=velocity()
    records=torch.cat([propagate(model,source_amplitudes[s:s+1],source_locations[s:s+1],receiver_locations[s:s+1],CONFIG)
                       for s in range(NSHOTS)],dim=0)
    data=float(data_loss(records,observed,scale,CONFIG))
    regularization=float(smoothness(model,v_initial,CONFIG))
    row={"iteration":int(iteration),"data_loss":data,"unweighted_smoothness":regularization,
         "total_objective":data+CONFIG["smoothness_weight"]*regularization,
         "relative_data_l2":float((records-observed).norm()/observed.norm()),
         "active_model_rmse_m_s":float((model[active]-v_true[active]).square().mean().sqrt()),
         "velocity_min_m_s":float(model.min()),"velocity_max_m_s":float(model.max())}
    history.append(row);snapshots.append(as_numpy(model).copy())
    write_json(run_dir/"history.json",{"evaluations":history,"training_pre_update":training_history})
    np.savez_compressed(run_dir/"latest_model_snapshot.npz",iteration=np.array(iteration),model_m_s=as_numpy(model),
        latent_theta=as_numpy(velocity.theta),observed=as_numpy(observed),predicted=as_numpy(records))
    print(f"iteration {iteration:3d} | objective={row['total_objective']:.6g} | data rel L2={row['relative_data_l2']:.4g} | active RMSE={row['active_model_rmse_m_s']:.2f} m/s")
    return records

initial_predicted=evaluate(0)

# %% [markdown]
# ## 2. 逐炮累积梯度并更新
# `/NSHOTS` 使逐炮反传严格对应全部等尺寸数据的全局平均。每炮重新建立物理模型与传播图；不能对同一次 native forward 多次 backward。
# 
# 评估在第 0、5、10、15、20、25 次更新后执行，损失与模型快照对应同一个状态。训练日志另外标明更新前的目标；两者不混用。`latest_model_snapshot.npz` 只用于诊断，未保存 Adam 动量，不是可精确续跑的 checkpoint。若发现非有限值，立即停止并保留已有文件，不吞掉错误继续画图。

# %%
if CONFIG["epochs"]<1 or CONFIG["evaluate_every"]<1: raise ValueError("epochs and evaluate_every must be positive.")
for iteration in range(1,CONFIG["epochs"]+1):
    optimizer.zero_grad(set_to_none=True)
    data_total=0.0
    for s in range(NSHOTS):
        model=velocity()
        pred=propagate(model,source_amplitudes[s:s+1],source_locations[s:s+1],receiver_locations[s:s+1],CONFIG)
        term=data_loss(pred,observed[s:s+1],scale,CONFIG)/NSHOTS
        finite("training data loss",term)
        term.backward()
        data_total+=float(term.detach())
    penalty=smoothness(velocity(),v_initial,CONFIG)
    weighted_penalty=CONFIG["smoothness_weight"]*penalty
    weighted_penalty.backward()
    if velocity.theta.grad is None: raise RuntimeError("Missing latent gradient.")
    finite("latent gradient",velocity.theta.grad)
    assert bool((velocity.theta.grad[~active]==0).all())
    grad_norm=torch.nn.utils.clip_grad_norm_(velocity.parameters(),CONFIG["latent_gradient_clip_norm"],error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        current=velocity()
        finite("updated velocity",current)
        if not bool(((current>=CONFIG["vmin_m_s"])&(current<=CONFIG["vmax_m_s"])).all()): raise RuntimeError("Velocity bounds failed.")
        assert torch.equal(current[~active],v_initial[~active])
    training_history.append({"update":iteration,"model_state_iteration":iteration-1,"data_loss":data_total,
        "unweighted_smoothness":float(penalty.detach()),"total_objective":data_total+float(weighted_penalty.detach()),
        "latent_gradient_norm_before_clip":float(grad_norm),"clip_threshold":CONFIG["latent_gradient_clip_norm"]})
    write_json(run_dir/"history.json",{"evaluations":history,"training_pre_update":training_history})
    if iteration%CONFIG["evaluate_every"]==0 or iteration==CONFIG["epochs"]:
        final_predicted=evaluate(iteration)
v_final=velocity().detach()
torch.cuda.synchronize(device)
metrics={"status":"optimization_completed_not_convergence_certified","updates":CONFIG["epochs"],"shots":NSHOTS,
    "observed_rms":float(scale),"initial":history[0],"final":history[-1],
    "data_loss_decreased":bool(history[-1]["data_loss"]<history[0]["data_loss"]),
    "active_model_rmse_decreased":bool(history[-1]["active_model_rmse_m_s"]<history[0]["active_model_rmse_m_s"]),
    "fixed_band_max_change_m_s":float((v_final[~active]-v_initial[~active]).abs().max()),
    "synthetic_same_solver_data":True,"final_is_last_iterate":True,"gpu_full_acceptance":False}
print(json.dumps(metrics,indent=2))

# %% [markdown]
# ## 3. 如实查看模型、残差与历史
# 主模型图统一速度色标。误差图统一零对称完整范围；初始和最终 gather 与残差也统一完整振幅范围。数据 loss 减小并不自动意味着模型 RMSE 减小，有限采集和有限迭代都会限制可恢复性。下面把两条指标同时展示。

# %%
plot_models([v_true,v_initial,v_final],["Analytic truth","Initial background","Final: last update"],x,z,run_dir,"models_truth_initial_final")
fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout="constrained")
errors=[as_numpy(v_initial-v_true),as_numpy(v_final-v_true)]
lim=max(float(abs(a).max()) for a in errors) or 1.0
for ax,a,title in zip(axes,errors,["Initial - truth","Final - truth"]):
    im=ax.imshow(a.T,extent=[float(x[0]),float(x[-1]),float(z[-1]),0],cmap="PuOr",vmin=-lim,vmax=lim)
    ax.set(title=title,xlabel="x (m)",ylabel="Depth z (m)")
fig.colorbar(im,ax=axes,label="Velocity error (m/s; shared scale)")
save_figure(fig,run_dir,"model_errors")

fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout="constrained")
steps=[h["iteration"] for h in history]
axes[0].plot(steps,[h["data_loss"] for h in history],"o-",color="#127C86",label="Data SmoothL1")
axes[0].plot(steps,[h["total_objective"] for h in history],"s--",color="#AD5E10",label="Data + regularization")
axes[0].set(title="Measured objective",xlabel="Completed optimizer updates",ylabel="Normalized loss");axes[0].legend()
axes[1].plot(steps,[h["active_model_rmse_m_s"] for h in history],"o-",color="#7D4E91")
axes[1].set(title="Model error (diagnostic only)",xlabel="Completed optimizer updates",ylabel="Active-cell RMSE (m/s)")
save_figure(fig,run_dir,"optimization_history")

s=NSHOTS//2
arrays=[as_numpy(observed[s]),as_numpy(initial_predicted[s]),as_numpy(final_predicted[s]),
        as_numpy(initial_predicted[s]-observed[s]),as_numpy(final_predicted[s]-observed[s])]
lim=max(float(abs(a).max()) for a in arrays) or 1.0
fig,axes=plt.subplots(1,5,figsize=(18,4),layout="constrained")
rx_m=receiver_locations[0,:,0]*CONFIG["dx_m"]
for ax,a,title in zip(axes,arrays,["Observed","Initial predicted","Final predicted","Initial residual","Final residual"]):
    im=ax.imshow(a.T,extent=[float(rx_m[0]),float(rx_m[-1]),float(t[-1]),0],aspect="auto",cmap="PuOr",vmin=-lim,vmax=lim)
    ax.set(title=title,xlabel="Receiver x (m)",ylabel="Time (s)")
fig.colorbar(im,ax=axes,label="Amplitude (a.u.; one shared full scale)",shrink=.8)
save_figure(fig,run_dir,"gathers_before_after")

fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout="constrained")
r=observed.shape[1]//2
for a,label,color,style in ((observed,"Observed","#127C86","-"),(initial_predicted,"Initial","#AD5E10","--"),(final_predicted,"Final","#7D4E91","-.")):
    axes[0].plot(as_numpy(t),as_numpy(a[s,r]),style,label=label,color=color)
axes[0].set(title=f"Unscaled trace: shot {s}, receiver {r}",xlabel="Time (s)",ylabel="Amplitude (a.u.)");axes[0].legend()
ix=int(torch.argmin(abs(x-CONFIG["lens_x_m"])))
for a,label,color,style in ((v_true,"Truth","#127C86","-"),(v_initial,"Initial","#AD5E10","--"),(v_final,"Final","#7D4E91","-.")):
    axes[1].plot(as_numpy(a[ix]),as_numpy(z),style,label=label,color=color)
axes[1].invert_yaxis();axes[1].set(title=f"Velocity profile at x={float(x[ix]):.0f} m",xlabel="Velocity (m/s)",ylabel="Depth z (m)");axes[1].legend()
save_figure(fig,run_dir,"trace_velocity_profile")

# %%
np.savez_compressed(run_dir/"fwi_results.npz",true_model_m_s=as_numpy(v_true),initial_model_m_s=as_numpy(v_initial),
    final_model_m_s=as_numpy(v_final),observed=as_numpy(observed),initial_predicted=as_numpy(initial_predicted),
    final_predicted=as_numpy(final_predicted),initial_residual=as_numpy(initial_predicted-observed),
    final_residual=as_numpy(final_predicted-observed),evaluation_iterations=np.asarray(steps,dtype=np.int64),
    model_snapshots_m_s=np.stack(snapshots),data_loss=np.asarray([h["data_loss"] for h in history]),
    total_objective=np.asarray([h["total_objective"] for h in history]),active_model_rmse_m_s=np.asarray([h["active_model_rmse_m_s"] for h in history]),
    training_pre_update_loss=np.asarray([h["total_objective"] for h in training_history]),
    latent_gradient_norm_before_clip=np.asarray([h["latent_gradient_norm_before_clip"] for h in training_history]))
finish_run(run_dir,environment,metrics)

# %% [markdown]
# ## 可控扩展
# 1. 先比较 3 次与 25 次更新的独立 run，而不是随意更换很多参数。
# 2. 之后再增加炮数、延长时窗或增加迭代，并保存新 config；这些都会提高成本。
# 3. 若收敛差，先检查源、坐标、CFL、梯度方向、初始模型与已知 native 限制。增加迭代无法替代梯度正确性检查。
# 4. 这个 notebook 仅生成教学结果，不修改 StarWave 源码、不声明修复 PML，也不自动发布网页。

# %% [markdown]
# ## 保存结果与复现实验
# 最后一格自动打包本次运行目录，路径会打印出来。请保存生成的 ZIP 和执行后的 notebook（含输出），以便复查本次运行。
# 
# 每次运行使用独立 UTC 时间戳，不覆盖旧实验。包里包含配置/环境/CFL JSON、输入与结果 NPZ、PNG/PDF 图件和文件哈希。只收集白名单软件/GPU/后端元数据，不保存用户名、主机名、完整环境变量、token 或机器目录。输出本身使用相对路径。运行前可以检查 helper 中的导出字段；分享运行包前仍应自行检查内容。
# 
# 如果出现异常，保留已写出的目录和报错，不要改图或删除失败点。重启 kernel 后从头运行；不要从中间复用上一轮有计算图的 Tensor。解释结果时应区分学习示例、数值诊断与未完成的 GPU 验收。

