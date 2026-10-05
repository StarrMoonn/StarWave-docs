# -*- coding: utf-8 -*-
# Paired executable script exported from the public notebook.

# %% [markdown]
# # 02 · Forward modelling & a velocity gradient
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
# ## 1. 配置与准备
# 先用 01 确认 CUDA 环境。下面从零生成同一小型解析模型，不读取 01 的输出。默认主流程包括 3 次观测正演和 3 次带梯度正演；每炮单独释放计算图，降低显存占用。
# 
# 速度梯度针对归一化 SmoothL1（Huber 家族）数据目标。观测 RMS 仅计算一次并固定；不是分别归一化每条预测/观测。梯度图表示增大速度对目标的局部一阶影响，不直接等于最终速度修正。

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

EXAMPLE="02_forward_gradient"
CONFIG["run_directional_check"]=False  # optional: 1 extra backward + 6 no-grad forwards

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

# %%
plot_models([v_true,v_initial],["Analytic truth","Starting background"],x,z,run_dir)
fig,axes=plt.subplots(1,2,figsize=(10,3.4),layout="constrained")
axes[0].imshow(as_numpy(active).T,extent=[float(x[0]),float(x[-1]),float(z[-1]),0],cmap="Greys",vmin=0,vmax=1)
axes[0].scatter(as_numpy(source_locations[:,0,0])*CONFIG["dx_m"],as_numpy(source_locations[:,0,1])*CONFIG["dx_m"],marker="*",s=90,c="#AD5E10",label="Sources")
axes[0].scatter(as_numpy(receiver_locations[0,:,0])*CONFIG["dx_m"],as_numpy(receiver_locations[0,:,1])*CONFIG["dx_m"],s=8,c="#127C86",label="Receivers")
axes[0].set(title="Active cells and acquisition",xlabel="x (m)",ylabel="Depth z (m)"); axes[0].legend(fontsize=8)
axes[1].plot(as_numpy(t),as_numpy(source_amplitudes[0,0]),color="#127C86")
axes[1].set(title="Analytic fixed forcing",xlabel="Time (s)",ylabel="Forcing (a.u. / m²)")
save_figure(fig,run_dir,"acquisition_source")

# %% [markdown]
# ## 2. 生成观测，然后求一阶梯度
# `v_parameter` 是叶子张量；`torch.where` 保留活动区的梯度，固定区始终取背景。不能把传播所需的模型 detach，也不能把梯度接到之前绘图用的 NumPy 数组。
# 
# 观测和预测使用同一个 solver 与网格，是有意的理想化合成实验。它排除了实测噪声、源子波不确定性和建模误差，也因此不代表真实数据表现。

# %%
with torch.no_grad():
    observed=torch.cat([propagate(v_true,source_amplitudes[s:s+1],source_locations[s:s+1],receiver_locations[s:s+1],CONFIG)
                        for s in range(NSHOTS)],dim=0)
scale=observed.square().mean().sqrt().detach()
if not float(scale)>0: raise RuntimeError("Observed RMS must be nonzero.")
v_parameter=torch.nn.Parameter(v_initial.clone())
predictions=[];loss_value=0.0
for s in range(NSHOTS):
    model=torch.where(active,v_parameter,v_initial)
    pred=propagate(model,source_amplitudes[s:s+1],source_locations[s:s+1],receiver_locations[s:s+1],CONFIG)
    term=data_loss(pred,observed[s:s+1],scale,CONFIG)/NSHOTS
    term.backward()  # once for this forward; accumulates exact global shot mean
    loss_value+=float(term.detach());predictions.append(pred.detach())
predicted=torch.cat(predictions,dim=0)
gradient=v_parameter.grad.detach().clone()
finite("velocity gradient",gradient)
if not float(gradient[active].norm())>0: raise RuntimeError("Active velocity gradient is zero.")
assert bool((gradient[~active]==0).all())
metrics={"status":"forward_gradient_completed","normalized_smooth_l1_loss":loss_value,"observed_rms":float(scale),
         "relative_data_l2":float((predicted-observed).norm()/observed.norm()),
         "gradient_l2":float(gradient.norm()),"gradient_active_abs_max":float(gradient[active].abs().max()),
         "fixed_band_gradient_abs_max":float(gradient[~active].abs().max()),
         "directional_check":{"status":"not_requested"},"full_adjoint_validation":False}
print(json.dumps(metrics,indent=2))

# %%
plot_gathers(observed,predicted,receiver_locations[0,:,0]*CONFIG["dx_m"],t,run_dir)
fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout="constrained")
g=as_numpy(gradient);glim=float(abs(g).max())
im=axes[0].imshow(g.T,extent=[float(x[0]),float(x[-1]),float(z[-1]),0],cmap="PuOr",vmin=-glim,vmax=glim)
axes[0].set(title="Velocity gradient (active cells)",xlabel="x (m)",ylabel="Depth z (m)")
fig.colorbar(im,ax=axes[0],label="d normalized loss / d velocity")
s=NSHOTS//2;r=observed.shape[1]//2
axes[1].plot(as_numpy(t),as_numpy(observed[s,r]),label="Observed",color="#127C86")
axes[1].plot(as_numpy(t),as_numpy(predicted[s,r]),"--",label="Initial",color="#AD5E10")
axes[1].plot(as_numpy(t),as_numpy(predicted[s,r]-observed[s,r]),label="Residual",color="#7D4E91",alpha=.8)
axes[1].set(title=f"Unscaled trace: shot {s}, receiver {r}",xlabel="Time (s)",ylabel="Amplitude (a.u.)");axes[1].legend()
save_figure(fig,run_dir,"gradient_trace")

# %% [markdown]
# ## 3. 可选：一个内部方向的中心差分诊断
# 把上方 `run_directional_check` 改为 True 并从头运行。这里只选中间一炮和一个平滑、固定边界为零的方向，比较 AD 方向导数与 ±0.5、±1、±2 m/s 的中心差分，额外 1 次 backward + 6 次无梯度 forward。
# 
# 使用固定 max_vel、同一源、同一目标和同一缩放。float32 的差分会受消减误差影响，步长太大又有截断误差；看步长序列，不挑最好的一点称作全局通过。2% 仅为本教学诊断的预先提示阈值，超出时保存原结果并调查，不放宽阈值。它不覆盖全部方向、PML、full/boundary 等价性或源梯度。

# %%
if CONFIG["run_directional_check"]:
    s=NSHOTS//2
    q=v_initial.clone().requires_grad_(True)
    q_model=torch.where(active,q,v_initial)
    q_record=propagate(q_model,source_amplitudes[s:s+1],source_locations[s:s+1],receiver_locations[s:s+1],CONFIG)
    q_loss=data_loss(q_record,observed[s:s+1],scale,CONFIG)
    q_loss.backward()
    finite("directional-check gradient",q.grad)
    X,Z=torch.meshgrid(x,z,indexing="ij")
    direction=torch.exp(-0.5*((X-480)/130)**2-0.5*((Z-270)/80)**2)*active
    direction=direction/direction.abs().max()  # max perturbation is epsilon m/s
    ad=float((q.grad.double()*direction.double()).sum())
    rows=[]
    for epsilon in (0.5,1.0,2.0):
        values=[]
        with torch.no_grad():
            for sign in (-1,1):
                perturbed=v_initial+sign*epsilon*direction
                rec=propagate(perturbed,source_amplitudes[s:s+1],source_locations[s:s+1],receiver_locations[s:s+1],CONFIG)
                values.append(float(data_loss(rec,observed[s:s+1],scale,CONFIG)))
        fd=(values[1]-values[0])/(2*epsilon)
        relative=abs(fd-ad)/max(abs(fd),abs(ad),1e-12)
        rows.append({"epsilon_m_s":epsilon,"ad":ad,"central_difference":fd,"relative_error":relative})
    metrics["directional_check"]={"status":"diagnostic_measured","shot":s,"warning_threshold":0.02,"results":rows}
    print(json.dumps(metrics["directional_check"],indent=2))
    if any(row["relative_error"]>0.02 for row in rows):
        warnings.warn("Directional diagnostic exceeds 2% at one or more steps. Keep the measured results; investigate FP32, steps, and known backend limits before using the gradient as validation evidence.")

# %%
np.savez_compressed(run_dir/"forward_gradient.npz",observed=as_numpy(observed),predicted=as_numpy(predicted),
    residual=as_numpy(predicted-observed),velocity_gradient=as_numpy(gradient),observed_rms=np.array(float(scale)))
torch.cuda.synchronize(device)
finish_run(run_dir,environment,metrics)

# %% [markdown]
# ## 保存结果与复现实验
# 最后一格自动打包本次运行目录，路径会打印出来。请保存生成的 ZIP 和执行后的 notebook（含输出），以便复查本次运行。
# 
# 每次运行使用独立 UTC 时间戳，不覆盖旧实验。包里包含配置/环境/CFL JSON、输入与结果 NPZ、PNG/PDF 图件和文件哈希。只收集白名单软件/GPU/后端元数据，不保存用户名、主机名、完整环境变量、token 或机器目录。输出本身使用相对路径。运行前可以检查 helper 中的导出字段；分享运行包前仍应自行检查内容。
# 
# 如果出现异常，保留已写出的目录和报错，不要改图或删除失败点。重启 kernel 后从头运行；不要从中间复用上一轮有计算图的 Tensor。解释结果时应区分学习示例、数值诊断与未完成的 GPU 验收。

