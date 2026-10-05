# -*- coding: utf-8 -*-
# Paired executable script exported from the public notebook.

# %% [markdown]
# # 01 · Installation & environment smoke test
# 
# 这是一套为 StarWave 独立设计的教学实验。只用解析模型、NumPy、Matplotlib、PyTorch 和 StarWave；无需外部地震数据、Deepwave 包或项目 utils。三份 notebook 都可单独从头运行。
# 
# 这是原始教学 notebook 的公开清理版：仅调整说明文字、清除输出与执行元数据，并从环境记录 helper 中删除不必要的开发来源字段。其余代码单元（数值模型、配置、传播、梯度、优化、绘图与结果导出）与原始交付逐字一致。原始与公开文件的 SHA-256 及逐单元核验范围见 [source_manifest.json](source_manifest.json)。
# 
# 原始实验后来在 NVIDIA A30 / PyTorch 2.5.1 / CUDA 11.8 / StarWave 0.1.0.dev9 环境实际运行；整理后的记录见 [results_summary.json](results_summary.json)。这些记录不证明公开 2.0.0 wheel 通过验证。此公开清理版没有重新执行，不能称为这些文件逐字节运行的结果。学习目标是读懂调用与梯度链；微型合成案例不构成现场 FWI、完整伴随、收敛或多 GPU 验收。
# 
# Public edition: both documentation languages download these same original Chinese-language teaching notebooks. Only privacy/editorial changes were made; numerical experiment cells are unchanged. This public edition has not been run byte-for-byte. Measured results came from the original notebooks on StarWave 0.1.0.dev9, not a tested public 2.0.0 wheel. See the linked source manifest and results summary for provenance and limits.

# %% [markdown]
# ## 安装方式：只在独立终端执行
# 已有可用 StarWave 环境可直接选相应 Jupyter kernel；不要在正在运行的 kernel 中升级包。以下提供公开 wheel 安装参考。
# 
# ### 公开二进制 wheel
# 下面是 PyTorch 2.5.1 / CUDA 11.8 配套环境的安装示例。先按服务器驱动和管理员要求选择兼容的 CUDA PyTorch；不要覆盖正在使用的环境。
# 
# ```bash
# python3.10 -m venv .venv-starwave-tutorial
# source .venv-starwave-tutorial/bin/activate
# python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
# python -m pip install starwave==2.0.0 numpy "matplotlib>=3.6" jupyterlab ipykernel
# python -m ipykernel install --user --name starwave-tutorial --display-name "StarWave tutorial (CUDA)"
# python -m jupyter lab
# ```
# 
# 2.0.0 已发布 wheel 的平台标签是 manylinux_2_35_x86_64，面向 Linux x86_64 / glibc ≥ 2.35；macOS、Windows 或更旧 glibc 不适用此 wheel。发行配套范围为 Python 3.10–3.12 / PyTorch 2.5.x；系统 libstdc++ 需提供 GLIBCXX_3.4.30 / CXXABI_1.3.13 或更新版本。CUDA 11.8 构建采用保守 Linux 驱动目标 520.61.05+，还需支持实际 GPU。包导入成功不等于 GPU 数值已通过。参考：[PyPI 2.0.0](https://pypi.org/project/starwave/2.0.0/)、[PyTorch 2.5.1 安装](https://pytorch.org/get-started/previous-versions/#v251)。
# 
# 安装后重启 kernel。Notebook 不自动安装或编译，不修改 CUDA_VISIBLE_DEVICES。这里的公开 wheel 安装说明不代表已用它复现下方记录；已记录的运行环境为 0.1.0.dev9。

# %% [markdown]
# ## 1. 查看环境，再做一个很小的正演与反传
# 下面没有安装命令。首次运行会记录包版本、公开 scalar 签名、实际 module/native 文件哈希和 CUDA 可用性。`native_status` / `prepare_native` 仅检查存在和加载，不是 GPU 数值认证。
# 
# 微型案例使用 48×40 均匀模型、单炮、12 个接收点、160 个样点。目标只是确认 `[B,R,T]` 记录、非零有限信号，以及一个一阶速度梯度能走通；不把它当作完整伴随或 FWI 验收。

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
EXAMPLE="01_installation"
CONFIG={"seed":20261005,"gpu_id":0,"output_root":"starwave_tutorial_results",
        "nx":48,"nz":40,"dx_m":10.0,"dt_s":0.001,"nt":160,"velocity_m_s":1800.0,
        "frequency_hz":15.0,"source_peak_time_s":0.055,"source_scale":1.0,
        "accuracy":4,"pml_width":12,"boundary_buffer":5,"memory":"boundary","max_vel_m_s":2200.0}
run_dir,environment=start_run(EXAMPLE,CONFIG)
device=prepare_device(CONFIG,run_dir,environment)

# %%
x=torch.arange(CONFIG["nx"],device=device,dtype=torch.float32)*CONFIG["dx_m"]
z=torch.arange(CONFIG["nz"],device=device,dtype=torch.float32)*CONFIG["dx_m"]
t=torch.arange(CONFIG["nt"],device=device,dtype=torch.float32)*CONFIG["dt_s"]
v0=torch.full((CONFIG["nx"],CONFIG["nz"]),CONFIG["velocity_m_s"],device=device,dtype=torch.float32)
active=torch.zeros_like(v0,dtype=torch.bool); active[4:-4,8:-4]=True
parameter=torch.nn.Parameter(v0.clone())
v=torch.where(active,parameter,v0)  # outer and source cells stay fixed
source_locations=torch.tensor([[[24,5]]],device=device,dtype=torch.long)
rx=torch.arange(8,44,3,device=device,dtype=torch.long)
receiver_locations=torch.stack((rx,torch.full_like(rx,5)),dim=-1)[None]
a=(math.pi*CONFIG["frequency_hz"]*(t-CONFIG["source_peak_time_s"]))**2
source_amplitudes=(CONFIG["source_scale"]*(1-2*a)*torch.exp(-a))[None,None,:].contiguous()
plan=make_time_sampling_plan(v,CONFIG["dx_m"],CONFIG["dt_s"],CONFIG["nt"],max_vel=CONFIG["max_vel_m_s"])
diagnostics={"cfl":{k:getattr(plan,k) for k in ("user_dt","user_nt","internal_dt","internal_nt","step_ratio","max_vel","max_dt")},
             "source_semantics":"fixed f; StarWave applies -v_source^2 * dt_internal^2; no manual scaling",
             "native_source_coefficient":-CONFIG["velocity_m_s"]**2*plan.internal_dt**2,
             "minimum_cells_per_wavelength_at_peak_frequency":CONFIG["velocity_m_s"]/CONFIG["dx_m"]/CONFIG["frequency_hz"],
             "model_axis_order":"x,z","record_axis_order":"shot,receiver,time","free_surface":False}
write_json(run_dir/"diagnostics.json",diagnostics)
print(json.dumps(diagnostics,indent=2))
result=scalar_api(v,CONFIG["dx_m"],CONFIG["dt_s"],source_amplitudes=source_amplitudes,
    source_locations=source_locations,receiver_locations=receiver_locations,accuracy=CONFIG["accuracy"],
    pml_freq=CONFIG["frequency_hz"],pml_width=CONFIG["pml_width"],boundary_buffer=CONFIG["boundary_buffer"],
    memory=CONFIG["memory"],max_vel=CONFIG["max_vel_m_s"])
if not isinstance(result,tuple) or len(result)!=1: raise RuntimeError("Expected a one-element scalar tuple.")
record=result[-1]
assert tuple(record.shape)==(1,len(rx),CONFIG["nt"])
finite("smoke record",record)
if not float(record.detach().abs().max())>0: raise RuntimeError("Smoke trace is identically zero.")
objective=record.square().mean()  # arbitrary differentiable scalar, not an FWI result
objective.backward()
if parameter.grad is None: raise RuntimeError("Velocity gradient is missing.")
finite("smoke velocity gradient",parameter.grad)
if not float(parameter.grad[active].abs().max())>0: raise RuntimeError("Interior smoke gradient is zero.")
assert bool((parameter.grad[~active]==0).all())
torch.cuda.synchronize(device)
metrics={"status":"passed_small_smoke_only","record_shape":list(record.shape),
         "record_abs_max":float(record.detach().abs().max()),"record_rms":float(record.detach().square().mean().sqrt()),
         "objective":float(objective.detach()),"gradient_l2":float(parameter.grad.norm()),
         "gradient_active_abs_max":float(parameter.grad[active].abs().max()),"fixed_band_gradient_abs_max":float(parameter.grad[~active].abs().max()),
         "full_gradient_validation":False,"fwi_convergence_validation":False,"multi_gpu_tested":False}
print(json.dumps(metrics,indent=2))
np.savez_compressed(run_dir/"smoke.npz",x_m=as_numpy(x),z_m=as_numpy(z),time_s=as_numpy(t),model_m_s=as_numpy(v),
    source_amplitudes=as_numpy(source_amplitudes),source_locations=as_numpy(source_locations),
    receiver_locations=as_numpy(receiver_locations),records=as_numpy(record),velocity_gradient=as_numpy(parameter.grad),active_mask=as_numpy(active))

# %%
fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout="constrained")
axes[0].plot(as_numpy(t),as_numpy(source_amplitudes[0,0]),color="#127C86")
axes[0].set(title="Fixed source forcing",xlabel="Time (s)",ylabel="Forcing (a.u. / m²)")
lim=float(record.detach().abs().max())
im=axes[1].imshow(as_numpy(record[0]).T,extent=[float(rx[0])*CONFIG["dx_m"],float(rx[-1])*CONFIG["dx_m"],float(t[-1]),0],
    aspect="auto",cmap="PuOr",vmin=-lim,vmax=lim)
axes[1].set(title="Smoke receiver gather",xlabel="Receiver x (m)",ylabel="Time (s)")
fig.colorbar(im,ax=axes[1],label="Amplitude (a.u.)")
g=as_numpy(parameter.grad);glim=float(abs(g).max())
im=axes[2].imshow(g.T,extent=[float(x[0]),float(x[-1]),float(z[-1]),0],cmap="PuOr",vmin=-glim,vmax=glim)
axes[2].set(title="Gradient of smoke objective",xlabel="x (m)",ylabel="Depth z (m)")
fig.colorbar(im,ax=axes[2],label="d objective / d velocity")
save_figure(fig,run_dir,"smoke_overview")
finish_run(run_dir,environment,metrics)

# %% [markdown]
# ## 常见阻塞
# - `CUDA is unavailable`：所选 kernel 是 CPU PyTorch，或 GPU/驱动不可见。先在同一个环境检查 `torch.cuda.is_available()`；不自动切换到 CPU 传播。
# - native 缺失/指纹或 ABI 不匹配：核对公开 wheel 所需的平台、驱动和安装包，重启 kernel；不手动混用其他 `.so`。
# - float32、设备、坐标或 stream 错误：以报错定位输入，不绕过库检查。
# - 非有限记录或梯度：保留配置与报错，先复现这个微型案例；不要用 NaN 替换、自动调学习率或改验收阈值掩盖问题。
# 
# 只有本 notebook 完整跑通，再运行 02 和 03。这里不要求管理员权限，也不会发起收费 GPU 任务。

# %% [markdown]
# ## 保存结果与复现实验
# 最后一格自动打包本次运行目录，路径会打印出来。请保存生成的 ZIP 和执行后的 notebook（含输出），以便复查本次运行。
# 
# 每次运行使用独立 UTC 时间戳，不覆盖旧实验。包里包含配置/环境/CFL JSON、输入与结果 NPZ、PNG/PDF 图件和文件哈希。只收集白名单软件/GPU/后端元数据，不保存用户名、主机名、完整环境变量、token 或机器目录。输出本身使用相对路径。运行前可以检查 helper 中的导出字段；分享运行包前仍应自行检查内容。
# 
# 如果出现异常，保留已写出的目录和报错，不要改图或删除失败点。重启 kernel 后从头运行；不要从中间复用上一轮有计算图的 Tensor。解释结果时应区分学习示例、数值诊断与未完成的 GPU 验收。

