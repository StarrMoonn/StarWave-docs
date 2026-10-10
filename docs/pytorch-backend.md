(pytorch-backend)=
# PyTorch 后端

V16 的统一新增后端通过原有 `starwave.scalar / vrz / vti / elastic / visco_gsls` 入口选择。使用 `backend="torch"`，正演由 PyTorch 张量运算执行，反传由 autograd 生成；`execution="compile"` 编译同一张量计算。旧 CUDA/C++ 科学核和省略新参数时的原生行为保留。源码 Release 与目标 PyPI `7.1.0` wheel 的发布核验仍待完成；历史 7.0.0 不含此功能，见[安装](installation.md)和[发行状态](status.md)。

(torch-selection)=
## 显式选择后端与设备

- scalar/VRZ/VTI/elastic 的 `backend=None` 是保留原生行为的兼容默认值，不是自动选择 Torch。`backend="cuda"` 要求 CUDA 模型。
- GSLS 默认 `backend="cuda"`，`backend="native_cpu"` 仍表示原生 C++ CPU 实现。`backend="torch"` 与 native_cpu 不同。
- Torch 支持 CPU/CUDA；模型设备决定执行设备。请自行把输入安排在同一设备，不自动搬移模型；`backend="cpu"` 无效。MPS 仅支持 Scalar2D accuracy=4 float32，使用 backend="torch"、full/checkpoint 与 eager/compile；Scalar3D、VRZ、VTI、elastic、GSLS 和其他 MPS 组合明确拒绝。本次不宣称最终 Mac 实机、500 轮 FWI 或性能验收通过。
- scalar/VRZ/VTI 原默认 `memory="boundary"` 保留。切换 Torch 时必须同时显式选择 `memory="full"` 或 `"checkpoint"`，否则报错，不静默替换存储策略。
- Torch 不需要传播原生库、nvcc 或 `compile_all.py`；eager 不需要原生编译。compile 需要当前 PyTorch 的 Inductor 工具链，CUDA 使用其适用的 Triton 环境；失败不回退 eager。此项并不使历史 Linux wheel 可安装到其他平台。

(torch-memory)=
## 后端、执行和内存表

| 入口 / 后端 | 设备 | memory | 默认 / 限制 |
|---|---|---|---|
| scalar / VRZ / VTI 原生 | CUDA | full、boundary | 默认 boundary；不接受 checkpoint |
| elastic 原生 | CPU/CUDA | full；boundary 仅 CUDA | 默认 full；原有卸载/压缩条件独立保留 |
| GSLS 原生 | CUDA 或显式 native_cpu | full、checkpoint | 默认 full；None 间隔按形状和所需历史估算 |
| 五个入口，backend="torch" | CPU/CUDA | full、checkpoint | scalar/VRZ/VTI 必须覆盖原 boundary 默认；eager/compile 均支持 |

| 选项 | 默认 | Torch 含义 / 拒绝条件 |
|---|---|---|
| execution | eager | eager 或 compile；原生仅 eager |
| compile_steps | 1 | 正整数，拒绝 bool；每块连续内部时间步数，compile 可尝试 4；原生仅 1 |
| checkpoint_interval | None | 仅 checkpoint；正整数，拒绝 bool；Torch None=32 个内部步；非 GSLS 原生必须 None |

Torch full 保留 autograd 所需的中间张量；checkpoint 保存完整段起始状态（含 PML/GSLS 记忆变量），按需正向重算，使用非重入 checkpoint。两者均不是原生 boundary 逆重建，也不通过除阻尼逆演耗散状态。GSLS 原生选择性历史/内存公式不能直接套用于 Torch。间隔变大时检查点变少，但局部反传图变长；短记录或小模型下不保证降低峰值显存。内存还包括输入、记录、梯度、工作区与优化器。

compile_steps 只改变执行块大小，不改变 dt、nt、精度、炮数、loss 或更新次数；非整块尾部完整执行。checkpoint_interval 是内部时间步数，boundary_buffer 是空间网格点，两者不能互换。没有按 GPU 型号、SM、L2、空闲显存自动调参或拆炮。

(torch-scope)=
## 科学契约与限制

| Torch 入口 | 维度 | dtype | 一阶梯度 |
|---|---|---|---|
| scalar | 2D/3D | float32 | v；source 仅 3D |
| VRZ | 2D | float32 | v 与 impedance 或 density；源固定 |
| VTI | 2D/3D | float32 | vp、epsilon、delta、rho；源固定 |
| elastic | 2D/3D | float32/64 | lamb、mu、buoyancy、源、初始物理/PML 状态 |
| GSLS | 2D | float32/64 | vp、q、源、材料 provider 张量参数；rho 固定 |

上述 CPU/CUDA 路径支持 2/4/6/8 阶 full/checkpoint；MPS 仅支持 Scalar2D accuracy=4 float32。各方程继续使用自己的轴序、源单位、源尺度、采样时刻、CFL 与重采样契约，不能相互套用。详见 [Usage](usage.md) 与 [GSLS](visco-gsls.md)。GSLS 保留 hao1、band_fit、sls_compat、material_model；单机制仅显式 `mode="sls_compat", n_mechanisms=1`，不恢复独立 SLS API。

scalar/VRZ/VTI 复制延拓值，但延拓区模型梯度冻结，保留 crop-image 约定。若对外边缘做有限差分且重新生成延拓，得到的是另一映射；应区分内部模型方向检查。GSLS 的复制延拓梯度完整累加回物理模型。Torch VRZ full/checkpoint 要求 `boundary_buffer >= accuracy//2`。

Torch elastic 对每个内部时间步求导，要求 `model_gradient_sampling_interval=1`。原生材料梯度按 `CFL step_ratio × model_gradient_sampling_interval` 抽样；即使 interval=1，step_ratio>1 时也可能与完整 AD 不同。精确 native/Torch 材料梯度对照还需 step_ratio=1；这是后端梯度语义差别，不能归因于 compile。

以下组合明确拒绝：Torch boundary、Scalar2D accuracy=4 float32 范围之外的 MPS、AMP/autocast、scalar illumination、elastic callback、elastic storage offload/压缩、elastic sampling_interval!=1。elastic 要求 forward_callback=None、storage_mode="device"、storage_compression=False；python_backend 只接受 False 或 eager，编译用 execution="compile"。高阶传播导数、外部 CUDA Graph 捕获、vmap、自定义 stream 与分布式训练未验收，不在本次保证范围。

Torch scalar/VRZ/VTI 与 GSLS 会检查传入记录的反传余切量；NaN/Inf 直接报错。有限 loss 不保证其记录梯度有限，调用者须检查目标函数与求导链。

(torch-example)=
## CPU 最小独立示例

需要提供新后端的 V16 源码环境。此小例子检查接线，不证明 FWI 收敛；本手册构建仅静态检查代码，不执行传播。切换 CUDA 时显式更改 device 并把全部张量放到该设备；切换编译时设置 execution="compile"，compile_steps 可尝试 4。

```python
import torch
import starwave

# Explicit CPU Torch propagation; no native-library preparation.
device = torch.device("cpu")
v = torch.full((24, 20), 1800.0, device=device,
               dtype=torch.float32, requires_grad=True)
t = torch.arange(48, device=device, dtype=v.dtype) * 0.001
a = (torch.pi * 15.0 * (t - 0.02)).square()
source = ((1 - 2 * a) * torch.exp(-a)).reshape(1, 1, -1)
src = torch.tensor([[[12, 4]]], device=device, dtype=torch.long)
rec = torch.tensor([[[8, 4], [12, 4], [16, 4]]], device=device,
                   dtype=torch.long)
records, = starwave.scalar(
    v, 10.0, 0.001,
    source_amplitudes=source, source_locations=src,
    receiver_locations=rec, accuracy=4, pml_freq=15.0,
    pml_width=8, boundary_buffer=5, max_vel=2000.0,
    memory="checkpoint", backend="torch", execution="eager",
    checkpoint_interval=32, compile_steps=1,
)
records.square().mean().backward()
assert records.shape == (1, 3, 48)
assert v.grad is not None and torch.isfinite(v.grad).all()
```

独立 CPU smoke 已通过：V16 源码、Python 3.12.14、PyTorch 2.14.1+cpu，eager + checkpoint；记录形状 `(1,3,48)`，反传梯度有限。中英代码逐字相同。此结果仅验证该小配置的 CPU 接线，不表示 compile、CUDA、MPS、FWI 收敛或性能验收。

(torch-all-options)=
## 全可选参数调用

下列为已准备有效输入后的接线片段，不是完整实验。四个入口显式列出全部可选参数；GSLS 的完整参数调用见 {ref}`GSLS 示例 <visco-gsls-example>`。None 初态表示从零状态开始，不省略 API 覆盖。不能把不同方程的 source / geometry / 模型变量直接共用。

<!-- torch-api:scalar:full:start -->
```python
outputs = starwave.scalar(v, grid_spacing, dt,
    source_amplitudes=source,
    source_locations=src,
    receiver_locations=rec,
    accuracy=8,
    pml_freq=25.0,
    pml_width=20,
    boundary_buffer=5,
    memory="checkpoint",
    max_vel=None,
    freq_taper_frac=0.0,
    time_pad_frac=0.0,
    time_taper=False,
    illumination=None,
    backend="torch",
    execution="eager",
    checkpoint_interval=32,
    compile_steps=1,
)
```
<!-- torch-api:scalar:full:end -->

<!-- torch-api:vrz:full:start -->
```python
outputs = starwave.vrz(v, grid_spacing, dt,
    impedance=None,
    density=rho,
    source_amplitudes=source,
    source_locations=src,
    receiver_locations=rec,
    accuracy=8,
    pml_freq=25.0,
    pml_width=20,
    boundary_buffer=5,
    memory="checkpoint",
    max_vel=None,
    freq_taper_frac=0.0,
    time_pad_frac=0.0,
    time_taper=False,
    illumination=None,
    backend="torch",
    execution="eager",
    checkpoint_interval=32,
    compile_steps=1,
)
```
<!-- torch-api:vrz:full:end -->

<!-- torch-api:vti:full:start -->
```python
outputs = starwave.vti(vp, epsilon, delta, rho, grid_spacing, dt,
    source_amplitudes=source,
    source_locations=src,
    receiver_locations=rec,
    source_fields=('sH', 'sV'),
    receiver_fields=('vz',),
    accuracy=4,
    pml_freq=25.0,
    pml_width=20,
    boundary_buffer=5,
    memory="checkpoint",
    max_vel=None,
    freq_taper_frac=0.0,
    time_pad_frac=0.0,
    time_taper=False,
    backend="torch",
    execution="eager",
    checkpoint_interval=32,
    compile_steps=1,
)
```
<!-- torch-api:vti:full:end -->

<!-- torch-api:elastic:full:start -->
```python
outputs = starwave.elastic(lamb, mu, buoyancy, grid_spacing, dt,
    source_amplitudes_z=None,
    source_amplitudes_y=source_y,
    source_amplitudes_x=None,
    source_amplitudes_p=None,
    source_locations_z=None,
    source_locations_y=src_y,
    source_locations_x=None,
    source_locations_p=None,
    receiver_locations_z=None,
    receiver_locations_y=rec_y,
    receiver_locations_x=None,
    receiver_locations_p=None,
    accuracy=4,
    pml_width=20,
    pml_freq=None,
    max_vel=None,
    survey_pad=None,
    vz_0=None,
    vy_0=None,
    vx_0=None,
    sigmazz_0=None,
    sigmayz_0=None,
    sigmaxz_0=None,
    sigmayy_0=None,
    sigmaxy_0=None,
    sigmaxx_0=None,
    m_vzz_0=None,
    m_vzy_0=None,
    m_vzx_0=None,
    m_vyz_0=None,
    m_vxz_0=None,
    m_vyy_0=None,
    m_vyx_0=None,
    m_vxy_0=None,
    m_vxx_0=None,
    m_sigmazzz_0=None,
    m_sigmayzy_0=None,
    m_sigmaxzx_0=None,
    m_sigmayzz_0=None,
    m_sigmaxzz_0=None,
    m_sigmayyy_0=None,
    m_sigmaxyy_0=None,
    m_sigmaxyx_0=None,
    m_sigmaxxx_0=None,
    origin=None,
    nt=None,
    model_gradient_sampling_interval=1,
    freq_taper_frac=0.0,
    time_pad_frac=0.0,
    time_taper=False,
    forward_callback=None,
    callback_frequency=1,
    python_backend=False,
    storage_mode='device',
    storage_path='.',
    storage_compression=False,
    memory="checkpoint",
    backend="torch",
    execution="eager",
    checkpoint_interval=32,
    compile_steps=1,
)
```
<!-- torch-api:elastic:full:end -->

(torch-performance)=
## 编译、重复训练与证据范围

首次编译/预热与稳态 forward + loss + backward + optimizer.step 分开计时，同时记录 allocated/reserved 峰值。形状、精度、炮数、阶数或求导需求变化可能重新编译；缓存仅可复用代码，不可复用旧模型波场/系数/梯度。eager/compile 的收益不能解释成相对原生 CUDA 的加速比。

如果同一环境还运行原生 elastic/GSLS，Python 入口变化会使源闭包身份变化，须从完整匹配源码重新构建并重启进程，不手改旧 manifest；见[安装](installation.md)。文档及 CPU 接口核验不能替代目标设备 GPU、DataParallel、长时 FWI 或性能验收。原有 Scalar3D 图件、教程与演示文稿保留各自历史证据范围，不改标为 V16 运行结果。

## Scalar2D MPS Notebook

源码中的 L2_True_StarWave_Acoustic_MPS.ipynb 和同步 .py 直接 import starwave，仅用 PROJECT_ROOT 显式读取资源，不隐式安装或构建。保留 dt=0.003、nt=2000、30 炮/5 批、500 epochs、Adam lr=10、checkpoint_interval=32 与 compile_steps=4；空间阶数与编译块长度独立。安装本次审阅源码后重启 Python，禁用 PYTORCH_ENABLE_MPS_FALLBACK；不支持的算子或编译失败直接报错，不回退 CPU/eager。既有 7.0.0 二进制不含本更新，发布核验仍待完成。
