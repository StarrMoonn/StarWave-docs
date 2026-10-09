# SLS 黏声学 API

V14 新增独立的二维单标准线性固体（single-SLS）黏声学接口。首次公开于 [StarWave 7.0.0](https://pypi.org/project/starwave/7.0.0/)，发行文件与核验范围见[文档状态](status.md)。既有 [Usage](usage.md) 中 scalar、VRZ、VTI、elastic 及原生运行时函数的签名与默认值保持不变。

本接口支持固定、可空间变化的密度，以及 Vp、Q 和源波形各自的一阶梯度。传播仅支持二维和 `memory="full"`；源物理量、模型轴序与时间采样均使用下述独立约定。

(visco-sls-function)=
## visco_sls

```{py:function} starwave.visco_sls(vp: torch.Tensor, q: torch.Tensor, rho: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, float], dt: float | int, *, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, f_ref: float | int, accuracy: int=4, pml_width: int | list[int] | tuple[int, int]=20, memory: str='full', max_vel: float | int | None=None, backend: str='cuda') -> tuple[torch.Tensor]

:param vp: **必填；位置或关键字。** `[nz,nx]` 二维相速度，单位 m/s，定义在 `f_ref`。每维至少 2；正、有限的 float32/float64 张量。可独立求一阶梯度。
:type vp: `torch.Tensor`
:param q: **必填；位置或关键字。** 与 `vp` 同形状、dtype、设备的复体积模量品质因子，在 `f_ref` 定义；正有限值或正无穷。可独立求一阶梯度；`+inf` 为无耗散极限，该处 Q 灵敏度为零。
:type q: `torch.Tensor`
:param rho: **必填；位置或关键字。** 正且有限的密度，单位 kg/m³；与 `vp` 同形状、dtype、设备，倒数须可表示。允许空间变化，但必须固定；拒绝 `requires_grad=True`。
:type rho: `torch.Tensor`
:param grid_spacing: **必填；位置或关键字。** 正有限间距，单位 m；单值表示等间距，二元素 tuple/list 表示 `(dz,dx)`，允许不等间距。
:type grid_spacing: `float | int | list[float] | tuple[float, float]`
:param dt: **必填；位置或关键字。** 正有限采样间隔，单位 s；必须满足 SLS 保守时间步限制。不隐式重采样或自动子步。
:type dt: `float | int`
:param source_amplitudes: **必填；仅关键字。** 有限的 `[B,S,T]` 张量，各维非空，与模型 dtype 和设备一致。二阶方程 forcing，单位 Pa/m²；支持独立一阶梯度。非连续 strided 输入转为连续存储时保留 autograd 链。
:type source_amplitudes: `torch.Tensor`
:param source_locations: **必填；仅关键字。** int32/int64 `[B,S,2]`，按 `[z,x]` 索引未填充的原始模型；可位于 CPU 或模型设备，调用时复制到模型设备的 int64。重合源相加。不接受省略源维度的简写。
:type source_locations: `torch.Tensor`
:param receiver_locations: **必填；仅关键字。** int32/int64 `[B,R,2]`，`R>=1`；坐标约定与源相同。重复接收点产生重复道，反传时其余切量相加。
:type receiver_locations: `torch.Tensor`
:param f_ref: **必填；仅关键字。** 正有限参考频率，单位 Hz；同时定义 Vp、Q 及固定 CPML alpha 剖面，角频率为 `2*pi*f_ref`。
:type f_ref: `float | int`
:param accuracy: **默认 `4`；仅关键字。** 空间交错网格有限差分阶数，支持 `2,4,6,8`；时间离散为二阶。
:type accuracy: `int`
:param pml_width: **默认 `20`；仅关键字。** 每侧非负整数网格数；单值或二元素 `(z,x)` tuple/list。每个轴两侧对称；某轴为 0 时关闭该轴 PML，两轴均为 0 时使用有限零外延计算域。
:type pml_width: `int | list[int] | tuple[int, int]`
:param memory: **默认 `'full'`；仅关键字。** 仅支持完整体积历史；拒绝 `boundary`。内存随填充网格、炮数和时间步增长。
:type memory: `str`
:param max_vel: **默认 `None`；仅关键字。** 非松弛速度的正有限上界，单位 m/s，通常大于 `vp`，必须覆盖模型。开启梯度且 Vp/Q 可训练、PML 非零时必须显式给定，并在训练和成对有限差分调用中固定。其余情况 `None` 可逐次计算分离于 autograd 的上界。
:type max_vel: `float | int | None`
:param backend: **默认 `'cuda'`；仅关键字。** 显式选择 `cuda`、`native_cpu` 或 `torch`。前两者分别要求 CUDA/CPU 张量和匹配的原生库；`torch` 为小模型参考后端。不自动切换后端或编译。
:type backend: `str`
:returns: 单元素元组 `(pressure_records,)`；记录 `[B,R,T]`，单位 Pa，与模型设备/dtype 一致。
:rtype: `tuple[torch.Tensor]`
```

(visco-sls-physics)=
## 物理参数与时间采样

`vp` 是参考频率处的相速度，不是松弛或非松弛速度。`q` 是采用 `exp(i*omega*t)` 约定时、同频复体积模量的实部与虚部之比。单 SLS 同时描述衰减和频散，其 Q 随频率变化；它不是宽频恒 Q 模型，也不将复模量 Q 与波数衰减定义的 Q 混用。

源为离散网格上的二阶 forcing，单位 **Pa/m²**。压力更新包含 `-dt**2 * c_rel**2 * forcing`；为满足零初始压力及压力时间导数，第一次 forcing 更新乘 `1/2`。没有隐藏的网格体积因子，也不接受把 Pa/s 压力率源直接当作相同物理量。迁移旧波形或观测数据时，应分别检查源标定、采样和方程约定。

记录为更新前的 `p[n]`，时间 `t=n*dt`，共 `T` 个样点。因此第一点严格为零，最后一个源样点只影响未返回的 `p[T]`，其记录反传梯度为零。`T=1` 时记录以及模型/源梯度均为零；不会把步后记录改标为步前时间。

(visco-sls-gradients)=
## 梯度、PML 与内存

- Vp、Q 和源可分别或同时训练；密度、几何、间距、时间步与 PML 设置固定。每次 forward 拥有独立历史，loss、归一化、mask 和优化器由调用者定义。
- 模型使用对称 replicate 扩边，autograd 将全部填充区域灵敏度累加回原始边缘/角点，包含扩边操作的完整转置。源的松弛速度系数保留在传播图中，源位置处的模型灵敏度不会丢失。
- CPML 的节点/面记忆采用离散转置反传；其阻尼、alpha 和速度上界属于固定数值设置，不是可微材料参数。
- 当开启梯度、Vp/Q 可训练且 PML 非零时，必须提供覆盖**非松弛速度**的显式 `max_vel`。训练迭代及正负有限差分扰动共用同一上界；不能仅按 `vp.max()` 估计。
- 仅正演、仅源求导或 `pml_width=0` 时可使用 `max_vel=None`。过小的显式上界会被拒绝。
- `full` 保存体积历史，随网格、炮数和时间步增加内存。不支持边界重建、checkpoint、压缩、照明、自由表面、3D、初始状态输入、最终状态返回、原生高阶梯度、AMP 或 CUDA graph。
- float32/float64 模型与源不自动转换 dtype/设备。拒绝非有限输入及传入的接收余切量；不通过隐藏 clamp 替换错误值。原生 backward 支持保存张量恢复为数值一致的非连续布局，必要时显式复制为连续存储。

(visco-sls-stability)=
## 时间步检查

令 `S` 为所选交错差分系数绝对值之和，`v_bound` 为覆盖实际非松弛速度的上界，则公开检查使用：

```text
dt <= 0.8 / (v_bound * sqrt(max(rho)/min(rho))
             * S * sqrt(1/dz**2 + 1/dx**2))
```

密度反差因子使界限更保守。超限会报错并提示最大 dt；调用者须重新选择时间步并生成相应采样的源，不会隐式修复。该无 PML 能量界及安全系数不能证明任意强反差、极端 Q、异质 CPML 和长时间实验都稳定；实际实验仍需检查有限性及边界反射。

(visco-sls-example)=
## 独立小模型调用

下面是可直接运行的 CPU Torch 参考接线，采用固定可变密度，显式训练 Vp/Q/source，并检查首/末采样约定。1000 是此例 Ricker forcing 的 Pa/m² 标定；损失只是记录能量，用于检查 autograd 接线，不代表反演收敛或实测结果。

```python
import torch
import starwave

device = "cpu"
dtype = torch.float64
nz, nx, nt = 12, 16, 100
dt = 0.001
vp = torch.full((nz, nx), 1800.0, dtype=dtype, device=device,
                requires_grad=True)
q = torch.full((nz, nx), 40.0, dtype=dtype, device=device,
               requires_grad=True)
depth = torch.linspace(0.0, 1.0, nz, dtype=dtype, device=device)
rho = (1800.0 + 200.0 * depth[:, None]).expand(nz, nx).contiguous()
t = torch.arange(nt, dtype=dtype, device=device) * dt
a = torch.pi * 18.0 * (t - 0.04)
source = (1000.0 * (1.0 - 2.0 * a.square()) * torch.exp(-a.square()))
source = source.reshape(1, 1, nt).requires_grad_()
src = torch.tensor([[[3, 8]]], dtype=torch.int64, device=device)
rec = torch.tensor([[[3, 4], [3, 8], [3, 12]]],
                   dtype=torch.int64, device=device)

records, = starwave.visco_sls(
    vp, q, rho, (10.0, 10.0), dt,
    source_amplitudes=source, source_locations=src,
    receiver_locations=rec, f_ref=18.0,
    accuracy=4, pml_width=(4, 4), memory="full",
    max_vel=2500.0, backend="torch",
)
assert records.shape == (1, 3, nt)
assert torch.isfinite(records).all()
assert torch.count_nonzero(records[..., 0]) == 0
records.square().mean().backward()
for gradient in (vp.grad, q.grad, source.grad):
    assert gradient is not None and torch.isfinite(gradient).all()
assert torch.count_nonzero(source.grad[..., -1]) == 0
```

切换原生 CPU 时，先运行 `starwave.prepare_visco_sls(backend="native_cpu")`，再将传播调用设为 `backend="native_cpu"`。CUDA 则使用显式逻辑设备，如 `device="cuda:0"`，在主线程预加载后，以相同设备构建模型和源，并设 `backend="cuda"`。不要根据 Torch 小例子推断 CUDA 性能。

固定 Q 反演 Vp 时，只让 `vp` 开启梯度；Vp/Q 联合反演时两者开启梯度，并分别进入用户优化器。rho 始终固定。真实 FWI 还需合理参数约束、源校准、频带选择及正则化，Vp/Q 之间存在权衡；传播器不暗含这些实验决策。

(visco-sls-runtime)=
## 原生运行时准备

```{py:function} starwave.prepare_visco_sls(device: str | torch.device='cuda:0', *, backend: str='cuda') -> dict

验证并载入已有的匹配 SLS 原生库，不执行编译或正演。应在开始训练或外部多设备分配前完成。

:param device: **默认 `'cuda:0'`；位置或关键字。** CUDA 必须指定有效的逻辑编号，例如 `'cuda:0'` 或 `torch.device('cuda:0')`；不能只写 `'cuda'`。`native_cpu` 忽略此参数。
:type device: `str | torch.device`
:param backend: **默认 `'cuda'`；仅关键字。** `'cuda'` 或 `'native_cpu'`，不接受 `'torch'`。
:type backend: `str`
:returns: 状态字典；正常返回说明库载入与兼容检查通过，不表示 GPU 数值验收。
:rtype: `dict`
```

```{py:function} starwave.visco_sls_native_status(*, backend: str='cuda') -> dict

只读所选 SLS 库的存在/载入状态，不载库、不编译。

:param backend: **默认 `'cuda'`；仅关键字。** `'cuda'` 或 `'native_cpu'`。
:type backend: `str`
:returns: 状态字典，包括 backend、library_exists、library_loaded、memory 和 gradient_order 等诊断字段。文件存在及 CUDA 可用均不能单独证明传播数值正确。
:rtype: `dict`
```

```python
import starwave

print(starwave.visco_sls_native_status(backend="cuda"))
status = starwave.prepare_visco_sls("cuda:0", backend="cuda")
print(status["library_exists"], status["library_loaded"])
```

公开 7.0.0 wheel 已包含预编译 SLS CPU/CUDA 库；安装条件与平台限制以[安装](installation.md)及[文档状态](status.md)为准。授权源码用户按随包说明显式构建。既有 `prepare_native` 或 `prepare_elastic` 不替代 SLS 准备。更新 Python 包或已载入库后必须重启解释器/Notebook kernel，不绕过兼容检查。

(visco-sls-references)=
## 方程参考与适用范围

- Bai, Yingst, Bloor and Leveille (2014), “Viscoacoustic waveform inversion of velocity structures in the time domain”, GEOPHYSICS 79(3), R103–R119。[论文](https://www.tgs.com/hubfs/ION%20Papers/2014_GEO_JBai_Q_WFI.pdf)，DOI: 10.1190/GEO2013-0030.1。
- [Devito 官方黏声学教程](https://www.devitoproject.org/examples/seismic/tutorials/11_viscoacoustic.html)：单 SLS 等流变模型的教学背景。

以上是物理方程参考。StarWave 的源负号、参考相速度参数化、交错有限域算子、CN 记忆/PML 时间安排及离散反传按本接口约定定义，不声称与论文或 Devito 逐位一致。真实的固定 Q 与联合 Vp/Q 反演见 [SLS Marmousi2 Example](examples/visco-sls.md)。旧 [Scalar3D Example](examples/index.md) 保留其原始版本与记录，不能用作 SLS 的验证结果。
