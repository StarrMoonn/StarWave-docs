(vrr)=
# VRR Function

这是已整合到经审阅 dev 源树的**独立源码扩展**，发行包名 `starwave-vrr`（`0.1.0.dev1`），导入名 `starwave_vrr`。它不在已发布的 V16 / StarWave 7.1.0 wheel 中，也没有 `starwave.vrr` 别名。使用前须获得包含 `extensions/vrr` 的完整授权源码；公开手册不分发私有实现。主包安装另见[安装页](installation.md)。

(vrr-install)=
## 安装与构建

在完整源码根目录，选择目标 Python 环境，并先准备兼容依赖：

```bash
python -m pip install --no-deps --no-build-isolation -e ./extensions/vrr
python -c "import starwave_vrr; print(starwave_vrr.__file__)"
# Native CUDA only; replace 80 with the actual GPU SM.
python -m starwave_vrr.build --plan --arch 80
python -m starwave_vrr.build --arch 80
```

使用 `backend="torch"` 时跳过两条构建命令。导入和传播不会隐式构建；根目录 `compile_all.py` 只构建主包。`--plan` 仅打印计划，不证明编译成功；`--arch` 接受 `80` 或 `8.0`，要求 SM 60 或更新。可用 `--nvcc`、`--ccbin` 选择已安装编译器。

独立 nvcc/C++17 构建不使用 Torch C++ 头文件或 libtorch，可在 CPU 版 Torch 环境运行构建入口；实际 GPU 运算仍须 CUDA 版 Torch、支持的 GPU 及兼容工具链/驱动/运行时，不能保证任意版本组合可用。扩展 C ABI 3 与主包独立。源码或库变化后须按同源重编译并重启 Python；不能复用 ABI 1/2 旧库。默认缓存为 `~/.cache/starwave-vrr/`，可用 `STARWAVE_VRR_CACHE` 另选扩展缓存目录。

仅卸载扩展注册时执行 `python -m pip uninstall starwave-vrr`；源码与缓存另行处理。editable 注册不会替换 `starwave`。

(vrr-api)=
## 签名与参数

```text
starwave_vrr.vrr(v, grid_spacing, dt, *, rx, rz, source_amplitudes, source_locations, receiver_locations, accuracy=8, pml_freq=25.0, pml_width=20, boundary_buffer=5, memory="boundary", max_vel=None, freq_taper_frac=0.0, time_pad_frac=0.0, time_taper=False, backend="cuda", execution="eager", checkpoint_interval=None, compile_steps=1)
```

前三项可按位置传入，其余均为仅关键字参数。返回 `(record,)`，记录形状为 `[B,R,T]`，可用 `result[-1]` 取出；不返回末态或照明。

| Parameter | Default | 约定 |
|---|---|---|
| `v` | `required` | 正有限速度 [nx,nz]，单位 m/s。 |
| `grid_spacing` | `required` | 正等距网格：标量或相等二元组，单位米。 |
| `dt` | `required` | 正用户采样间隔，单位秒。 |
| `rx` | `required` | 有限带符号 x 向反射率向量场，与 v 同形、同设备、同精度；单位 1/m。 |
| `rz` | `required` | 有限带符号 z 向反射率向量场，与 v 同形、同设备、同精度；单位 1/m。 |
| `source_amplitudes` | `required` | 外力 [B,S,T]；共享子波须显式扩展到各炮、各源。 |
| `source_locations` | `required` | 整数 [B,S,2]，每炮单源也可用 [B,2]。 |
| `receiver_locations` | `required` | 整数 [B,R,2]；坐标索引未填充的物理模型。 |
| `accuracy` | `8` | 内部空间差分阶数：2、4、6 或 8。 |
| `pml_freq` | `25.0` | 正 CPML 参考频率，单位 Hz。 |
| `pml_width` | `20` | 正标量、四面相等正宽度，或正整数 w 对应的 [w,w,0,w]。 |
| `boundary_buffer` | `5` | 空间网格数；至少 accuracy//2+1，包括 full 模式。 |
| `memory` | `"boundary"` | CUDA：full/boundary。Torch：full/checkpoint，须显式选择。 |
| `max_vel` | `None` | 省略时取当前速度最大值并 detach；显式正上界须覆盖 v。 |
| `freq_taper_frac` | `0.0` | FFT 重采样的频率窗比例。 |
| `time_pad_frac` | `0.0` | FFT 重采样的时间补零比例。 |
| `time_taper` | `False` | 可选重采样时间窗。 |
| `backend` | `"cuda"` | cuda 或 torch；不自动切换后端。 |
| `execution` | `"eager"` | Torch：eager/compile；原生 CUDA 仅接受 eager。 |
| `checkpoint_interval` | `None` | 仅 Torch checkpoint；None 为 32 个内部时间步，否则须为正整数。 |
| `compile_steps` | `1` | 每个编译块的内部时间步数，须为正整数；原生 CUDA 仅接受 1。 |

(vrr-backends)=
## 后端与梯度

| Backend | Device / dtype | Memory | Execution |
|---|---|---|---|
| `cuda` (default) | CUDA float32 | `full`, `boundary` (default) | `eager` selector; native execution |
| `torch` | CPU/CUDA float32/float64 | `full`, `checkpoint` | `eager`, `compile` |

仅选择 Torch 不会改变 boundary 默认值，须同时指定 `memory="full"` 或 `"checkpoint"`。无效组合明确报错，不静默切换。Torch compile 显式使用已安装 Inductor，编译错误直接抛出。原生 CUDA 要求 `execution="eager"`、`compile_steps=1`、`checkpoint_interval=None`。

CUDA full 保存六状态时序（两级压力及四个 CPML 记忆场）；boundary 保存条带历史和末态、逆重建内部，历史仍随时间和条带规模增长。Torch checkpoint 保存完整六状态段端点，反传重算段内正演，与原生 boundary 重建不同。

支持一次一阶反传，覆盖独立 `v`、`rx`、`rz` 和震源幅值，包括源点速度缩放及完整复制扩边转置。几何与规划输入不求导。不支持 MPS、AMP/autocast、CUDA graphs、原生 checkpoint、Torch boundary、VRR3D 或高阶 AD。`starwave_vrr.vrr_reference` 是显式选择的 CPU/CUDA float32/64 full 历史参考入口，无 backend/execution/checkpoint/compile 选择参数，也不是失败回退。

(vrr-physics)=
## 模型、震源与时间约定

```text
p_tt = v² (p_xx + p_zz) + (v v_x - 2 v² rx) p_x + (v v_z - 2 v² rz) p_z
```

二维声学反射率向量模型采用公开 `[x,z]` 数组，深度在最后一轴；采集坐标为整数 `[ix,iz]`。常见 `[z,x]` 模型须逐个转置。`rx/rz` 为带符号逆长度场，不是密度、Q 或无量纲反射系数。对光滑正阻抗 `Z=v*rho`，连续关系为 `r=0.5*gradient(log Z)`；离散差分不必满足链式/乘积法则，因此不代表与离散 VRZ 等价。API 不自动换算密度，独立向量场也不必对应物理可实现的阻抗。

若压力单位为 P，震源外力单位为 P/m²；注入量为 `-v_source² * dt_internal² * U(source)`，其中 U 为时间上采样。初态为零；公开记录对更新前对齐序列 `[0, raw[:-1]]` 下采样，对应用户时刻 `0,dt,...`。FFT 重采样可能振铃，不能保证所有子步比例的首个公开样点精确为零。

内部规划使用 `max_dt=0.6*h/(max_vel*sqrt(2))` 和 `ratio=max(1,ceil(dt/max_dt))`。该主速度上界不保证任意反射率、速度梯度或 CPML 下稳定。优化和有限差分宜固定覆盖模型的 `max_vel`，检查实际时长并做空间/时间加密。传播器不裁剪或平滑模型，不修补 loss，也不选择优化器。

(vrr-top)=
## 水平顶部边界

面顺序为 `[x-low,x-high,z-low,z-high]`。仅正整数 w 对应的 `pml_width=[w,w,0,w]` 选择水平顶面；正标量或四面相等正宽度保留全吸收。其他零面、多零面、全零或不等正吸收宽度均拒绝。

顶部移除 PML/buffer，但保留 `M=accuracy//2` 层零 halo；物理 `z=0` 仍参与更新。这是零 halo 闭合，不是奇镜像延拓：二阶零压面位于首个 ghost 行，高阶内部差分不意味着同阶表面收敛精度。不增加自由表面开关，也不支持任意边界面或地形。

(vrr-example)=
## 小型合成示例

```python
import torch
from starwave_vrr import vrr

# A small synthetic CPU example; all models use [x,z].
v = torch.full((32, 32), 1500.0, dtype=torch.float64, requires_grad=True)
rx = torch.zeros_like(v, requires_grad=True)
rz = torch.zeros_like(v, requires_grad=True)
t = torch.arange(40, dtype=v.dtype) * 0.001
phase = torch.pi * 25.0 * (t - 0.02)
source = ((1 - 2 * phase.square()) * torch.exp(-phase.square())).reshape(1, 1, -1)
source = source.detach().requires_grad_()
src = torch.tensor([[[16, 12]]], dtype=torch.int64)
rec = torch.tensor([[[12, 12], [20, 12]]], dtype=torch.int64)
record = vrr(
    v, 10.0, 0.001, rx=rx, rz=rz,
    source_amplitudes=source, source_locations=src, receiver_locations=rec,
    accuracy=4, pml_freq=25.0, pml_width=8, boundary_buffer=5,
    memory="checkpoint", max_vel=1600.0,
    freq_taper_frac=0.0, time_pad_frac=0.0, time_taper=False,
    backend="torch", execution="eager", checkpoint_interval=10, compile_steps=1,
)[-1]
assert record.shape == (1, 2, 40)
record.square().mean().backward()
assert all(x.grad is not None and torch.isfinite(x.grad).all()
           for x in (v, rx, rz, source))
```

改用原生 CUDA 时，先构建，将所有模型、源与几何张量放到同一 CUDA 设备，模型/源使用 float32，并显式选择 `backend="cuda", memory="boundary", execution="eager", checkpoint_interval=None, compile_steps=1`。该 CPU 示例检查调用链路，不代表 GPU 或物理收敛验收。

(vrr-status)=
## 源码与验证范围

经审阅的精确 dev 运行时基线为 `904c9106bdb883325df8586b7c3412f7814a9737`，保留 `6b9ec41d3bef1a8a3593a63f124c6ee4139955b0` 整合的 VRR 运行时。该 dev 扩展晚于公开 7.1.0 所依据的冻结 V16 源码 `5a224dbb1e86154ff034bbedaf67c0e4cf7a1b6a`。经审阅的 VRR 功能输入为 `e8ccf9b0fe70edd023fc4c6d0319abae6c203c65`；dev 整合保留既有主包自由表面工作，扩展仍独立打包。这些源码身份不表示新增公开 wheel 或 Release。

物理参数化依据 [Whitmore 等，EAGE 2020](https://doi.org/10.3997/2214-4609.202010332)。张量顶部约定依据 [SWEEP dev](https://github.com/DeepWave-KAUST/sweep/tree/32235f796ffa5b272ca81b9016f8cc85e25ecdf1) 提交 `32235f796ffa5b272ca81b9016f8cc85e25ecdf1`；原生 VRR CUDA 步进/VJP 与边界重建为独立实现，不是上游 CUDA 移植。

经审阅源码记录了用户提供的 RTX 4060 Laptop 与 A30 数值报告，包含原生 full/boundary 和输入梯度；CPU/Inductor、主机加载、实际 GPU、多卡、长程 FWI 与性能属于不同证据类别。本次文档构建不重跑或认证这些 GPU 结果，也不将结果转移为 7.1.0 wheel 或任意硬件/物理条件的保证。一次一阶反传不代表 Born/LSRTM 或 Hessian API。
