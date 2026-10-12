# Usage

GSLS: [visco_gsls](#visco-gsls). 独立源码扩展：[VRR Function](#vrr)，不在 7.1.0 wheel 中。

```{container} sw-page-toc

**本页内容**

- [Scalar Function](#scalar)
- [Born Function](#born)
- [VRZ Function](#vrz)
- [VRR Function](#vrr)
- [VTI Function](#vti)
- [Elastic Function](#elastic)
- [GSLS Function](#visco-gsls)
- [材料参数转换](#elastic-conversions)
- [starwave.native_status](#native-status)
- [starwave.prepare_native](#prepare-native)
- [starwave.prepare_elastic](#prepare-elastic)
- [其它导出名称与范围](#other-exports)
```

本参考面向已审查的 V17 源码 `0.1.0.dev17`，包括 Born、已合并的平顶边界行为与导入时原生初始化。[Release V17](https://github.com/StarrMoonn/StarWave/releases/tag/V17) 已发布，PyPI `8.0.0` 尚待发布；已发布的 `7.1.0` wheel 是 V16 基线，不会因文档更新获得这些变化。详见[文档状态](status.md)。原有位置参数与原生默认行为保留；显式 [PyTorch 后端](pytorch-backend.md) 选项仍遵循各方程的限制。下方逐项列出完整签名、参数、返回与限制。

下方 Scalar 节明确描述后续 V18 API，其版本说明优先于本页 V17 基线。

6.0.0 的内部存储和 PML 转置维护见[发布说明](release-notes.md)，二维存储变化见[scalar 说明](modeling/scalar.md)。它们不引入新的公共参数；源码升级需重新构建配套原生库。

页面组织参考 [Deepwave 官方 Usage](https://ausargeo.com/deepwave/usage) 的 Sphinx Python API 风格，正文按 StarWave 的实际契约重新编写。两者的参数集合、源单位、返回结构和可微范围不能互换。

(propagators)=
## 传播函数速查

- {py:func}`starwave.scalar`：二维/三维标量声学；速度模型 `v`；默认返回记录元组，dev 可选最终物理波场。

- {py:func}`starwave.scalar_born`：一维/二维/三维常密度声学 Born；速度扰动 δv，返回背景/散射记录及终态。V17 源码已包含，8.0.0 尚待发布。

- {py:func}`starwave.vrz`：二维变密度声学；`v` 加恰好一种 `impedance` / `density` 参数化。

- {py:func}`starwave_vrr.vrr`：二维声学反射率向量；独立 `v, rx, rz` 模型场。单独安装的源码扩展，不在 7.1.0 wheel 中。

- {py:func}`starwave.vti`：二维/三维声学 VTI；`vp, epsilon, delta, rho`；按所选分量顺序返回记录。

- {py:func}`starwave.elastic`：二维/三维各向同性弹性；`lamb, mu, buoyancy`；返回完整最终状态及 p/速度记录。

- {py:func}`starwave.visco_gsls`：二维广义标准线性固体声学；固定可变密度，原生 full/checkpoint；详见下方 [GSLS Function](#visco-gsls)。

符号约定：`B` 炮数、`S` 每炮源数、`R` 每炮接收点数、`T` 用户时间采样数、`D` 空间维数。源和接收点使用物理模型的整数网格下标，不是米坐标，不包含 PML 偏移。

(scalar)=
## Scalar Function

本 Scalar 节描述由 `bc63dd3c3e438256e46fce3a1738b94e6c9a2970` 审阅并合入 dev `e040aa9b8225ab387f7e4853bfe6162f5351f5d5` 的 V18 扩展，须使用匹配源码；原生执行还需重编译对应库。冻结 V17（`27c2e14b184b94d8031a5b3a83c25b1ed3d4ac67`）、8.0.0 发布基线与已发布 7.1.0 均保留较早的 20 参数 scalar API：没有终态/回调关键字，二维源固定，不支持任意零面；V17 只支持已说明的顶部零宽排列。下方其它传播器仍遵循各自发布契约。

[Release V18](https://github.com/StarrMoonn/StarWave/releases/tag/V18) 已发布，冻结提交为 `fba1f496b890f2dbaf3fbb3011d39f6d723cfea1`，保留已审 dev `e040aa9` 的 scalar API。PyPI **9.0.0 仍待发布**；7.1.0 仍是已核验的最新公开 wheel。这些功能请使用匹配的 V18 源码。V17 与 8.0.0 基线保持不变。

```{py:function} starwave.scalar(v: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, accuracy: int=8, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float | int=0.0, time_pad_frac: float | int=0.0, time_taper: bool=False, illumination: ScalarIllumination | None=None, backend: str | None=None, execution: str='eager', checkpoint_interval: int | None=None, compile_steps: int=1, return_wavefield: bool=False, forward_callback: Callable | None=None, callback_frequency: int=1) -> tuple[torch.Tensor, ...]

根据 `v.ndim` 在二维或三维网格上传播标量声学波，支持独立炮。两种维数均支持一阶速度与源波形梯度；坐标和数值设置不参与求导。每次调用从新状态开始，可选返回可微的最终物理波场；它不是完整重启状态。

:param v: **必填；位置或关键字参数。** 二维 `[N0,N1]` 或三维 `[N0,N1,N2]` 模型，每维至少为 2；float32（原生 CUDA；Torch 为 CPU/CUDA），典型单位 m/s。由 `v.ndim` 选择传播维数。坐标 `[i,j]` / `[i,j,k]` 直接索引 `v[i,j]` / `v[i,j,k]`；三维示例采用 `[nx,ny,nz]`，不自动交换物理轴。所有值须有限；允许负值或零值，但不保证物理合理性。模型轴变换保留 autograd 链；`requires_grad=True` 时请求速度梯度。
:type v: `torch.Tensor`
:param grid_spacing: **必填；位置或关键字参数。** 正且有限的网格间距，典型单位 m。二维接受单值或两个相等的值；三维接受单值或按模型轴序排列的三个值（例如 `[dx,dy,dz]`），允许不等间距。拒绝布尔值、零/负值和长度不匹配。
:type grid_spacing: `float | int | list[float] | tuple[float, ...]`
:param dt: **必填；位置或关键字参数。** 正且有限的用户输入/输出采样间隔，典型单位 s。内部可能按 CFL 采用更小的时间步，返回间隔仍为此值。不能传入 Tensor 或布尔值，不对其求导。
:type dt: `float | int`
:param source_amplitudes: **必填；仅关键字参数。** 实数张量 `[B,S,T]`，三个维度均非空；数据须有限、可表示为 float32，会转换到模型设备和 float32。二维和三维都支持一阶源梯度，可单独或同时训练源与速度。它是归一化 forcing `f`，不是每步压力增量；若波场用 Pa、长度用 m，单位为 Pa/m²。不能只传 `[T]` 或 `[B,T]`，共享波形应显式扩展炮/源维。
:type source_amplitudes: `torch.Tensor`
:param source_locations: **必填；仅关键字参数。** 固定整数张量 `[B,S,D]`，`D=v.ndim`；单源时也接受 `[B,D]`。建议 `torch.long`。坐标按输入模型轴序直接索引物理网格；三维 `[i,j,k]` 对应 `v[i,j,k]`。不能包含 PML 偏移或越界值。每个源有对应波形，同一炮内重合源的增量相加。
:type source_locations: `torch.Tensor`
:param receiver_locations: **必填；仅关键字参数。** 固定整数张量 `[B,R,D]`，`D=v.ndim`、`R>=0`，建议 `torch.long`。炮数与源一致，使用输入模型轴序的物理网格下标。允许重复位置；此关键字仍必填，可传整型空张量 `[B,0,D]`，返回 `[B,0,T]` 记录。仅终态 loss 不需要虚构接收器。
:type receiver_locations: `torch.Tensor`
:param accuracy: **默认 `8`；仅关键字参数。** 空间有限差分阶数，可选 `2,4,6,8`，不是误差容差。影响模板范围、计算量和 boundary buffer 最小值；拒绝布尔值及其它阶数。
:type accuracy: `int`
:param pml_freq: **默认 `25.0`；仅关键字参数。** 正且有限的 PML 设置频率，典型单位 Hz，也用于空间采样诊断。不是波形生成器，不会自动测量源信号频率；应按实际实验选择。
:type pml_freq: `float | int`
:param pml_width: **默认 `20`；仅关键字参数。** 非负整数网格厚度，或二维四个/三维六个非负逐面宽度。任意面组合可为零，包括标量 0/全零；所有正宽度仍须等于同一 A。顺序为 `[x-low,x-high,z-low,z-high]` / `[x-low,x-high,y-low,y-high,z-low,z-high]`。零面仅保留固定零 FD ghost，不含 PML/buffer，物理边缘样点仍活动。拒绝负值、非整数、bool 及不相等的正宽度。该 dev scalar 取值域不扩展 VRZ/VTI。
:type pml_width: `int | list[int] | tuple[int, ...]`
:param boundary_buffer: **默认 `5`；仅关键字参数。** 非负空间网格 buffer，不是时间步数或 checkpoint 间隔。boundary 模式要求至少 `accuracy // 2 + 1`，8 阶时为 5；full 模式允许 0。各吸收侧总填充为 `pml_width + boundary_buffer + accuracy // 2`；各零宽面只保留 `accuracy // 2` 个 halo 单元，不含该面的 PML 或 buffer。
:type boundary_buffer: `int`
:param memory: **默认 `'boundary'`；仅关键字参数。** 反传历史策略。`"boundary"` 保存压力边界条带与两个终态压力场并重构；三维六个面的保存宽度为 `M=accuracy//2`（Radius-M）。三维 `"full"` 保存逐内部时间步、完整填充体积的未缩放 `Lap(u)` 历史，适合小规模对照。两种维度只训练源也会保存所选历史；只有不需要速度或源梯度的正演才不保存。full 可能耗尽显存；原生没有自动降级、CPU/磁盘卸载或 checkpoint；Torch 另支持 full/checkpoint（不支持 boundary），保存 autograd 图或完整状态检查点，不使用这里的原生 Lap(u) 历史存储。
:type memory: `str`
:param max_vel: **默认 `None`；仅关键字参数。** CFL/PML 速度包络，典型单位 m/s。`None` 时每次调用从当前模型 `max(abs(v))` 重新规划。显式值必须正且有限并覆盖该最大值；不是裁剪上限。全零模型需要显式正值。极值、时间子步与 PML 设置不参与求导。
:type max_vel: `float | int | None`
:param freq_taper_frac: **默认 `0.0`；仅关键字参数。** `[0,1]` 内有限比例，控制时间重采样中高频 FFT bin 的余弦 taper，数量按比例取整。重采样时即使为 0，最后一个正 rFFT bin 也会被抑制，奇数长度同样如此；不保证保留所有频率。内部重采样比为 1 时此设置不改变信号。
:type freq_taper_frac: `float | int`
:param time_pad_frac: **默认 `0.0`；仅关键字参数。** `[0,1]` 内有限比例；重采样前在尾部补零，补零数为 `int(time_pad_frac*T)` 用户样本，之后移除。不增加公开记录长度或实际传播时长；内部重采样比为 1 时不改变信号。
:type time_pad_frac: `float | int`
:param time_taper: **默认 `False`；仅关键字参数。** 是否在时间重采样中应用非周期 Hann 窗：上采样后、下采样前应用。改变信号及其反传转置；仅接受布尔值，不接受整数 0/1。内部重采样比为 1 时不改变信号。
:type time_taper: `bool`
:param illumination: **默认 `None`；仅关键字参数。** 二维可选的新 `ScalarIllumination` 收集器；需启用梯度且存在可训练模型。收集 detached 的源、接收与几何乘积统计；forward 后 seal，一次 backward 后 reduce。三维只接受 `None`，其它值会抛出 `NotImplementedError`。`None` 不分配照明 buffer；统计量不是精确 Hessian，也不自动预条件梯度。 Torch 或 scalar 任意面零宽时必须为 None。
:type illumination: `ScalarIllumination | None`
:param backend: **默认 `None`；仅关键字参数。** None 保留原生行为，不自动选择 Torch；`'cuda'` 显式要求 CUDA 模型；`'torch'` 使用 CPU/CUDA 张量正演和 autograd。设备取自输入，不自动移动模型；不接受 `'cpu'` 后端。MPS 支持 Scalar2D accuracy=4 float32；Scalar3D 不支持 MPS。scalar/VRZ/VTI 的 Torch 路径须显式选择 full/checkpoint。
:type backend: `str | None`
:param execution: **默认 `'eager'`；仅关键字参数。** Torch 可选 `'eager'` 或 `'compile'`；compile 使用 PyTorch Inductor/AOTAutograd，失败不回退 eager。原生仅接受 eager。
:type execution: `str`
:param checkpoint_interval: **默认 `None`；仅关键字参数。** Torch checkpoint 分段长度，正整数内部时间步，拒绝 bool；None 表示 32。full 或原生传播必须保持 None。增大间隔会增加单段重算图，不能保证更省显存。
:type checkpoint_interval: `int | None`
:param compile_steps: **默认 `1`；仅关键字参数。** 正整数，拒绝 bool；Torch 每个执行段包含的连续内部时间步数。compile 可尝试 4，完整处理尾段；不改变 dt、nt、炮数或损失。原生只接受 1。
:type compile_steps: `int`
:param return_wavefield: **默认 `False`；仅关键字参数。** 布尔值。False 保留 `(records,)`；True 返回 `(terminal_physical_wavefield, records)`，`[-1]` 仍为记录。可微终态形状 `[B,*v.shape]`，按公开模型轴序，不含 PML/buffer/FD ghost；对应最后一个内部更新及源注入完成后的 `T*dt`。不返回前一时间层或 CPML 记忆。
:type return_wavefield: `bool`
:param forward_callback: **默认 `None`；仅关键字参数。** 接受 ScalarCallbackState 的 callable，在完整前向区间及最后一步更新后执行。物理波场为独立、detached、只读快照；回调中修改会报错。初始时刻和 Torch checkpoint 反传重算不触发回调。可微 loss 使用返回终态，不使用回调快照；时钟见下文。
:type forward_callback: `Callable | None`
:param callback_frequency: **默认 `1`；仅关键字参数。** 正整数，拒绝 bool，按用户采样步计数。内部间隔为 `CFL_ratio * callback_frequency`；即使末段不足一个间隔，也总在最终更新后回调。forward_callback 为 None 时仍验证该值。
:type callback_frequency: `int`
:returns: **默认 `(receiver_amplitudes,)`；`return_wavefield=True` 时为 `(terminal_physical_wavefield, receiver_amplitudes)`。** 记录为模型设备上的连续 float32 `[B,R,T]`，名义时间为 `0, dt, ..., (T-1)*dt`，`[-1]` 始终取记录。可选可微终态形状为 `[B,*v.shape]`，对应最后一次源更新之后的 `internal_nt*internal_dt = T*dt`，与最后一个记录样点时间不同。不含 PML/buffer/ghost，也不能单独重启传播。
:rtype: `tuple[torch.Tensor, ...]`
```

**Torch 路径：** 同一主入口设置 `backend="torch"` 并显式 `memory="full"` 或 `"checkpoint"`，可使用 CPU/CUDA，无需原生构建；导入仍会初始化已有可用的原生库。下方历史存储/原生 stream 限制只描述原生路径。完整选项和内存表见 [PyTorch 后端](pytorch-backend.md)。

(scalar-details)=
### 震源与自动微分

公开方程约定为 `u_tt = v² (Lap(u) - f)`。二维和三维用户均可训练 `f`；内部源增量为 `-v_source² * internal_dt² * U(f)`，其中 `U` 是内部上采样。不要在外部再次乘 `-v² dt²`。源位置的速度缩放链保留在模型梯度中。

步后记录在下采样前通过前置零、丢弃最后一步对齐用户时钟。时间重采样和移位的转置保留在 autograd 中；FFT 的非因果滤波可能使初始用户样本出现振铃。

模型梯度采用固定扩展模型、CFL 与 PML 的条件。重新生成物理边缘的 replicate padding 时，当前返回梯度不包含完整扩展链的转置累积，不能当作整个重建边界过程的全导数；切换 full 不改变这一限制。

各零宽面外部保留 `M=accuracy//2` 个固定零 ghost，不含该面的 PML/buffer。物理边缘样点仍活动，坐标不偏移。这是截断中心差分，不是奇反射：二阶的等效零平面在物理边缘之外一个网格间距；更高内部阶数不意味着奇镜像闭合或表面收敛阶。含零面时原生 full 使用完整网格历史；boundary 仍保存/重构所需面条带，不回退 full。任意零面均拒绝 illumination。原生 CUDA full/boundary 与 Torch CPU/CUDA full/checkpoint eager/compile 支持观测选项。MPS 资格仍限 Scalar2D float32、accuracy=4；未新增 MPS 设备验收。

回调状态提供 `wavefield`、`step`（整数已完成用户步）、`user_step`、`internal_step`、`internal_dt`、`step_ratio`、`dt`（internal_dt 的别名）及 `time=internal_step*internal_dt`。秒时间使用 `state.time`；CFL 比率大于 1 时 `state.step*state.dt` 不正确。比率为 2、频率为 3 时，在用户步 3,6,... / 内部步 6,12,... 后回调，并总包含最后一次更新。快照独立克隆，不传递 loss 梯度；DataParallel 中的回调闭包须自行处理并发观测记录。

(scalar-examples)=
### 示例

以下为已准备有效模型、采集和原生库后的调用片段；不是独立运行程序。所有省略的可选参数采用上方默认值。

```python
import starwave

receiver_amplitudes, = starwave.scalar(
    v, grid_spacing=10.0, dt=0.001,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
)
```

完整、无数据文件依赖的输入构造与一次 FWI 更新见[快速入门](quickstart.md)和[FWI](inversion/fwi.md)。默认 `pml_freq=25.0` 不会随源频率自动调整。


V18 的 CPU 示例：仅终态 loss、可训练二维源、全零面、空接收器与 checkpoint 安全观测。它只检查小规模梯度接线，不代表目标波场收敛或 GPU 验收。

<!-- scalar-dev:example:start -->
```python
import torch
import starwave

v = torch.full((12, 10), 1500.0, dtype=torch.float32, requires_grad=True)
source = torch.zeros((1, 1, 7), dtype=v.dtype)
source[0, 0, 1] = 1.0
source.requires_grad_()
src = torch.tensor([[[6, 5]]], dtype=torch.long)
rec = torch.empty((1, 0, 2), dtype=torch.long)
steps = []

def observe(state):
    assert state.wavefield.shape == (1, 12, 10)
    assert not state.wavefield.requires_grad
    steps.append((state.step, state.internal_step, state.time))

terminal, records = starwave.scalar(
    v, 10.0, 0.004,
    source_amplitudes=source, source_locations=src,
    receiver_locations=rec, accuracy=4, pml_width=0,
    pml_freq=15.0, max_vel=1600.0, backend="torch",
    memory="checkpoint", checkpoint_interval=4,
    return_wavefield=True, forward_callback=observe, callback_frequency=3,
)
assert terminal.shape == (1, 12, 10) and records.shape == (1, 0, 7)
assert [step for step, _, _ in steps] == [3, 6, 7]
assert torch.isfinite(terminal).all()
terminal.square().mean().backward()
assert len(steps) == 3  # No callbacks during checkpoint backward replay.
for gradient in (v.grad, source.grad):
    assert gradient is not None and torch.isfinite(gradient).all()
```
<!-- scalar-dev:example:end -->

授权的当前 dev 源码中，完整用户目标波场示例位于 `notebook/Scalar2D_Target_Wavefield.ipynb`，配套脚本为 `notebook/Scalar2D_Target_Wavefield.py`。请从该源码检出中打开；本公开手册不重新分发私有 notebook、报告素材或数据集。示例以终态波场目标优化源，回调快照仅作诊断。用户提供的 RTX 4060/A30 结果描述经审阅的功能分支，不是云端 GPU 重跑，也不是发布 wheel 验收。

(scalar-time-sampling)=
### CFL 与内部时间重采样

scalar 使用保守系数 0.6，按全部模型轴间距与 `max(abs(v))`（或经校验的 `max_vel`）计算时间上界。内部子步比 `r` 为满足该上界的正整数，`internal_dt=dt/r`、`internal_nt=T*r`。3D 不等间距会逐轴参与规划；`accuracy` 不会开启另一套公开 CFL 参数。时间规划、速度包络和 PML 系数是固定设置，不参与 autograd。

源波形先经 FFT 上采样，再施加源位置速度缩放；记录先做时间对齐，再降采样回 `[B,R,T]`。反传使用这些实际操作的转置，不能用简单重复或抽样代替。子步比为 1 时，合法 taper/padding 设置不改变信号；子步不能修复空间色散。

(scalar-memory)=
### 三维原生内存：full 与 Radius-M boundary

`M=accuracy//2` 是压力 Laplacian 的差分半径，和 `boundary_buffer`、`pml_width` 各司其职。boundary 保存六个宽度恰为 M 的压力面以及两个终态压力场；不保存完整时间体积，也不会失败后自动改为 full。full 保存逐内部步的未缩放 `Lap(u)` 体积历史。两种模式都保留传播/伴随工作场，三维 CPML 记忆使用方向条带；只训练源也会保存所选历史。显存还随内部时间步数和炮数增加，boundary 不保证大规模三维一定放得下。精确存储公式见{ref}`Scalar3D 重建说明 <reconstruction-scalar3d>`。

(scalar-3d-example)=
### 三维可运行调用：速度与源的一阶梯度

取得V18 scalar 源码、构建匹配原生库并确保可见逻辑 CUDA 设备 0 可用后，可运行下面的独立小例子。模型采用 `[x,y,z]`，间距为 `[dx,dy,dz]`，坐标是网格下标。loss 仅检查求导接线。此代码已检查语法与接口；本次文档维护未运行 GPU，不把断言视为已通过的数值验收。

```python
import torch
import starwave

starwave.prepare_native([0])  # Optional explicit device check.
device = torch.device("cuda:0")
v = torch.full((24, 20, 16), 1800.0, device=device,
               dtype=torch.float32, requires_grad=True)
t = torch.arange(96, device=device, dtype=torch.float32) * 0.001
a = (torch.pi * 15.0 * (t - 0.04)).square()
source_amplitudes = ((1 - 2 * a) * torch.exp(-a)).reshape(1, 1, -1)
source_amplitudes = source_amplitudes.detach().requires_grad_()
source_locations = torch.tensor([[[12, 10, 3]]], device=device,
                                dtype=torch.long)
receiver_locations = torch.tensor(
    [[[i, 10, 3] for i in range(4, 20)]], device=device, dtype=torch.long)
receiver_amplitudes, = starwave.scalar(
    v, grid_spacing=(10.0, 12.0, 8.0), dt=0.001,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    accuracy=4, pml_freq=15.0, pml_width=8, boundary_buffer=5,
    memory="boundary", max_vel=2000.0,
    freq_taper_frac=0.0, time_pad_frac=0.0, time_taper=False,
    illumination=None, backend=None, execution="eager",
    checkpoint_interval=None, compile_steps=1,
    return_wavefield=False, forward_callback=None, callback_frequency=1,
)
assert receiver_amplitudes.shape == (1, 16, 96)
loss = receiver_amplitudes.square().mean()
loss.backward()
assert v.grad is not None and source_amplitudes.grad is not None
assert torch.isfinite(receiver_amplitudes).all()
assert torch.isfinite(v.grad).all()
assert torch.isfinite(source_amplitudes.grad).all()
```

(scalar-notes)=
### 注意事项

- 原生路径仅 CUDA FP32 默认 stream；每次 forward 只支持一次一阶 backward。二维和三维均支持一次一阶源梯度。高阶导数、AMP、CUDA graphs、自定义 stream、CPU 传播、公开初始/重启状态不在支持范围；可选终态物理波场不是完整状态。
- 时间 CFL 子步不能弥补空间采样不足；空间分辨率警告是诊断，不是精度或稳定性证书。
- DataParallel 应把完整模型放入 Module，只拆分炮维采集；参见[多卡入门](inversion/dataparallel.md)。
- 源码/API 与有限 CPU 检查同用户提供的功能分支 GPU 报告分开记录。本次文档更新未重跑目标 GPU 传播，不建立合并 dev/wheel 的性能或 FWI 验收。

(born)=
## Born Function

本节为 V17 源码 API；[Release V17](https://github.com/StarrMoonn/StarWave/releases/tag/V17) 已发布，PyPI 8.0.0 尚待发布，已发布的 7.1.0 不含 Born。此常密度声学 Born 算子源自 [Deepwave 0.0.27](https://github.com/ar4/deepwave/tree/dc2f8af71e1fb67676c50a91a22bdd059bac8772)。`scatter` 表示速度扰动 δv，单位与 `v` 相同，不是相对速度平方或慢度平方扰动。背景算子与 `starwave.scalar` 不同，应使用本算子返回的背景记录构造相容观测；无需安装 Deepwave 包。

```{py:function} starwave.scalar_born(v: torch.Tensor, scatter: torch.Tensor, grid_spacing: Union[float, Sequence[float]], dt: float, source_amplitudes: Optional[torch.Tensor]=None, source_locations: Optional[torch.Tensor]=None, receiver_locations: Optional[torch.Tensor]=None, bg_receiver_locations: Optional[torch.Tensor]=None, accuracy: int=4, pml_width: Union[int, Sequence[int]]=20, pml_freq: Optional[float]=None, max_vel: Optional[float]=None, survey_pad: Optional[Union[int, Sequence[Optional[int]]]]=None, wavefield_0: Optional[torch.Tensor]=None, wavefield_m1: Optional[torch.Tensor]=None, psiz_m1: Optional[torch.Tensor]=None, psiy_m1: Optional[torch.Tensor]=None, psix_m1: Optional[torch.Tensor]=None, zetaz_m1: Optional[torch.Tensor]=None, zetay_m1: Optional[torch.Tensor]=None, zetax_m1: Optional[torch.Tensor]=None, wavefield_sc_0: Optional[torch.Tensor]=None, wavefield_sc_m1: Optional[torch.Tensor]=None, psiz_sc_m1: Optional[torch.Tensor]=None, psiy_sc_m1: Optional[torch.Tensor]=None, psix_sc_m1: Optional[torch.Tensor]=None, zetaz_sc_m1: Optional[torch.Tensor]=None, zetay_sc_m1: Optional[torch.Tensor]=None, zetax_sc_m1: Optional[torch.Tensor]=None, origin: Optional[Sequence[int]]=None, nt: Optional[int]=None, model_gradient_sampling_interval: int=1, freq_taper_frac: float=0.0, time_pad_frac: float=0.0, time_taper: bool=False, forward_callback: Optional[common.Callback]=None, backward_callback: Optional[common.Callback]=None, callback_frequency: int=1, python_backend: Union[Literal['eager', 'jit', 'compile'], bool]=False, storage_mode: Literal['device', 'cpu', 'disk', 'none']='device', storage_path: str='.', storage_compression: bool=False, *, memory: Literal['full', 'boundary', 'checkpoint']='full', backend=None, execution='eager', checkpoint_interval=None, compile_steps=1) -> Tuple[torch.Tensor, ...]

:param v: **必填；位置或关键字参数。** 背景速度张量，含 1–3 个空间轴，可带模型批次维；须为有限 float32/64。
:type v: `torch.Tensor`
:param scatter: **必填；位置或关键字参数。** 速度扰动 δv，空间形状、dtype 和设备与 v 一致；单位与 v 相同，不是相对速度平方或慢度平方扰动。
:type scatter: `torch.Tensor`
:param grid_spacing: **必填；位置或关键字参数。** 正标量，或按空间轴序给出每轴一个正间距。
:type grid_spacing: `Union[float, Sequence[float]]`
:param dt: **必填；位置或关键字参数。** 用户源/接收采样间隔；内部可能按 CFL 细分。
:type dt: `float`
:param source_amplitudes: **默认 `None`；位置或关键字参数。** 可选 forcing，形状 `[shot,source,time]`，dtype/device 与模型一致；源长度确定 nt。
:type source_amplitudes: `Optional[torch.Tensor]`
:param source_locations: **默认 `None`；位置或关键字参数。** 可选 int64 物理网格下标 `[shot,source,ndim]`；公共准备逻辑支持 `common.IGNORE_LOCATION`。
:type source_locations: `Optional[torch.Tensor]`
:param receiver_locations: **默认 `None`；位置或关键字参数。** 可选散射场接收坐标，int64 `[shot,receiver,ndim]`。
:type receiver_locations: `Optional[torch.Tensor]`
:param bg_receiver_locations: **默认 `None`；位置或关键字参数。** 可选背景场接收坐标，int64 `[shot,receiver,ndim]`。
:type bg_receiver_locations: `Optional[torch.Tensor]`
:param accuracy: **默认 `4`；位置或关键字参数。** 空间有限差分阶数，可选 2、4、6 或 8。
:type accuracy: `int`
:param pml_width: **默认 `20`；位置或关键字参数。** 非负整数，或按空间轴低端/高端顺序排列的 `2*ndim` 个面宽。某面为零时，该侧使用固定零 ghost 单元闭合。
:type pml_width: `Union[int, Sequence[int]]`
:param pml_freq: **默认 `None`；位置或关键字参数。** 非负 PML 参考频率，单位 Hz，允许零；None 沿用上游 25 Hz 默认值并给出警告。
:type pml_freq: `Optional[float]`
:param max_vel: **默认 `None`；位置或关键字参数。** 可选固定 CFL/PML 速度上界；None 从 max(abs(v)) 推导。显式值取绝对值，零被拒绝；小于实际 max(abs(v)) 时给出警告，而非按 scalar 的规则拒绝。数值规划不参与求导。
:type max_vel: `Optional[float]`
:param survey_pad: **默认 `None`；位置或关键字参数。** 可选基于采集几何的传播域裁剪余量；接受整数或逐面序列，序列中的 None 表示保留该侧完整模型。
:type survey_pad: `Optional[Union[int, Sequence[Optional[int]]]]`
:param wavefield_0: **默认 `None`；位置或关键字参数。** 可选背景场当前初始波场；包含 PML、不含有限差分 halo；可求导。
:type wavefield_0: `Optional[torch.Tensor]`
:param wavefield_m1: **默认 `None`；位置或关键字参数。** 可选背景场前一时刻初始波场；包含 PML、不含有限差分 halo；可求导。
:type wavefield_m1: `Optional[torch.Tensor]`
:param psiz_m1: **默认 `None`；位置或关键字参数。** 可选背景场 psiz 初始 PML 记忆（仅三维）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type psiz_m1: `Optional[torch.Tensor]`
:param psiy_m1: **默认 `None`；位置或关键字参数。** 可选背景场 psiy 初始 PML 记忆（二维/三维）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type psiy_m1: `Optional[torch.Tensor]`
:param psix_m1: **默认 `None`；位置或关键字参数。** 可选背景场 psix 初始 PML 记忆（所有维数）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type psix_m1: `Optional[torch.Tensor]`
:param zetaz_m1: **默认 `None`；位置或关键字参数。** 可选背景场 zetaz 初始 PML 记忆（仅三维）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type zetaz_m1: `Optional[torch.Tensor]`
:param zetay_m1: **默认 `None`；位置或关键字参数。** 可选背景场 zetay 初始 PML 记忆（二维/三维）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type zetay_m1: `Optional[torch.Tensor]`
:param zetax_m1: **默认 `None`；位置或关键字参数。** 可选背景场 zetax 初始 PML 记忆（所有维数）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type zetax_m1: `Optional[torch.Tensor]`
:param wavefield_sc_0: **默认 `None`；位置或关键字参数。** 可选散射场当前初始波场；包含 PML、不含有限差分 halo；可求导。
:type wavefield_sc_0: `Optional[torch.Tensor]`
:param wavefield_sc_m1: **默认 `None`；位置或关键字参数。** 可选散射场前一时刻初始波场；包含 PML、不含有限差分 halo；可求导。
:type wavefield_sc_m1: `Optional[torch.Tensor]`
:param psiz_sc_m1: **默认 `None`；位置或关键字参数。** 可选散射场 psiz 初始 PML 记忆（仅三维）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type psiz_sc_m1: `Optional[torch.Tensor]`
:param psiy_sc_m1: **默认 `None`；位置或关键字参数。** 可选散射场 psiy 初始 PML 记忆（二维/三维）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type psiy_sc_m1: `Optional[torch.Tensor]`
:param psix_sc_m1: **默认 `None`；位置或关键字参数。** 可选散射场 psix 初始 PML 记忆（所有维数）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type psix_sc_m1: `Optional[torch.Tensor]`
:param zetaz_sc_m1: **默认 `None`；位置或关键字参数。** 可选散射场 zetaz 初始 PML 记忆（仅三维）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type zetaz_sc_m1: `Optional[torch.Tensor]`
:param zetay_sc_m1: **默认 `None`；位置或关键字参数。** 可选散射场 zetay 初始 PML 记忆（二维/三维）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type zetay_sc_m1: `Optional[torch.Tensor]`
:param zetax_sc_m1: **默认 `None`；位置或关键字参数。** 可选散射场 zetax 初始 PML 记忆（所有维数）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type zetax_sc_m1: `Optional[torch.Tensor]`
:param origin: **默认 `None`；位置或关键字参数。** 所给初始状态在物理模型内的起点，按空间轴序给出网格下标。
:type origin: `Optional[Sequence[int]]`
:param nt: **默认 `None`；位置或关键字参数。** 无源时的用户采样数；如同时给出源，必须与源长度一致。允许零；无源调用仍须用源几何或初始状态确定炮数。
:type nt: `Optional[int]`
:param model_gradient_sampling_interval: **默认 `1`；位置或关键字参数。** 以内部传播步计数的正整数 M。原生 M>1 请求近似采样模型梯度；统一 Torch 后端只接受 1；旧张量参考路径仍计算完整 AD。
:type model_gradient_sampling_interval: `int`
:param freq_taper_frac: **默认 `0.0`；位置或关键字参数。** FFT 时间重采样的频域 taper 比例。
:type freq_taper_frac: `float`
:param time_pad_frac: **默认 `0.0`；位置或关键字参数。** FFT 时间重采样的时间补零比例。
:type time_pad_frac: `float`
:param time_taper: **默认 `False`；位置或关键字参数。** 在重采样时应用可选时间 taper。
:type time_taper: `bool`
:param forward_callback: **默认 `None`；位置或关键字参数。** 可选 `common.Callback`，在正演块开始前观察状态；时钟与修改限制见下文。
:type forward_callback: `Optional[common.Callback]`
:param backward_callback: **默认 `None`；位置或关键字参数。** 原生 full 与原生 CUDA boundary 支持的观察回调；Python 参考与统一 Torch 后端拒绝此项。
:type backward_callback: `Optional[common.Callback]`
:param callback_frequency: **默认 `1`；位置或关键字参数。** 按用户采样时钟计数的正整数回调间隔，与 M 无关。
:type callback_frequency: `int`
:param python_backend: **默认 `False`；位置或关键字参数。** 旧路径选择器 False/eager/jit/compile/True；不能与显式 cuda 或统一 torch 混用。True 根据可用原生构建的 OpenMP 元数据选择 compile 或 jit。
:type python_backend: `Union[Literal['eager', 'jit', 'compile'], bool]`
:param storage_mode: **默认 `'device'`；位置或关键字参数。** 原生 full 历史存储位置：device/cpu/disk/none。其它路径要求 device；none 省去传播过程对模型梯度的贡献。
:type storage_mode: `Literal['device', 'cpu', 'disk', 'none']`
:param storage_path: **默认 `'.'`；位置或关键字参数。** 原生 full 磁盘历史目录；不控制模型或源的设备传输。
:type storage_path: `str`
:param storage_compression: **默认 `False`；位置或关键字参数。** 原生 full 历史量化，可能使模型梯度近似；其它路径要求 False。
:type storage_compression: `bool`
:param memory: **默认 `'full'`；仅关键字参数。** 原生 CPU：full；原生 CUDA：full/boundary；统一 Torch：full/checkpoint；旧参考路径：full。
:type memory: `Literal['full', 'boundary', 'checkpoint']`
:param backend: **默认 `None`；仅关键字参数。** None 保留按输入设备选择原生实现或由 python_backend 选择旧参考路径；cuda 要求 CUDA；torch 显式使用 CPU/CUDA 张量执行。不接受 cpu/native_cpu，也不自动回退。
:type backend: `str | None`
:param execution: **默认 `'eager'`；仅关键字参数。** 统一 Torch 可选 eager/compile；其它路径要求 eager。编译失败直接报错。
:type execution: `str | None`
:param checkpoint_interval: **默认 `None`；仅关键字参数。** 仅统一 Torch checkpoint 支持的正整数内部步分段长度；None 使用 32。
:type checkpoint_interval: `int | None`
:param compile_steps: **默认 `1`；仅关键字参数。** Torch 有界编译块内的正整数内部步数；其它路径要求 1。
:type compile_steps: `int | None`
:returns: 完整背景场/散射场/PML 终态元组，最后为背景记录与散射记录；一维/二维/三维分别返回 10/14/18 个张量。
:rtype: `Tuple[torch.Tensor, ...]`
```

(born-backends)=
### 后端与历史存储

| 选择方式 | 设备 / dtype | 内存策略 | 执行方式 |
|---|---|---|---|
| `backend=None`, `python_backend=False` | CPU float32/64 | `full` | native |
| `backend=None` 使用 CUDA 输入，或 `backend="cuda"` | CUDA float32/64 | `full`, `boundary` | native |
| `backend="torch"` | CPU/CUDA float32/64 | `full`, `checkpoint` | `eager`, `compile` |
| `backend=None`, `python_backend="eager"/"jit"/"compile"` | CPU/CUDA float32/64 | `full` | 旧参考路径 |

所有路径均支持一维/二维/三维（`1D/2D/3D`）以及 2/4/6/8 阶空间差分。模型、源与初始状态的浮点 dtype/device 必须一致。不支持 MPS、AMP、浮点模型/源/初始状态的隐式设备传输或静默回退；几何坐标可归一化到模型设备。原生/参考路径要求 `execution="eager"`、`compile_steps=1`、`checkpoint_interval=None`。统一 Torch 要求 `python_backend=False`、模型梯度间隔 1、device 存储且不压缩。分块/间隔的正整数设置拒绝 bool。`python_backend=True` 根据可用原生构建元数据中的 OpenMP 信息选择 compile/jit，不进行运行时源码哈希核验。

原生 full 支持 device/cpu/disk/none 历史存储和可选量化；原生 CPU 的 cpu 历史归一化为 device 存储。压缩可能使模型梯度近似。`storage_mode="none"` 省去传播过程对模型梯度的贡献，尽管源缩放仍可能贡献部分梯度；不能以它替代完整梯度。原生 CUDA 仅支持 full 与 boundary，不支持原生 checkpoint。

boundary 仅支持 CUDA、device 存储且不压缩，要求裁剪后每个物理维度均大于 `accuracy`。即使采样模型梯度，也保存/重构每个内部时间步。小体积、厚 PML 或 full 历史高度抽稀时，boundary 不一定更省内存。所有原生导数仅支持一阶。

Torch checkpoint 保留完整背景场/散射场/PML 状态并重放正演分段，不丢弃初始状态、源或模型梯度。首次显式编译可能较慢，错误直接抛出。Born CPU Inductor 使用局部融合限制规避已确认的反传融合问题，CUDA 保持正常融合。这是实际编译，不是 eager 回退，也不构成性能承诺。

(born-conventions)=
### 模型、源与边界

轴序为 `[x]`、`[y,x]`、`[z,y,x]`，坐标按此顺序索引物理网格。共享模型、批次为 1 的模型和逐炮模型遵循公共批次推断。无源传播除 `nt` 外，还须提供能确定炮数的初始状态或源几何；仅模型和接收坐标不能确定炮数。每炮内有效源位置必须唯一，每组接收坐标内的有效位置也必须唯一；`common.IGNORE_LOCATION` 项不参与唯一性检查。初始状态包含 PML、不含有限差分 halo。z 记忆仅在三维启用，y 在二维/三维启用，x 在所有维数启用；被忽略的非 PML 记忆位置导数为零。

PML 某面宽为零时，该侧有限差分 ghost 单元保持零，第一行物理网格仍参与传播。二阶时负镜像壁面位于第一物理行外一格；高阶沿用上游零 ghost 闭合，不能宣称具有相应奇延拓高阶表面精度。这不等同于弹性无牵引边界。

```text
u_next  = v² dt_internal² L(u) + 2u - u_previous - v² dt_internal² f
us_next = v² dt_internal² L(us) + 2us - us_previous
          + 2v δv dt_internal² L(u) - 2v δv dt_internal² f
```

背景场与散射场使用同一固定版本的标量 PML 算子 L。散射源包含速度相关源缩放的导数。速度采用 replicate 扩边，scatter 则补零：若物理边缘 δv 非零，其结果不是“将扰动后速度重新 replicate 到 PML”这一完整包装过程的导数。做有限差分线性化测试时应固定外部区域，或使扰动在物理边界消失，并使用固定且足够大的 `max_vel` 保持数值规划不变。

(born-gradients)=
### 按内部步采样模型梯度

原生采样对任意正整数间隔 M 都按内部传播步计数。默认 `M=1` 保存并成像每个内部步；在历史启用且无压缩、数值规划固定时，对应一阶离散模型 VJP，仅有浮点/重构误差。`M>1` 明确请求近似的采样加权模型梯度；full 分配 `ceil(N_internal/M)` 个历史槽位，即使最后一段不足 M 步也保持权重 M。它不省略正演/伴随时间步、接收样点、最终状态或源/初始状态伴随。

Deepwave 0.0.27 的用户间隔 k 对应有效步长 `CFL_ratio * k`。本接口若要使用该采样步长，应传 `M=CFL_ratio*k`；StarWave 默认内部步长 1 是为梯度精度作出的明确适配，不能称为与上游采样语义完全相同。统一 Torch 拒绝 M≠1；旧张量参考路径接受该参数，但仍计算完整张量 AD。

例如 `dt=0.003`、`nt=2000`、CFL 比为 2 时，full 和 boundary 均以 0.0015 s 执行 4000 个内部步，返回 2000 个样点。M=2 只减少模型成像采样；boundary 仍重构全部 4000 步，不承诺 2 倍加速。

(born-callbacks)=
### 回调时钟与限制

正演回调在分块开始前收到 `starwave.common.CallbackState`。原生 full 与原生 CUDA boundary 还支持带模型梯度视图的反传回调，Python 参考和统一 Torch 路径拒绝反传回调。Torch checkpoint 回调仅在真正的正演中执行一次，不在反传重放中重复。boundary 回调使用受检查的只读快照，修改会报错；其它路径也应仅将回调用于观察。

回调频率和 `state.step` 使用用户采样时钟，与 M 无关。`state.dt` 则是内部 CFL 时间步；子步比大于 1 时，`state.step * state.dt` 不等于物理时间。生成物理时间戳时应保留用户 dt。

(born-returns)=
### 完整返回顺序

元组先包含 `4+4*ndim` 个终态张量，再包含两个记录张量；未启用的空间轴不占空槽：

1. 背景场当前与前一时刻波场
2. 按 z/y/x 有效轴顺序排列的背景 psi，再按同序排列背景 zeta
3. 散射场当前与前一时刻波场
4. 按有效轴顺序排列的散射 psi，再排列散射 zeta
5. 背景记录 `out[-2]`，再为散射记录 `out[-1]`

一维/二维/三维分别返回 10/14/18 个张量。记录形状为 `[shot,receiver,time]`，采样更新前的当前波场；未给出的接收类型返回空张量。最终状态包含 PML、不含有限差分 halo。继续传播时须保持内部采样一致：分别对源分段进行 FFT 重采样，通常不等价于对完整信号一次重采样。

(born-module)=
### ScalarBorn

```{py:class} starwave.ScalarBorn(v: torch.Tensor, scatter: torch.Tensor, grid_spacing: Union[float, Sequence[float]], v_requires_grad: bool=False, scatter_requires_grad: bool=False, storage_mode: Literal['device', 'cpu', 'disk', 'none']='device', storage_path: str='.', storage_compression: bool=False, *, memory: Literal['full', 'boundary', 'checkpoint']='full', backend=None, execution='eager', checkpoint_interval=None, compile_steps=1)

:param v: **必填；位置或关键字参数。** 背景速度张量，含 1–3 个空间轴，可带模型批次维；须为有限 float32/64。
:type v: `torch.Tensor`
:param scatter: **必填；位置或关键字参数。** 速度扰动 δv，空间形状、dtype 和设备与 v 一致；单位与 v 相同，不是相对速度平方或慢度平方扰动。
:type scatter: `torch.Tensor`
:param grid_spacing: **必填；位置或关键字参数。** 正标量，或按空间轴序给出每轴一个正间距。
:type grid_spacing: `Union[float, Sequence[float]]`
:param v_requires_grad: **默认 `False`；位置或关键字参数。** 模块持有的速度 Parameter 是否需要梯度。
:type v_requires_grad: `bool`
:param scatter_requires_grad: **默认 `False`；位置或关键字参数。** 模块持有的 scatter Parameter 是否需要梯度。
:type scatter_requires_grad: `bool`
:param storage_mode: **默认 `'device'`；位置或关键字参数。** 原生 full 历史存储位置：device/cpu/disk/none。其它路径要求 device；none 省去传播过程对模型梯度的贡献。
:type storage_mode: `Literal['device', 'cpu', 'disk', 'none']`
:param storage_path: **默认 `'.'`；位置或关键字参数。** 原生 full 磁盘历史目录；不控制模型或源的设备传输。
:type storage_path: `str`
:param storage_compression: **默认 `False`；位置或关键字参数。** 原生 full 历史量化，可能使模型梯度近似；其它路径要求 False。
:type storage_compression: `bool`
:param memory: **默认 `'full'`；仅关键字参数。** 原生 CPU：full；原生 CUDA：full/boundary；统一 Torch：full/checkpoint；旧参考路径：full。
:type memory: `Literal['full', 'boundary', 'checkpoint']`
:param backend: **默认 `None`；仅关键字参数。** None 保留按输入设备选择原生实现或由 python_backend 选择旧参考路径；cuda 要求 CUDA；torch 显式使用 CPU/CUDA 张量执行。不接受 cpu/native_cpu，也不自动回退。
:type backend: `str | None`
:param execution: **默认 `'eager'`；仅关键字参数。** 统一 Torch 可选 eager/compile；其它路径要求 eager。编译失败直接报错。
:type execution: `str | None`
:param checkpoint_interval: **默认 `None`；仅关键字参数。** 仅统一 Torch checkpoint 支持的正整数内部步分段长度；None 使用 32。
:type checkpoint_interval: `int | None`
:param compile_steps: **默认 `1`；仅关键字参数。** Torch 有界编译块内的正整数内部步数；其它路径要求 1。
:type compile_steps: `int | None`
:returns: 持有自身 v/scatter Parameter 包装的模块，不克隆底层存储。
:rtype: `starwave.ScalarBorn`
```

模块持有 Parameter 包装，但不克隆数据。需要独立存储时应先克隆输入；梯度累积在模块参数上，不沿原始输入张量的历史回传。源码未标注类型的选择参数在展示签名中也不添加注解，接受的值见对应 type 字段。

(born-forward)=
### ScalarBorn.forward

```{py:method} starwave.ScalarBorn.forward(dt: float, source_amplitudes: Optional[torch.Tensor]=None, source_locations: Optional[torch.Tensor]=None, receiver_locations: Optional[torch.Tensor]=None, bg_receiver_locations: Optional[torch.Tensor]=None, accuracy: int=4, pml_width: Union[int, Sequence[int]]=20, pml_freq: Optional[float]=None, max_vel: Optional[float]=None, survey_pad: Optional[Union[int, Sequence[Optional[int]]]]=None, wavefield_0: Optional[torch.Tensor]=None, wavefield_m1: Optional[torch.Tensor]=None, psiz_m1: Optional[torch.Tensor]=None, psiy_m1: Optional[torch.Tensor]=None, psix_m1: Optional[torch.Tensor]=None, zetaz_m1: Optional[torch.Tensor]=None, zetay_m1: Optional[torch.Tensor]=None, zetax_m1: Optional[torch.Tensor]=None, wavefield_sc_0: Optional[torch.Tensor]=None, wavefield_sc_m1: Optional[torch.Tensor]=None, psiz_sc_m1: Optional[torch.Tensor]=None, psiy_sc_m1: Optional[torch.Tensor]=None, psix_sc_m1: Optional[torch.Tensor]=None, zetaz_sc_m1: Optional[torch.Tensor]=None, zetay_sc_m1: Optional[torch.Tensor]=None, zetax_sc_m1: Optional[torch.Tensor]=None, origin: Optional[Sequence[int]]=None, nt: Optional[int]=None, model_gradient_sampling_interval: int=1, freq_taper_frac: float=0.0, time_pad_frac: float=0.0, time_taper: bool=False, forward_callback: Optional[common.Callback]=None, backward_callback: Optional[common.Callback]=None, callback_frequency: int=1, python_backend: Union[Literal['eager', 'jit', 'compile'], bool]=False, *, memory=None, backend=None, execution=None, checkpoint_interval=None, compile_steps=None) -> Tuple[torch.Tensor, ...]

:param dt: **必填；位置或关键字参数。** 用户源/接收采样间隔；内部可能按 CFL 细分。
:type dt: `float`
:param source_amplitudes: **默认 `None`；位置或关键字参数。** 可选 forcing，形状 `[shot,source,time]`，dtype/device 与模型一致；源长度确定 nt。
:type source_amplitudes: `Optional[torch.Tensor]`
:param source_locations: **默认 `None`；位置或关键字参数。** 可选 int64 物理网格下标 `[shot,source,ndim]`；公共准备逻辑支持 `common.IGNORE_LOCATION`。
:type source_locations: `Optional[torch.Tensor]`
:param receiver_locations: **默认 `None`；位置或关键字参数。** 可选散射场接收坐标，int64 `[shot,receiver,ndim]`。
:type receiver_locations: `Optional[torch.Tensor]`
:param bg_receiver_locations: **默认 `None`；位置或关键字参数。** 可选背景场接收坐标，int64 `[shot,receiver,ndim]`。
:type bg_receiver_locations: `Optional[torch.Tensor]`
:param accuracy: **默认 `4`；位置或关键字参数。** 空间有限差分阶数，可选 2、4、6 或 8。
:type accuracy: `int`
:param pml_width: **默认 `20`；位置或关键字参数。** 非负整数，或按空间轴低端/高端顺序排列的 `2*ndim` 个面宽。某面为零时，该侧使用固定零 ghost 单元闭合。
:type pml_width: `Union[int, Sequence[int]]`
:param pml_freq: **默认 `None`；位置或关键字参数。** 非负 PML 参考频率，单位 Hz，允许零；None 沿用上游 25 Hz 默认值并给出警告。
:type pml_freq: `Optional[float]`
:param max_vel: **默认 `None`；位置或关键字参数。** 可选固定 CFL/PML 速度上界；None 从 max(abs(v)) 推导。显式值取绝对值，零被拒绝；小于实际 max(abs(v)) 时给出警告，而非按 scalar 的规则拒绝。数值规划不参与求导。
:type max_vel: `Optional[float]`
:param survey_pad: **默认 `None`；位置或关键字参数。** 可选基于采集几何的传播域裁剪余量；接受整数或逐面序列，序列中的 None 表示保留该侧完整模型。
:type survey_pad: `Optional[Union[int, Sequence[Optional[int]]]]`
:param wavefield_0: **默认 `None`；位置或关键字参数。** 可选背景场当前初始波场；包含 PML、不含有限差分 halo；可求导。
:type wavefield_0: `Optional[torch.Tensor]`
:param wavefield_m1: **默认 `None`；位置或关键字参数。** 可选背景场前一时刻初始波场；包含 PML、不含有限差分 halo；可求导。
:type wavefield_m1: `Optional[torch.Tensor]`
:param psiz_m1: **默认 `None`；位置或关键字参数。** 可选背景场 psiz 初始 PML 记忆（仅三维）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type psiz_m1: `Optional[torch.Tensor]`
:param psiy_m1: **默认 `None`；位置或关键字参数。** 可选背景场 psiy 初始 PML 记忆（二维/三维）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type psiy_m1: `Optional[torch.Tensor]`
:param psix_m1: **默认 `None`；位置或关键字参数。** 可选背景场 psix 初始 PML 记忆（所有维数）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type psix_m1: `Optional[torch.Tensor]`
:param zetaz_m1: **默认 `None`；位置或关键字参数。** 可选背景场 zetaz 初始 PML 记忆（仅三维）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type zetaz_m1: `Optional[torch.Tensor]`
:param zetay_m1: **默认 `None`；位置或关键字参数。** 可选背景场 zetay 初始 PML 记忆（二维/三维）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type zetay_m1: `Optional[torch.Tensor]`
:param zetax_m1: **默认 `None`；位置或关键字参数。** 可选背景场 zetax 初始 PML 记忆（所有维数）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type zetax_m1: `Optional[torch.Tensor]`
:param wavefield_sc_0: **默认 `None`；位置或关键字参数。** 可选散射场当前初始波场；包含 PML、不含有限差分 halo；可求导。
:type wavefield_sc_0: `Optional[torch.Tensor]`
:param wavefield_sc_m1: **默认 `None`；位置或关键字参数。** 可选散射场前一时刻初始波场；包含 PML、不含有限差分 halo；可求导。
:type wavefield_sc_m1: `Optional[torch.Tensor]`
:param psiz_sc_m1: **默认 `None`；位置或关键字参数。** 可选散射场 psiz 初始 PML 记忆（仅三维）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type psiz_sc_m1: `Optional[torch.Tensor]`
:param psiy_sc_m1: **默认 `None`；位置或关键字参数。** 可选散射场 psiy 初始 PML 记忆（二维/三维）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type psiy_sc_m1: `Optional[torch.Tensor]`
:param psix_sc_m1: **默认 `None`；位置或关键字参数。** 可选散射场 psix 初始 PML 记忆（所有维数）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type psix_sc_m1: `Optional[torch.Tensor]`
:param zetaz_sc_m1: **默认 `None`；位置或关键字参数。** 可选散射场 zetaz 初始 PML 记忆（仅三维）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type zetaz_sc_m1: `Optional[torch.Tensor]`
:param zetay_sc_m1: **默认 `None`；位置或关键字参数。** 可选散射场 zetay 初始 PML 记忆（二维/三维）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type zetay_sc_m1: `Optional[torch.Tensor]`
:param zetax_sc_m1: **默认 `None`；位置或关键字参数。** 可选散射场 zetax 初始 PML 记忆（所有维数）；有效位置可求导，被忽略的非 PML 位置导数为零。
:type zetax_sc_m1: `Optional[torch.Tensor]`
:param origin: **默认 `None`；位置或关键字参数。** 所给初始状态在物理模型内的起点，按空间轴序给出网格下标。
:type origin: `Optional[Sequence[int]]`
:param nt: **默认 `None`；位置或关键字参数。** 无源时的用户采样数；如同时给出源，必须与源长度一致。允许零；无源调用仍须用源几何或初始状态确定炮数。
:type nt: `Optional[int]`
:param model_gradient_sampling_interval: **默认 `1`；位置或关键字参数。** 以内部传播步计数的正整数 M。原生 M>1 请求近似采样模型梯度；统一 Torch 后端只接受 1；旧张量参考路径仍计算完整 AD。
:type model_gradient_sampling_interval: `int`
:param freq_taper_frac: **默认 `0.0`；位置或关键字参数。** FFT 时间重采样的频域 taper 比例。
:type freq_taper_frac: `float`
:param time_pad_frac: **默认 `0.0`；位置或关键字参数。** FFT 时间重采样的时间补零比例。
:type time_pad_frac: `float`
:param time_taper: **默认 `False`；位置或关键字参数。** 在重采样时应用可选时间 taper。
:type time_taper: `bool`
:param forward_callback: **默认 `None`；位置或关键字参数。** 可选 `common.Callback`，在正演块开始前观察状态；时钟与修改限制见下文。
:type forward_callback: `Optional[common.Callback]`
:param backward_callback: **默认 `None`；位置或关键字参数。** 原生 full 与原生 CUDA boundary 支持的观察回调；Python 参考与统一 Torch 后端拒绝此项。
:type backward_callback: `Optional[common.Callback]`
:param callback_frequency: **默认 `1`；位置或关键字参数。** 按用户采样时钟计数的正整数回调间隔，与 M 无关。
:type callback_frequency: `int`
:param python_backend: **默认 `False`；位置或关键字参数。** 旧路径选择器 False/eager/jit/compile/True；不能与显式 cuda 或统一 torch 混用。True 根据可用原生构建的 OpenMP 元数据选择 compile 或 jit。
:type python_backend: `Union[Literal['eager', 'jit', 'compile'], bool]`
:param memory: **默认 `None`；仅关键字参数。** 省略或显式传入 None 均继承构造时设置，只有非 None 值才覆盖。原生 CPU：full；原生 CUDA：full/boundary；统一 Torch：full/checkpoint；旧参考路径：full。
:type memory: `str | None`
:param backend: **默认 `None`；仅关键字参数。** 省略或显式传入 None 均继承构造时设置，只有非 None 值才覆盖。None 保留按输入设备选择原生实现或由 python_backend 选择旧参考路径；cuda 要求 CUDA；torch 显式使用 CPU/CUDA 张量执行。不接受 cpu/native_cpu，也不自动回退。
:type backend: `str | None`
:param execution: **默认 `None`；仅关键字参数。** 省略或显式传入 None 均继承构造时设置，只有非 None 值才覆盖。统一 Torch 可选 eager/compile；其它路径要求 eager。编译失败直接报错。
:type execution: `str | None`
:param checkpoint_interval: **默认 `None`；仅关键字参数。** 省略或显式传入 None 均继承构造时设置，只有非 None 值才覆盖。仅统一 Torch checkpoint 支持的正整数内部步分段长度；None 使用 32。
:type checkpoint_interval: `int | None`
:param compile_steps: **默认 `None`；仅关键字参数。** 省略或显式传入 None 均继承构造时设置，只有非 None 值才覆盖。Torch 有界编译块内的正整数内部步数；其它路径要求 1。
:type compile_steps: `int | None`
:returns: 完整背景场/散射场/PML 终态元组，最后为背景记录与散射记录；一维/二维/三维分别返回 10/14/18 个张量。
:rtype: `Tuple[torch.Tensor, ...]`
```

forward 的 None 覆盖值表示继承构造设置，不能将已保存的 checkpoint 间隔或 backend 重置为 None。storage_mode、storage_path 和 storage_compression 仅由构造时设置，不提供逐次调用覆盖。

(born-example)=
### 完整 CPU checkpoint 示例

```python
import torch
from starwave import scalar_born

v = torch.full((20, 24), 1500.0, dtype=torch.float64, requires_grad=True)
scatter = torch.full_like(v, 10.0, requires_grad=True)
source = torch.zeros((1, 1, 12), dtype=v.dtype)
source[0, 0, 1] = 1.0
source.requires_grad_()
locations = torch.tensor([[[8, 10]]], dtype=torch.int64)
out = scalar_born(
    v, scatter, 10.0, 0.0005,
    source_amplitudes=source, source_locations=locations,
    receiver_locations=locations, bg_receiver_locations=locations,
    accuracy=4, pml_width=4, pml_freq=20.0, max_vel=1600.0,
    backend="torch", memory="checkpoint", checkpoint_interval=4,
)
assert len(out) == 14
assert out[-2].shape == out[-1].shape == (1, 1, 12)
loss = out[-2].square().sum() + out[-1].square().sum()
loss.backward()
assert all(x.grad is not None and torch.isfinite(x.grad).all()
           for x in (v, scatter, source))
```

(born-build)=
### 原生构建与导入初始化

```bash
# 已获准取得的完整 V17 源码；按实际 GPU SM 修改。
python compile_born.py --arch 80
# 或只构建原生 CPU full：
python compile_born.py --cpu-only
```

Born 已并入主 Python 包，原生实现使用独立库。冻结 V17（`27c2e14b184b94d8031a5b3a83c25b1ed3d4ac67`）的根目录 `compile_all.py` 只构建 core/elastic/GSLS，Born 须另行执行上面的命令。当前开发版 `3f2b24d` 的 `compile_all.py` 已包含 core/elastic/GSLS/Born；不能将这一后续构建行为倒写为冻结 V17 的契约。独立 Born 源码包的构建入口仅支持 CPU；VRR 仍单独构建。纯 Torch 不需要 Born 原生构建，CPU 输入也不需要 GPU。

显式构建后，V17 的 `import starwave` 会加载可用的 core/elastic/Born/GSLS 库，并在返回前完成共享分派绑定。Born 可选 boundary 入口签名由对应调用路径在执行前绑定。已安装且可导入的独立 VRR 扩展也会准备好，其命名空间不变。普通首次并行传播无需串行预热或强制手工 prepare。导入不编译、不初始化 CUDA、不选择设备、不分配波场，也不传播。未构建库会跳过以允许受支持的纯 Torch 使用；已有但失效的库不会静默替换。构建路径应在导入前配置；更新后重新构建并重启 Python/Jupyter。运行时不监控源码/库变化，也不在每次传播时重新计算源码哈希。可选 prepare 辅助函数仍可用于显式设备检查。

(born-scope)=
### 版本与验证范围

本节记录冻结 V17 源码 `27c2e14b184b94d8031a5b3a83c25b1ed3d4ac67`（`0.1.0.dev17`），已发布为 [Release V17](https://github.com/StarrMoonn/StarWave/releases/tag/V17)。源码发布、原生编译、公开 wheel 发布、干净安装、CPU 数值验证与目标 GPU/多卡验证分别属于不同证据；文档构建本身不能建立这些运行结论。PyPI 8.0.0 在发布记录确认核验产物前仍为待发布；7.1.0 wheel 不会因本次文档更新获得 Born。原生高阶 AD、MPS、AMP 和性能保证不在本契约内。

(vrz)=
## VRZ Function

```{py:function} starwave.vrz(v: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, impedance: torch.Tensor | None=None, density: torch.Tensor | None=None, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, accuracy: int=8, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float | int=0.0, time_pad_frac: float | int=0.0, time_taper: bool=False, illumination: None=None, backend: str | None=None, execution: str='eager', checkpoint_interval: int | None=None, compile_steps: int=1) -> tuple[torch.Tensor]

使用速度和阻抗（或密度）进行二维声学传播，支持多炮批处理。恰好指定 `impedance` 或 `density` 一种参数化。输出可对所选模型参数求一次一阶梯度；源波形和采集坐标保持固定。

:param v: **必填；位置或关键字参数。** 二维模型 `[N0,N1]`，每个维度至少为 2；float32（原生 CUDA；Torch 为 CPU/CUDA）。典型单位 m/s。坐标 `[i,j]` 直接索引 `v[i,j]`；不自动推断物理轴序。 速度必须正且有限；不接受 scalar 的 signed/zero 模型域。设 `requires_grad=True` 时请求模型梯度。
:type v: `torch.Tensor`
:param grid_spacing: **必填；位置或关键字参数。** 正且有限的网格间距，典型单位 m。接受单值或两个相等的值；scalar/VRZ 不支持两轴不等间距。拒绝布尔值、零/负值和长度不匹配。
:type grid_spacing: `float | int | list[float] | tuple[float, ...]`
:param dt: **必填；位置或关键字参数。** 正且有限的用户输入/输出采样间隔，典型单位 s。内部可能按 CFL 采用更小的时间步，返回间隔仍为此值。不能传入 Tensor 或布尔值，不对其求导。
:type dt: `float | int`
:param impedance: **默认 `None`；仅关键字参数。** 声阻抗 `Z`，float32（原生 CUDA；Torch 为 CPU/CUDA），与 `v` 形状和设备相同，正且有限。典型 SI 单位 kg/(m²·s)，与 `v*rho` 一致。与 `density` 恰好给出一个；两者同时提供或同时省略均不合法。训练此输入表示独立的速度/阻抗参数化。
:type impedance: `torch.Tensor | None`
:param density: **默认 `None`；仅关键字参数。** 密度 `rho`，float32（原生 CUDA；Torch 为 CPU/CUDA），与 `v` 形状和设备相同，正且有限。选择一致单位体系；SI 时为 kg/m³。接口通过可微的 `Z=v*rho` 转换，梯度包含此链式法则，不隐式换算单位。与 `impedance` 恰好给出一个。
:type density: `torch.Tensor | None`
:param source_amplitudes: **必填；仅关键字参数。** 固定实数张量 `[B,S,T]`，三个维度均非空，不能设置 `requires_grad=True`。数据须有限、可表示为 float32；会转换到模型设备和 float32。它是归一化 forcing `f`，不是每步压力增量；若波场用 Pa、长度用 m，单位为 Pa/m²。不能只传 `[T]` 或 `[B,T]`，共享波形应显式扩展炮/源维。
:type source_amplitudes: `torch.Tensor`
:param source_locations: **必填；仅关键字参数。** 固定整数张量 `[B,S,2]`；单源时也接受 `[B,2]`。建议 `torch.long`。使用输入模型轴序的物理网格下标，不能包含 PML 偏移或越界值。每个源有对应波形；同一炮内重合源的增量相加。
:type source_locations: `torch.Tensor`
:param receiver_locations: **必填；仅关键字参数。** 固定整数张量 `[B,R,2]`，`R>0`，建议 `torch.long`。炮数与源一致，使用输入模型轴序的物理网格下标。允许重复位置；接收点不可省略，不能用空张量请求无记录传播。
:type receiver_locations: `torch.Tensor`
:param accuracy: **默认 `8`；仅关键字参数。** 空间有限差分阶数，可选 `2,4,6,8`，不是误差容差。影响模板范围、计算量和 boundary buffer 最小值；拒绝布尔值及其它阶数。
:type accuracy: `int`
:param pml_freq: **默认 `25.0`；仅关键字参数。** 正且有限的 PML 设置频率，典型单位 Hz，也用于空间采样诊断。不是波形生成器，不会自动测量源信号频率；应按实际实验选择。
:type pml_freq: `float | int`
:param pml_width: **默认 `20`；仅关键字参数。** 正整数网格厚度，或按 `[x-low,x-high,z-low,z-high]` 给出四个相等正整数。V17 另支持平顶形式 `[A,A,0,A]`，其余三面宽度均为同一个正整数 A。拒绝其它零面、全零、其余面宽不等、负值、非整数或 bool。第一物理行仍参与传播，见下文。
:type pml_width: `int | list[int] | tuple[int, ...]`
:param boundary_buffer: **默认 `5`；仅关键字参数。** 非负空间网格 buffer，不是时间步数或 checkpoint 间隔。boundary 模式要求至少 `accuracy // 2 + 1`，8 阶时为 5；full 模式允许 0。各吸收侧总填充为 `pml_width + boundary_buffer + accuracy // 2`；顶面为零时只保留 `accuracy // 2` 个 halo 单元，不含顶部 PML 或 buffer。 Torch VRZ 额外要求至少 accuracy//2，full/checkpoint 均适用。
:type boundary_buffer: `int`
:param memory: **默认 `'boundary'`；仅关键字参数。** 模型反传历史策略。`"boundary"` 保存边界条带并重构，`"full"` 保存完整历史。full 可能耗尽显存；原生没有自动降级或 checkpoint；Torch 另支持 full/checkpoint（不支持 boundary），其图/检查点存储与原生历史不同。没有模型梯度需求的正演不保留模型反传历史。
:type memory: `str`
:param max_vel: **默认 `None`；仅关键字参数。** CFL/PML 速度包络，典型单位 m/s。`None` 时每次根据当前 `max(v)` 重新规划；显式值须正且有限并覆盖当前最大速度，不是裁剪上限。规划与 PML 设置不参与求导。该主速度约束不保证任意阻抗反差的空间稳定性。
:type max_vel: `float | int | None`
:param freq_taper_frac: **默认 `0.0`；仅关键字参数。** `[0,1]` 内有限比例，控制时间重采样中高频 FFT bin 的余弦 taper，数量按比例取整。重采样时即使为 0，最后一个正 rFFT bin 也会被抑制，奇数长度同样如此；不保证保留所有频率。内部重采样比为 1 时此设置不改变信号。
:type freq_taper_frac: `float | int`
:param time_pad_frac: **默认 `0.0`；仅关键字参数。** `[0,1]` 内有限比例；重采样前在尾部补零，补零数为 `int(time_pad_frac*T)` 用户样本，之后移除。不增加公开记录长度或实际传播时长；内部重采样比为 1 时不改变信号。
:type time_pad_frac: `float | int`
:param time_taper: **默认 `False`；仅关键字参数。** 是否在时间重采样中应用非周期 Hann 窗：上采样后、下采样前应用。改变信号及其反传转置；仅接受布尔值，不接受整数 0/1。内部重采样比为 1 时不改变信号。
:type time_taper: `bool`
:param illumination: **默认 `None`；仅关键字参数。** 仅保留兼容关键字，必须为 `None`。VRZ 不支持照明，显式传入收集器或其它值会引发 `NotImplementedError`；不要沿用 scalar 的照明接线。 Torch 必须为 None。
:type illumination: `None`
:param backend: **默认 `None`；仅关键字参数。** None 保留原生行为，不自动选择 Torch；`'cuda'` 显式要求 CUDA 模型；`'torch'` 使用 CPU/CUDA 张量正演和 autograd。设备取自输入，不自动移动模型；不接受 `'cpu'` 或 MPS。scalar/VRZ/VTI 的 Torch 路径须显式选择 full/checkpoint。
:type backend: `str | None`
:param execution: **默认 `'eager'`；仅关键字参数。** Torch 可选 `'eager'` 或 `'compile'`；compile 使用 PyTorch Inductor/AOTAutograd，失败不回退 eager。原生仅接受 eager。
:type execution: `str`
:param checkpoint_interval: **默认 `None`；仅关键字参数。** Torch checkpoint 分段长度，正整数内部时间步，拒绝 bool；None 表示 32。full 或原生传播必须保持 None。增大间隔会增加单段重算图，不能保证更省显存。
:type checkpoint_interval: `int | None`
:param compile_steps: **默认 `1`；仅关键字参数。** 正整数，拒绝 bool；Torch 每个执行段包含的连续内部时间步数。compile 可尝试 4，完整处理尾段；不改变 dt、nt、炮数或损失。原生只接受 1。
:type compile_steps: `int`
:returns: **`(receiver_amplitudes,)`**，单元素元组。唯一记录张量为 `[B,R,T]`，连续 float32（原生 CUDA；Torch 为 CPU/CUDA），与模型在同一设备；名义时间 `0, dt, ..., (T-1)*dt`。是 pressure-like 记录，可用 `[0]` 或 `[-1]` 取得，不包含最终波场或边界状态。
:rtype: `tuple[torch.Tensor]`
```

**Torch 路径：** 同一主入口设置 `backend="torch"` 并显式 `memory="full"` 或 `"checkpoint"`，可使用 CPU/CUDA，无需原生构建；导入仍会初始化已有可用的原生库。下方历史存储/原生 stream 限制只描述原生路径。完整选项和内存表见 [PyTorch 后端](pytorch-backend.md)。

(vrz-details)=
### 参数化、震源与梯度

使用 `impedance=Z` 时，速度和阻抗是两个独立输入；使用 `density=rho` 时，内部由 `v*rho` 得到阻抗。两种梯度的含义不同，不能不经过链式法则就互换。可训练输入用 `requires_grad=True`，其它介质值仍参与正演。

震源采用与 scalar 相同的归一化 forcing 协议，内部增量为 `-v_source² * internal_dt² * U(f)`，不是直接压力增量。不要额外乘时间步或速度缩放。记录在下采样前通过前置零并移去最后一步对齐用户时间；FFT 重采样可能出现端点振铃。

正且有限、系数可表示及 CFL 检查不保证任意强阻抗反差的稳定性；近零与强反差模型可能严重病态。

模型梯度采用固定扩展模型、CFL 与 PML 的条件。重新生成物理边缘的 replicate padding 时，当前返回梯度不包含完整扩展链的转置累积，不能当作整个重建边界过程的全导数；切换 full 不改变这一限制。

顶面为零的 `[A,A,0,A]` 使用 scalar/VRZ 零 halo 约定：外侧 M 个单元为零，用户 `z=0` 仍参与传播，顶部 PML/buffer 被移除。源/接收坐标不加偏移。二阶时有效压力释放平面位于第一物理行上方一格；内部差分阶数提高不等于获得奇镜像高阶表面。CUDA full/boundary 与 Torch full/checkpoint 仍使用固定公开源，保留速度相关源缩放链及已有的裁剪模型梯度限制。

(vrz-examples)=
### 示例

以下为已准备有效模型、采集和原生库后的调用片段；不是独立运行程序。所有省略的可选参数采用上方默认值。

```python
import starwave

receiver_amplitudes, = starwave.vrz(
    v, grid_spacing=10.0, dt=0.001,
    density=rho,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
)
```

若选择阻抗参数化，将 `density=rho` 替换为 `impedance=Z`，不要同时保留两者。更多背景见{ref}`VRZ 建模说明 <wave-vrz>`。完整独立 GPU 算例仍待补充。

(vrz-notes)=
### 注意事项

- 原生路径仅 CUDA FP32 默认 stream；每次 forward 只支持一次一阶 backward。源梯度、高阶导数、AMP、CUDA graphs、自定义 stream、公开初始/最终状态不在支持范围。
- 时间 CFL 子步不能弥补空间采样不足；空间分辨率警告是诊断，不是精度或稳定性证书。
- DataParallel 应把完整模型放入 Module，只拆分炮维采集；参见[多卡入门](inversion/dataparallel.md)。
- 本页已核对接口契约，未完成目标 GPU 数值、性能或 FWI 验收。

(vrr)=
## VRR Function

```{py:function} starwave_vrr.vrr(v: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, rx: torch.Tensor, rz: torch.Tensor, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, accuracy: int=8, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float=0.0, time_pad_frac: float=0.0, time_taper: bool=False, backend: str='cuda', execution: str='eager', checkpoint_interval: int | None=None, compile_steps: int=1) -> tuple[torch.Tensor]

这是已整合到经审阅 dev 源树的**独立源码扩展**，发行包名 `starwave-vrr`（`0.1.0.dev1`），导入名 `starwave_vrr`。它不在已发布的 V16 / StarWave 7.1.0 wheel 中，也没有 `starwave.vrr` 别名。使用前须获得包含 `extensions/vrr` 的完整授权源码；公开手册不分发私有实现。主包安装另见[安装页](installation.md)。

:param v: **必填。** 正有限速度 [nx,nz]，单位 m/s。
:type v: `torch.Tensor`
:param grid_spacing: **必填。** 正等距网格：标量或相等二元组，单位米。
:type grid_spacing: `float | int | list[float] | tuple[float, ...]`
:param dt: **必填。** 正用户采样间隔，单位秒。
:type dt: `float | int`
:param rx: **必填。** 仅关键字参数。有限带符号 x 向反射率向量场，与 v 同形、同设备、同精度；单位 1/m。
:type rx: `torch.Tensor`
:param rz: **必填。** 仅关键字参数。有限带符号 z 向反射率向量场，与 v 同形、同设备、同精度；单位 1/m。
:type rz: `torch.Tensor`
:param source_amplitudes: **必填。** 仅关键字参数。外力 [B,S,T]；共享子波须显式扩展到各炮、各源。
:type source_amplitudes: `torch.Tensor`
:param source_locations: **必填。** 仅关键字参数。整数 [B,S,2]，每炮单源也可用 [B,2]。
:type source_locations: `torch.Tensor`
:param receiver_locations: **必填。** 仅关键字参数。整数 [B,R,2]；坐标索引未填充的物理模型。
:type receiver_locations: `torch.Tensor`
:param accuracy: **默认 `8`。** 仅关键字参数。内部空间差分阶数：2、4、6 或 8。
:type accuracy: `int`
:param pml_freq: **默认 `25.0`。** 仅关键字参数。正 CPML 参考频率，单位 Hz。
:type pml_freq: `float | int`
:param pml_width: **默认 `20`。** 仅关键字参数。正标量、四面相等正宽度，或正整数 w 对应的 [w,w,0,w]。
:type pml_width: `int | list[int] | tuple[int, ...]`
:param boundary_buffer: **默认 `5`。** 仅关键字参数。空间网格数；至少 accuracy//2+1，包括 full 模式。
:type boundary_buffer: `int`
:param memory: **默认 `"boundary"`。** 仅关键字参数。CUDA：full/boundary。Torch：full/checkpoint，须显式选择。
:type memory: `str`
:param max_vel: **默认 `None`。** 仅关键字参数。省略时取当前速度最大值并 detach；显式正上界须覆盖 v。
:type max_vel: `float | int | None`
:param freq_taper_frac: **默认 `0.0`。** 仅关键字参数。FFT 重采样的频率窗比例。
:type freq_taper_frac: `float`
:param time_pad_frac: **默认 `0.0`。** 仅关键字参数。FFT 重采样的时间补零比例。
:type time_pad_frac: `float`
:param time_taper: **默认 `False`。** 仅关键字参数。可选重采样时间窗。
:type time_taper: `bool`
:param backend: **默认 `"cuda"`。** 仅关键字参数。cuda 或 torch；不自动切换后端。
:type backend: `str`
:param execution: **默认 `"eager"`。** 仅关键字参数。Torch：eager/compile；原生 CUDA 仅接受 eager。
:type execution: `str`
:param checkpoint_interval: **默认 `None`。** 仅关键字参数。仅 Torch checkpoint；None 为 32 个内部时间步，否则须为正整数。
:type checkpoint_interval: `int | None`
:param compile_steps: **默认 `1`。** 仅关键字参数。每个编译块的内部时间步数，须为正整数；原生 CUDA 仅接受 1。
:type compile_steps: `int`
:returns: 单元素 `(record,)` 元组；记录形状 `[B,R,T]`。使用 `result[-1]` 取出；不返回末态或照明。
:rtype: tuple[torch.Tensor]
```

(vrr-install)=
### 安装与构建

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

(vrr-backends)=
### 后端与梯度

| Backend | Device / dtype | Memory | Execution |
|---|---|---|---|
| `cuda` (default) | CUDA float32 | `full`, `boundary` (default) | `eager` selector; native execution |
| `torch` | CPU/CUDA float32/float64 | `full`, `checkpoint` | `eager`, `compile` |

仅选择 Torch 不会改变 boundary 默认值，须同时指定 `memory="full"` 或 `"checkpoint"`。无效组合明确报错，不静默切换。Torch compile 显式使用已安装 Inductor，编译错误直接抛出。原生 CUDA 要求 `execution="eager"`、`compile_steps=1`、`checkpoint_interval=None`。

CUDA full 保存六状态时序（两级压力及四个 CPML 记忆场）；boundary 保存条带历史和末态、逆重建内部，历史仍随时间和条带规模增长。Torch checkpoint 保存完整六状态段端点，反传重算段内正演，与原生 boundary 重建不同。

支持一次一阶反传，覆盖独立 `v`、`rx`、`rz` 和震源幅值，包括源点速度缩放及完整复制扩边转置。几何与规划输入不求导。不支持 MPS、AMP/autocast、CUDA graphs、原生 checkpoint、Torch boundary、VRR3D 或高阶 AD。`starwave_vrr.vrr_reference` 是显式选择的 CPU/CUDA float32/64 full 历史参考入口，无 backend/execution/checkpoint/compile 选择参数，也不是失败回退。

(vrr-physics)=
### 模型、震源与时间约定

```text
p_tt = v² (p_xx + p_zz) + (v v_x - 2 v² rx) p_x + (v v_z - 2 v² rz) p_z
```

二维声学反射率向量模型采用公开 `[x,z]` 数组，深度在最后一轴；采集坐标为整数 `[ix,iz]`。常见 `[z,x]` 模型须逐个转置。`rx/rz` 为带符号逆长度场，不是密度、Q 或无量纲反射系数。对光滑正阻抗 `Z=v*rho`，连续关系为 `r=0.5*gradient(log Z)`；离散差分不必满足链式/乘积法则，因此不代表与离散 VRZ 等价。API 不自动换算密度，独立向量场也不必对应物理可实现的阻抗。

若压力单位为 P，震源外力单位为 P/m²；注入量为 `-v_source² * dt_internal² * U(source)`，其中 U 为时间上采样。初态为零；公开记录对更新前对齐序列 `[0, raw[:-1]]` 下采样，对应用户时刻 `0,dt,...`。FFT 重采样可能振铃，不能保证所有子步比例的首个公开样点精确为零。

内部规划使用 `max_dt=0.6*h/(max_vel*sqrt(2))` 和 `ratio=max(1,ceil(dt/max_dt))`。该主速度上界不保证任意反射率、速度梯度或 CPML 下稳定。优化和有限差分宜固定覆盖模型的 `max_vel`，检查实际时长并做空间/时间加密。传播器不裁剪或平滑模型，不修补 loss，也不选择优化器。

(vrr-top)=
### 水平顶部边界

面顺序为 `[x-low,x-high,z-low,z-high]`。仅正整数 w 对应的 `pml_width=[w,w,0,w]` 选择水平顶面；正标量或四面相等正宽度保留全吸收。其他零面、多零面、全零或不等正吸收宽度均拒绝。

顶部移除 PML/buffer，但保留 `M=accuracy//2` 层零 halo；物理 `z=0` 仍参与更新。这是零 halo 闭合，不是奇镜像延拓：二阶零压面位于首个 ghost 行，高阶内部差分不意味着同阶表面收敛精度。不增加自由表面开关，也不支持任意边界面或地形。

(vrr-example)=
### 小型合成示例

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
### 源码与验证范围

当前审查的 V17 源码为 `27c2e14b184b94d8031a5b3a83c25b1ed3d4ac67`（`0.1.0.dev17`），包含导入时原生初始化。此前审查的 dev 基线 `904c9106bdb883325df8586b7c3412f7814a9737` 保留了在 `6b9ec41d3bef1a8a3593a63f124c6ee4139955b0` 集成的 VRR 运行时。此 dev 扩展晚于公开 7.1.0 所依据的冻结 V16 源码 `5a224dbb1e86154ff034bbedaf67c0e4cf7a1b6a`；审查的 VRR 功能输入为 `e8ccf9b0fe70edd023fc4c6d0319abae6c203c65`。dev 集成保留主包已有自由表面工作，扩展仍独立打包。Release V17 现在指向上述冻结源码；独立 VRR 扩展不会因此进入 7.1.0 wheel，PyPI 8.0.0 仍待发布。

物理参数化依据 [Whitmore 等，EAGE 2020](https://doi.org/10.3997/2214-4609.202010332)。张量顶部约定依据 [SWEEP dev](https://github.com/DeepWave-KAUST/sweep/tree/32235f796ffa5b272ca81b9016f8cc85e25ecdf1) 提交 `32235f796ffa5b272ca81b9016f8cc85e25ecdf1`；原生 VRR CUDA 步进/VJP 与边界重建为独立实现，不是上游 CUDA 移植。

经审阅源码记录了用户提供的 RTX 4060 Laptop 与 A30 数值报告，包含原生 full/boundary 和输入梯度；CPU/Inductor、主机加载、实际 GPU、多卡、长程 FWI 与性能属于不同证据类别。本次文档构建不重跑或认证这些 GPU 结果，也不将结果转移为 7.1.0 wheel 或任意硬件/物理条件的保证。一次一阶反传不代表 Born/LSRTM 或 Hessian API。

(vti)=
## VTI Function

```{py:function} starwave.vti(vp: torch.Tensor, epsilon: torch.Tensor, delta: torch.Tensor, rho: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, source_fields: str | list[str] | tuple[str, ...]=('sH', 'sV'), receiver_fields: str | list[str] | tuple[str, ...]=('vz',), accuracy: int=4, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float | int=0.0, time_pad_frac: float | int=0.0, time_taper: bool=False, backend: str | None=None, execution: str='eager', checkpoint_interval: int | None=None, compile_steps: int=1) -> tuple[torch.Tensor, ...]

传播 Duveneck 型一阶声学 VTI 系统，以 `vp.ndim` 选择二维或三维。输出按所选接收分量排列；四个模型可独立请求一次一阶梯度。固定模型仍参与全部物理计算；本接口不是完整弹性 VTI。

:param vp: **必填；位置或关键字参数。** 竖直 P 波速度，单位 m/s。float32（原生 CUDA；Torch 为 CPU/CUDA），正且有限，形状 `[nx,nz]` 或 `[nx,ny,nz]`，每维至少为 2；最后一轴始终竖直。四个模型必须形状、dtype、设备一致；不会自动推断或转换物理轴序。
:type vp: `torch.Tensor`
:param epsilon: **必填；位置或关键字参数。** 无量纲 Thomsen epsilon，float32（原生 CUDA；Torch 为 CPU/CUDA），与 `vp` 形状/设备相同；有限且严格大于 -0.5，并须满足系数和导数的数值可表示性。允许 `epsilon<delta`，不作隐式裁剪；存在增长模式的已知限制。
:type epsilon: `torch.Tensor`
:param delta: **必填；位置或关键字参数。** 无量纲 Thomsen delta，float32（原生 CUDA；Torch 为 CPU/CUDA），与 `vp` 形状/设备相同；有限且严格大于 -0.5，并须满足系数和导数的数值可表示性。没有 `delta<=epsilon` 的输入强制约束，不会隐式改模型。
:type delta: `torch.Tensor`
:param rho: **必填；位置或关键字参数。** 密度，单位 kg/m³。float32（原生 CUDA；Torch 为 CPU/CUDA），与 `vp` 形状/设备相同，正且有限；倒数、刚度及导数须可表示。无默认密度，不做隐式单位换算；若原数据单位已确认为 g/cm³，应在调用前显式乘 1000。
:type rho: `torch.Tensor`
:param grid_spacing: **必填；位置或关键字参数。** 正且有限的网格间距，单位 m。接受单值或逐轴间距 `[dx,dz]` / `[dx,dy,dz]`，允许不等间距；轴序必须对应模型。拒绝布尔值、零/负值和长度不匹配。
:type grid_spacing: `float | int | list[float] | tuple[float, ...]`
:param dt: **必填；位置或关键字参数。** 正且有限的用户输入/输出采样间隔，典型单位 s。内部可能按 CFL 采用更小的时间步，返回间隔仍为此值。不能传入 Tensor 或布尔值，不对其求导。
:type dt: `float | int`
:param source_amplitudes: **必填；仅关键字参数。** 固定 float32/float64 张量 `[B,S,T]`，单位 **Pa/s**，是保持于内部时间步内的应力变化率。三个维度须非空；须有限并可转换为 float32。不能设置 `requires_grad=True`，不接受整数/布尔/复数源。内部应用 `internal_dt*U(rate)`；不需要外部再乘 dt。
:type source_amplitudes: `torch.Tensor`
:param source_locations: **必填；仅关键字参数。** 固定整数张量 `[B,S,D]`，单源时接受 `[B,D]`；`D=vp.ndim`，建议 `torch.long`。以模型 `[x,z]` / `[x,y,z]` 轴序给出物理网格下标，不含 PML 偏移，必须在范围内。重合源累加各自增量。
:type source_locations: `torch.Tensor`
:param receiver_locations: **必填；仅关键字参数。** 固定整数张量 `[B,R,D]`，`R>0`，炮数与源一致，建议 `torch.long`。使用模型轴序；这些整数索引所选分量的交错网格，不做空间插值到网格中心。允许重复位置，不能越界。
:type receiver_locations: `torch.Tensor`
:param source_fields: **默认 `('sH', 'sV')`；仅关键字参数。** 接收相同源变化率的法向应力分量，只能选择 `sH`、`sV`；可用单个名称或非空、不重复的名称序列。默认同时注入两者，不会把振幅除以 2。不支持速度源分量或逐源分量映射。
:type source_fields: `str | list[str] | tuple[str, ...]`
:param receiver_fields: **默认 `('vz',)`；仅关键字参数。** 接收分量及返回顺序，非空且不重复。二维可选 `vx,vz,sH,sV`，三维另有 `vy`。速度单位 m/s，位于对应轴正向半网格位置；应力单位 Pa，位于网格中心。没有 `pressure` 别名，二维不能选择 `vy`。
:type receiver_fields: `str | list[str] | tuple[str, ...]`
:param accuracy: **默认 `4`；仅关键字参数。** 交错网格空间有限差分阶数，可选 `2,4,6,8`；影响模板、CFL 和 buffer 最小值，不是误差容差。拒绝布尔值及其它阶数。
:type accuracy: `int`
:param pml_freq: **默认 `25.0`；仅关键字参数。** 正且有限的 PML 设置频率，典型单位 Hz，也用于空间采样诊断。不是波形生成器，不会自动测量源信号频率；应按实际实验选择。
:type pml_freq: `float | int`
:param pml_width: **默认 `20`；仅关键字参数。** 正整数网格厚度，或按各轴低端/高端成对给出的 `2*D` 个相等正整数。V17 另支持二维 `[x,z]` 的 `[A,A,0,A]` 或三维 `[x,y,z]` 的 `[A,A,A,A,0,A]`，其余各面均为相等正整数 A。只允许顶面为零；拒绝其它零面、全零、其余面宽不等、负值、非整数或 bool。此形式选择下述投影法向牵引闭合。
:type pml_width: `int | list[int] | tuple[int, ...]`
:param boundary_buffer: **默认 `5`；仅关键字参数。** 非负空间网格 buffer，不是时间步数或 checkpoint 间隔。boundary 模式要求至少 `accuracy // 2 + 1`，8 阶时为 5；full 模式允许 0。各吸收侧总填充为 `pml_width + boundary_buffer + accuracy // 2`；顶面为零时只保留 `accuracy // 2` 个 halo 单元，不含顶部 PML 或 buffer。
:type boundary_buffer: `int`
:param memory: **默认 `'boundary'`；仅关键字参数。** 模型反传历史策略。`"boundary"` 保存边界条带并重构，`"full"` 保存完整历史。full 可能耗尽显存；原生没有自动降级或 checkpoint；Torch 另支持 full/checkpoint（不支持 boundary），其图/检查点存储与原生历史不同。没有模型梯度需求的正演不保留模型反传历史。
:type memory: `str`
:param max_vel: **默认 `None`；仅关键字参数。** CFL/PML 各向异性速度包络，单位 m/s。`None` 时动态计算；显式值须正、有限并覆盖 `max(vp*sqrt(max(1,1+2*epsilon,sqrt(1+2*delta))))`。只覆盖 `max(vp)` 不足。密度反差仍重新计算；规划还取决于维数、间距和差分系数，不参与求导。
:type max_vel: `float | int | None`
:param freq_taper_frac: **默认 `0.0`；仅关键字参数。** `[0,1]` 内有限比例，控制时间重采样中高频 FFT bin 的余弦 taper，数量按比例取整。重采样时即使为 0，最后一个正 rFFT bin 也会被抑制，奇数长度同样如此；不保证保留所有频率。内部重采样比为 1 时此设置不改变信号。
:type freq_taper_frac: `float | int`
:param time_pad_frac: **默认 `0.0`；仅关键字参数。** `[0,1]` 内有限比例；重采样前在尾部补零，补零数为 `int(time_pad_frac*T)` 用户样本，之后移除。不增加公开记录长度或实际传播时长；内部重采样比为 1 时不改变信号。
:type time_pad_frac: `float | int`
:param time_taper: **默认 `False`；仅关键字参数。** 是否在时间重采样中应用非周期 Hann 窗：上采样后、下采样前应用。改变信号及其反传转置；仅接受布尔值，不接受整数 0/1。内部重采样比为 1 时不改变信号。
:type time_taper: `bool`
:param backend: **默认 `None`；仅关键字参数。** None 保留原生行为，不自动选择 Torch；`'cuda'` 显式要求 CUDA 模型；`'torch'` 使用 CPU/CUDA 张量正演和 autograd。设备取自输入，不自动移动模型；不接受 `'cpu'` 或 MPS。scalar/VRZ/VTI 的 Torch 路径须显式选择 full/checkpoint。
:type backend: `str | None`
:param execution: **默认 `'eager'`；仅关键字参数。** Torch 可选 `'eager'` 或 `'compile'`；compile 使用 PyTorch Inductor/AOTAutograd，失败不回退 eager。原生仅接受 eager。
:type execution: `str`
:param checkpoint_interval: **默认 `None`；仅关键字参数。** Torch checkpoint 分段长度，正整数内部时间步，拒绝 bool；None 表示 32。full 或原生传播必须保持 None。增大间隔会增加单段重算图，不能保证更省显存。
:type checkpoint_interval: `int | None`
:param compile_steps: **默认 `1`；仅关键字参数。** 正整数，拒绝 bool；Torch 每个执行段包含的连续内部时间步数。compile 可尝试 4，完整处理尾段；不改变 dt、nt、炮数或损失。原生只接受 1。
:type compile_steps: `int`
:returns: **`(records_0, ..., records_n)`**，每个 `receiver_fields` 条目对应一个 `[B,R,T]` 连续 float32（原生 CUDA；Torch 为 CPU/CUDA） 张量，顺序严格对应输入分量顺序。默认只有 `(vz_records,)`，为竖直质点速度 m/s，不是压力；`sH/sV` 为应力 Pa。张量与模型在同一设备，名义用户时间 `0, dt, ..., (T-1)*dt`，不包含最终状态。
:rtype: `tuple[torch.Tensor, ...]`
```

**Torch 路径：** 同一主入口设置 `backend="torch"` 并显式 `memory="full"` 或 `"checkpoint"`，可使用 CPU/CUDA，无需原生构建；导入仍会初始化已有可用的原生库。下方历史存储/原生 stream 限制只描述原生路径。完整选项和内存表见 [PyTorch 后端](pytorch-backend.md)。

(vti-details)=
### 分量、时间与梯度

正的应力变化率对每个所选应力分量增加正应力；内部应用 `internal_dt*U(rate)`。没有 scalar 的 `-vp² dt²` 因子，也没有模型相关源缩放。旧的每步应力增量不能直接作为公开变化率输入。

应力记录在下采样前通过前置零、移去末项对齐时间；速度记录则将前后半时间步样本平均到整数时间（初始前半步取零）。两者输出都标在名义用户时间上，但物理量与空间位置不同。重采样可能出现端点振铃。

`vp`、`epsilon`、`delta`、`rho` 各自仅在 `requires_grad=True` 时请求梯度；固定字段仍参与正演和伴随。没有按模型数值自动冻结的规则，`nn.Parameter` 默认启用梯度；固定字段可注册为 buffer。

`epsilon<delta` 不被自动修复；较小 dt 不能保证消除负刚度增长或任意 PML 不稳定。速度包络的覆盖只是一项必要的输入条件。

模型梯度采用固定扩展模型、CFL 与 PML 的条件。重新生成物理边缘的 replicate padding 时，当前返回梯度不包含完整扩展链的转置累积，不能当作整个重建边界过程的全导数；切换 full 不改变这一限制。

顶面 `pml_width` 为零时在用户 `z=0` 处施加 `sV=0`。消去法向应变后，表面本构系数为 `diag(2*rho*vp**2*(epsilon-delta),0)`；法向应力奇镜像与面速度偶镜像构成精确负转置导数对。这是局部伪声学闭合，不是完整弹性 VTI，也不承诺表面达到内部高阶精度。拒绝 `z=0` 处含 `sV` 的源，包括默认双分量源；表面 `sV` 接收值为零。`sH` 源和速度接收点保留原生交错位置。原生 full/boundary 保留顶部条带，Torch full/checkpoint 使用相同闭合。投影材料链参与求导，但固定扩展/裁剪模型 VJP 限制和 `epsilon<delta` 不稳定性仍存在。

(vti-examples)=
### 示例

以下为已准备有效模型、采集和原生库后的调用片段；不是独立运行程序。所有省略的可选参数采用上方默认值。

```python
import starwave

stress_h, velocity_z = starwave.vti(
    vp, epsilon, delta, rho,
    grid_spacing=[10.0, 10.0], dt=0.001,
    source_amplitudes=stress_rate,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    receiver_fields=("sH", "vz"),
)
```

本例为二维输入；三维需同时使用 `[nx,ny,nz]` 四模型、三维坐标和相应间距。省略 `receiver_fields` 时只返回 `vz`，应使用单变量元组解包。VTI 没有 `illumination` 参数。完整独立 GPU 算例仍待补充，见{ref}`VTI 建模说明 <wave-vti>`。

(vti-notes)=
### 注意事项

- 原生路径仅 CUDA FP32 默认 stream；每次 forward 只支持一次一阶 backward。源梯度、高阶导数、AMP、CUDA graphs、自定义 stream、公开初始/最终状态不在支持范围。
- 时间 CFL 子步不能弥补空间采样不足；空间分辨率警告是诊断，不是精度或稳定性证书。
- DataParallel 应把完整模型放入 Module，只拆分炮维采集；参见[多卡入门](inversion/dataparallel.md)。
- 本页已核对接口契约，未完成目标 GPU 数值、性能或 FWI 验收。

(elastic)=
## Elastic Function

```{py:function} starwave.elastic(lamb: torch.Tensor, mu: torch.Tensor, buoyancy: torch.Tensor, grid_spacing: Union[float, Sequence[float]], dt: float, source_amplitudes_z: Optional[torch.Tensor]=None, source_amplitudes_y: Optional[torch.Tensor]=None, source_amplitudes_x: Optional[torch.Tensor]=None, source_amplitudes_p: Optional[torch.Tensor]=None, source_locations_z: Optional[torch.Tensor]=None, source_locations_y: Optional[torch.Tensor]=None, source_locations_x: Optional[torch.Tensor]=None, source_locations_p: Optional[torch.Tensor]=None, receiver_locations_z: Optional[torch.Tensor]=None, receiver_locations_y: Optional[torch.Tensor]=None, receiver_locations_x: Optional[torch.Tensor]=None, receiver_locations_p: Optional[torch.Tensor]=None, accuracy: int=4, pml_width: Union[int, Sequence[int]]=20, pml_freq: Optional[float]=None, max_vel: Optional[float]=None, survey_pad: Optional[Union[int, Sequence[Optional[int]]]]=None, vz_0: Optional[torch.Tensor]=None, vy_0: Optional[torch.Tensor]=None, vx_0: Optional[torch.Tensor]=None, sigmazz_0: Optional[torch.Tensor]=None, sigmayz_0: Optional[torch.Tensor]=None, sigmaxz_0: Optional[torch.Tensor]=None, sigmayy_0: Optional[torch.Tensor]=None, sigmaxy_0: Optional[torch.Tensor]=None, sigmaxx_0: Optional[torch.Tensor]=None, m_vzz_0: Optional[torch.Tensor]=None, m_vzy_0: Optional[torch.Tensor]=None, m_vzx_0: Optional[torch.Tensor]=None, m_vyz_0: Optional[torch.Tensor]=None, m_vxz_0: Optional[torch.Tensor]=None, m_vyy_0: Optional[torch.Tensor]=None, m_vyx_0: Optional[torch.Tensor]=None, m_vxy_0: Optional[torch.Tensor]=None, m_vxx_0: Optional[torch.Tensor]=None, m_sigmazzz_0: Optional[torch.Tensor]=None, m_sigmayzy_0: Optional[torch.Tensor]=None, m_sigmaxzx_0: Optional[torch.Tensor]=None, m_sigmayzz_0: Optional[torch.Tensor]=None, m_sigmaxzz_0: Optional[torch.Tensor]=None, m_sigmayyy_0: Optional[torch.Tensor]=None, m_sigmaxyy_0: Optional[torch.Tensor]=None, m_sigmaxyx_0: Optional[torch.Tensor]=None, m_sigmaxxx_0: Optional[torch.Tensor]=None, origin: Optional[Sequence[int]]=None, nt: Optional[int]=None, model_gradient_sampling_interval: int=1, freq_taper_frac: float=0.0, time_pad_frac: float=0.0, time_taper: bool=False, forward_callback: Optional[common.Callback]=None, callback_frequency: int=1, python_backend: Union[Literal['eager', 'jit', 'compile'], bool]=False, storage_mode: Literal['device', 'cpu', 'disk', 'none']='device', storage_path: str='.', storage_compression: bool=False, *, memory: Literal['full', 'boundary', 'checkpoint']='full', backend: str | None=None, execution: str='eager', checkpoint_interval: int | None=None, compile_steps: int=1) -> Tuple[torch.Tensor, ...]

二维/三维各向同性弹性波传播，使用 Deepwave 0.0.27 派生后端。输入是原始 Lamé 参数和浮力参数，支持多炮、材料/源/初始状态的一阶自动微分。memory 与新增 backend/execution/checkpoint_interval/compile_steps 仅关键字；其余可按位置或关键字传入，只有前五项必填。Torch 路径的限制与原生/Python 兼容路径分开，见 [PyTorch 后端](pytorch-backend.md)。

:param lamb: **必填。** 第一 Lamé 参数，典型单位 Pa。空间形状 `[Ny,Nx]`（二维）或 `[Nz,Ny,Nx]`（三维）；可选前导模型批次维为 1 或 B。CPU/float32（原生 CUDA；Torch 为 CPU/CUDA） 或 float64，值须有限。与 mu、buoyancy 的空间形状、dtype 和 device 相同；不会识别或自动转换 vp。
:type lamb: `torch.Tensor`
:param mu: **必填。** 第二 Lamé 参数（剪切模量），典型单位 Pa；模型批次、轴序、dtype 与 device 规则同 lamb。可训练输入保留 autograd 链。
:type mu: `torch.Tensor`
:param buoyancy: **必填。** 浮力参数，即逆密度，典型单位 m³/kg；模型形状规则同 lamb。不是密度 rho；由 vp/vs/rho 建模时先显式调用下方转换函数。
:type buoyancy: `torch.Tensor`
:param grid_spacing: **必填。** 网格间距，典型单位 m；正数或长度 D 的正数序列，按模型空间轴顺序排列，可为各向不同间距。
:type grid_spacing: `Union[float, Sequence[float]]`
:param dt: **必填。** 用户源和记录的采样间隔，典型单位 s。内部可按 CFL 细分；速度与应力使用交错时间网格，详见下方时钟说明。
:type dt: `float`
:param source_amplitudes_z: **默认 `None`。** 仅三维。张量 `[B,S,T]`，沿 z 轴的力密度，典型单位 N/m³。须与对应 source_locations_z 配对，dtype/device 与材料一致；可求源梯度。各已提供分量的炮数与时间长度须相容。
:type source_amplitudes_z: `Optional[torch.Tensor]`
:param source_amplitudes_y: **默认 `None`。** 二维/三维。张量 `[B,S,T]`，沿 y 轴的力密度，典型单位 N/m³。须与对应 source_locations_y 配对，dtype/device 与材料一致；可求源梯度。各已提供分量的炮数与时间长度须相容。
:type source_amplitudes_y: `Optional[torch.Tensor]`
:param source_amplitudes_x: **默认 `None`。** 二维/三维。张量 `[B,S,T]`，沿 x 轴的力密度，典型单位 N/m³。须与对应 source_locations_x 配对，dtype/device 与材料一致；可求源梯度。各已提供分量的炮数与时间长度须相容。
:type source_amplitudes_x: `Optional[torch.Tensor]`
:param source_amplitudes_p: **默认 `None`。** 二维/三维。张量 `[B,S,T]`，压力变化率，典型单位 Pa/s，正值表示压缩。须与对应 source_locations_p 配对，dtype/device 与材料一致；可求源梯度。各已提供分量的炮数与时间长度须相容。
:type source_amplitudes_p: `Optional[torch.Tensor]`
:param source_locations_z: **默认 `None`。** 仅三维。整数网格坐标 `[B,S,D]`，建议 torch.long，与 source_amplitudes_z 配对，使用输入模型轴序，不加 PML 偏移。同一炮内同一分量的源位置须唯一；`starwave.common.IGNORE_LOCATION` 可屏蔽位置。
:type source_locations_z: `Optional[torch.Tensor]`
:param source_locations_y: **默认 `None`。** 二维/三维。整数网格坐标 `[B,S,D]`，建议 torch.long，与 source_amplitudes_y 配对，使用输入模型轴序，不加 PML 偏移。同一炮内同一分量的源位置须唯一；`starwave.common.IGNORE_LOCATION` 可屏蔽位置。
:type source_locations_y: `Optional[torch.Tensor]`
:param source_locations_x: **默认 `None`。** 二维/三维。整数网格坐标 `[B,S,D]`，建议 torch.long，与 source_amplitudes_x 配对，使用输入模型轴序，不加 PML 偏移。同一炮内同一分量的源位置须唯一；`starwave.common.IGNORE_LOCATION` 可屏蔽位置。
:type source_locations_x: `Optional[torch.Tensor]`
:param source_locations_p: **默认 `None`。** 二维/三维。整数网格坐标 `[B,S,D]`，建议 torch.long，与 source_amplitudes_p 配对，使用输入模型轴序，不加 PML 偏移。同一炮内同一分量的源位置须唯一；`starwave.common.IGNORE_LOCATION` 可屏蔽位置。
:type source_locations_p: `Optional[torch.Tensor]`
:param receiver_locations_z: **默认 `None`。** 仅三维。整数网格坐标 `[B,R,D]`，建议 torch.long；按模型轴序，不加 PML 偏移。请求z 速度记录；省略时仍保留该返回槽位的空张量。同一炮内同一分量的有效接收位置须唯一；可用 `starwave.common.IGNORE_LOCATION` 屏蔽位置。
:type receiver_locations_z: `Optional[torch.Tensor]`
:param receiver_locations_y: **默认 `None`。** 二维/三维。整数网格坐标 `[B,R,D]`，建议 torch.long；按模型轴序，不加 PML 偏移。请求y 速度记录；省略时仍保留该返回槽位的空张量。同一炮内同一分量的有效接收位置须唯一；可用 `starwave.common.IGNORE_LOCATION` 屏蔽位置。
:type receiver_locations_y: `Optional[torch.Tensor]`
:param receiver_locations_x: **默认 `None`。** 二维/三维。整数网格坐标 `[B,R,D]`，建议 torch.long；按模型轴序，不加 PML 偏移。请求x 速度记录；省略时仍保留该返回槽位的空张量。同一炮内同一分量的有效接收位置须唯一；可用 `starwave.common.IGNORE_LOCATION` 屏蔽位置。
:type receiver_locations_x: `Optional[torch.Tensor]`
:param receiver_locations_p: **默认 `None`。** 二维/三维。整数网格坐标 `[B,R,D]`，建议 torch.long；按模型轴序，不加 PML 偏移。请求压力记录；省略时仍保留该返回槽位的空张量。同一炮内同一分量的有效接收位置须唯一；可用 `starwave.common.IGNORE_LOCATION` 屏蔽位置。
:type receiver_locations_p: `Optional[torch.Tensor]`
:param accuracy: **默认 `4`。** 空间有限差分阶数，可选 `2,4,6,8`；full 和 boundary 的二维/三维均支持。
:type accuracy: `int`
:param pml_width: **默认 `20`。** 正整数网格厚度，或按各空间轴低端/高端排列的 `2*D` 个正面宽，正面宽可不同。V17 支持二维 `[y,x]` 的顶面为零形式 `[0,A,B,C]` 和三维 `[z,y,x]` 的 `[0,A,B,C,D,E]`，其余面必须全正；拒绝其它零面和全零。它在第一模型轴应用改进真空准备：第 0 行为固定真空，第 1 行是首个固体应力行。顶面为零时拒绝 survey_pad、origin 和所有已给初始场，见下文。
:type pml_width: `Union[int, Sequence[int]]`
:param pml_freq: **默认 `None`。** PML 设置频率，典型单位 Hz。None 使用 25 Hz 并发出提示；建议显式给出源的主频。
:type pml_freq: `Optional[float]`
:param max_vel: **默认 `None`。** 用于 CFL/PML 的最大速度，典型单位 m/s；None 从材料重建的 P/S 波速取最大值。显式值应覆盖模型波速，不是速度裁剪或可训练参数。
:type max_vel: `Optional[float]`
:param survey_pad: **默认 `None`。** 围绕本次调用所有炮的源/接收点范围提取一个公共子域：无初始状态时 None 保留全模型；整数指定各面余量；长度 2D 序列按各轴低/高侧给出余量，其中 None 延伸到该侧模型边缘。余量以格点计，不包括另加的 PML；裁剪可能改变结果，必须保留波传播所需区域。提供初始状态时，None 可改由状态形状与 origin 确定子域；survey_pad 与 origin 不能同时非 None。
:type survey_pad: `Optional[Union[int, Sequence[Optional[int]]]]`
:param vz_0: **默认 `None`。** 仅三维；初始速度 vz，典型单位 m/s，内部时间 −h/2（h 为内部时间步）。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type vz_0: `Optional[torch.Tensor]`
:param vy_0: **默认 `None`。** 二维/三维；初始速度 vy，典型单位 m/s，内部时间 −h/2（h 为内部时间步）。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type vy_0: `Optional[torch.Tensor]`
:param vx_0: **默认 `None`。** 二维/三维；初始速度 vx，典型单位 m/s，内部时间 −h/2（h 为内部时间步）。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type vx_0: `Optional[torch.Tensor]`
:param sigmazz_0: **默认 `None`。** 仅三维；初始应力 sigmazz，典型单位 Pa，内部时间 0。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type sigmazz_0: `Optional[torch.Tensor]`
:param sigmayz_0: **默认 `None`。** 仅三维；初始应力 sigmayz，典型单位 Pa，内部时间 0。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type sigmayz_0: `Optional[torch.Tensor]`
:param sigmaxz_0: **默认 `None`。** 仅三维；初始应力 sigmaxz，典型单位 Pa，内部时间 0。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type sigmaxz_0: `Optional[torch.Tensor]`
:param sigmayy_0: **默认 `None`。** 二维/三维；初始应力 sigmayy，典型单位 Pa，内部时间 0。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type sigmayy_0: `Optional[torch.Tensor]`
:param sigmaxy_0: **默认 `None`。** 二维/三维；初始应力 sigmaxy，典型单位 Pa，内部时间 0。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type sigmaxy_0: `Optional[torch.Tensor]`
:param sigmaxx_0: **默认 `None`。** 二维/三维；初始应力 sigmaxx，典型单位 Pa，内部时间 0。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type sigmaxx_0: `Optional[torch.Tensor]`
:param m_vzz_0: **默认 `None`。** 仅三维；初始 PML 记忆变量 m_vzz。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type m_vzz_0: `Optional[torch.Tensor]`
:param m_vzy_0: **默认 `None`。** 仅三维；初始 PML 记忆变量 m_vzy。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type m_vzy_0: `Optional[torch.Tensor]`
:param m_vzx_0: **默认 `None`。** 仅三维；初始 PML 记忆变量 m_vzx。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type m_vzx_0: `Optional[torch.Tensor]`
:param m_vyz_0: **默认 `None`。** 仅三维；初始 PML 记忆变量 m_vyz。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type m_vyz_0: `Optional[torch.Tensor]`
:param m_vxz_0: **默认 `None`。** 仅三维；初始 PML 记忆变量 m_vxz。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type m_vxz_0: `Optional[torch.Tensor]`
:param m_vyy_0: **默认 `None`。** 二维/三维；初始 PML 记忆变量 m_vyy。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type m_vyy_0: `Optional[torch.Tensor]`
:param m_vyx_0: **默认 `None`。** 二维/三维；初始 PML 记忆变量 m_vyx。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type m_vyx_0: `Optional[torch.Tensor]`
:param m_vxy_0: **默认 `None`。** 二维/三维；初始 PML 记忆变量 m_vxy。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type m_vxy_0: `Optional[torch.Tensor]`
:param m_vxx_0: **默认 `None`。** 二维/三维；初始 PML 记忆变量 m_vxx。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type m_vxx_0: `Optional[torch.Tensor]`
:param m_sigmazzz_0: **默认 `None`。** 仅三维；初始 PML 记忆变量 m_sigmazzz。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type m_sigmazzz_0: `Optional[torch.Tensor]`
:param m_sigmayzy_0: **默认 `None`。** 仅三维；初始 PML 记忆变量 m_sigmayzy。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type m_sigmayzy_0: `Optional[torch.Tensor]`
:param m_sigmaxzx_0: **默认 `None`。** 仅三维；初始 PML 记忆变量 m_sigmaxzx。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type m_sigmaxzx_0: `Optional[torch.Tensor]`
:param m_sigmayzz_0: **默认 `None`。** 仅三维；初始 PML 记忆变量 m_sigmayzz。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type m_sigmayzz_0: `Optional[torch.Tensor]`
:param m_sigmaxzz_0: **默认 `None`。** 仅三维；初始 PML 记忆变量 m_sigmaxzz。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type m_sigmaxzz_0: `Optional[torch.Tensor]`
:param m_sigmayyy_0: **默认 `None`。** 二维/三维；初始 PML 记忆变量 m_sigmayyy。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type m_sigmayyy_0: `Optional[torch.Tensor]`
:param m_sigmaxyy_0: **默认 `None`。** 二维/三维；初始 PML 记忆变量 m_sigmaxyy。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type m_sigmaxyy_0: `Optional[torch.Tensor]`
:param m_sigmaxyx_0: **默认 `None`。** 二维/三维；初始 PML 记忆变量 m_sigmaxyx。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type m_sigmaxyx_0: `Optional[torch.Tensor]`
:param m_sigmaxxx_0: **默认 `None`。** 二维/三维；初始 PML 记忆变量 m_sigmaxxx。形状 `[B,*传播域]`，包含 PML、不包含差分 halo；dtype/device 与材料一致，可求梯度。None 初始化为零；交错网格边缘掩码仍适用。
:type m_sigmaxxx_0: `Optional[torch.Tensor]`
:param origin: **默认 `None`。** 提供的初始波场对应的模型原点，长度 D 的整数坐标，按模型轴序。与状态形状及 PML 一起确定继续传播的区域；未给出时按上游规则推断。
:type origin: `Optional[Sequence[int]]`
:param nt: **默认 `None`。** 没有任何源波形时必须提供的用户时间采样数。有源波形时通常由其最后一维决定时长；若也给出 nt，必须等于波形长度。无源时仍需坐标或初始状态提供足够的维数/炮数信息。
:type nt: `Optional[int]`
:param model_gradient_sampling_interval: **默认 `1`。** 模型梯度的时间采样间隔，整数 ≥1；内部累计步幅为 CFL 子步比 q 乘本值。默认 1 最直接；大于 1 是采样梯度，不能承诺与逐步累计相同。当前原生路径只处理完整采样组，详见存储与采样说明。 Torch 仅接受 1，并对所有内部步求导。
:type model_gradient_sampling_interval: `int`
:param freq_taper_frac: **默认 `0.0`。** CFL 引起 FFT 重采样时，对频谱高频端施加余弦 taper 的比例。影响源和记录，不改变用户输出长度。
:type freq_taper_frac: `float`
:param time_pad_frac: **默认 `0.0`。** FFT 重采样前追加零的长度比例，之后移除；作用于源和记录，不增加公开记录的时长。
:type time_pad_frac: `float`
:param time_taper: **默认 `False`。** 是否在重采样时对源和记录施加时间 Hann 窗。
:type time_taper: `bool`
:param forward_callback: **默认 `None`。** 接收 `starwave.common.CallbackState` 的观测函数。原生回调按后端外循环组触发；state.dt 是内部 h，state.step 是外循环索引。boundary 只允许只读快照。 Torch 仅接受 None。
:type forward_callback: `Optional[common.Callback]`
:param callback_frequency: **默认 `1`。** 正整数；两次回调之间的后端外循环组数。原生路径一组跨越 q × model_gradient_sampling_interval 个内部步，不是固定的用户记录采样间隔。
:type callback_frequency: `int`
:param python_backend: **默认 `False`。** backend=None 时，False 选择 C/CUDA；`"eager"`、`"jit"`、`"compile"` 显式选择 PyTorch 路径。True 在已核验构建声明 OpenMP 时选择 compile，否则 jit；能力依赖 PyTorch/工具链。原 python_backend 兼容路径仅接受 full，且要求 device 存储、无压缩。 新增 backend="torch" 时仅接受 False 或 eager；编译改用 execution="compile"。
:type python_backend: `Union[Literal['eager', 'jit', 'compile'], bool]`
:param storage_mode: **默认 `'device'`。** 中间数据存储，可选 `"device"`、`"cpu"`、`"disk"`、`"none"`。只在 full 原生路径支持 offload/disk/none；none 不提供完整材料梯度，详见下方说明。 Torch 仅接受 device。
:type storage_mode: `Literal['device', 'cpu', 'disk', 'none']`
:param storage_path: **默认 `'.'`。** disk 模式临时历史文件的目录。保留计算图时也可能保留这些文件；反传仍需使用的文件不能提前删除。
:type storage_path: `str`
:param storage_compression: **默认 `False`。** full 原生路径的有损中间数据压缩，可能改变模型梯度。boundary 和 Python 后端不接受 True。 Torch 仅接受 False。
:type storage_compression: `bool`
:param memory: **默认 `'full'`。** StarWave 的仅关键字扩展。`"full"` 保存体积历史；`"boundary"` 使用 CUDA 边界存储与重构。二者支持二维/三维及 2/4/6/8 阶；限制见下方，未支持组合不会自动退回 full。 Torch 另支持 checkpoint，仅通过 backend="torch"；不支持 Torch boundary。
:type memory: `Literal['full', 'boundary', 'checkpoint']`
:param backend: **默认 `None`；仅关键字参数。** None 保留原生行为，不自动选择 Torch；`'cuda'` 显式要求 CUDA 模型；`'torch'` 使用 CPU/CUDA 张量正演和 autograd。设备取自输入，不自动移动模型；不接受 `'cpu'` 或 MPS。scalar/VRZ/VTI 的 Torch 路径须显式选择 full/checkpoint。
:type backend: `str | None`
:param execution: **默认 `'eager'`；仅关键字参数。** Torch 可选 `'eager'` 或 `'compile'`；compile 使用 PyTorch Inductor/AOTAutograd，失败不回退 eager。原生仅接受 eager。
:type execution: `str`
:param checkpoint_interval: **默认 `None`；仅关键字参数。** Torch checkpoint 分段长度，正整数内部时间步，拒绝 bool；None 表示 32。full 或原生传播必须保持 None。增大间隔会增加单段重算图，不能保证更省显存。
:type checkpoint_interval: `int | None`
:param compile_steps: **默认 `1`；仅关键字参数。** 正整数，拒绝 bool；Torch 每个执行段包含的连续内部时间步数。compile 可尝试 4，完整处理尾段；不改变 dt、nt、炮数或损失。原生只接受 1。
:type compile_steps: `int`
:returns: 完整最终波场/PML 状态与记录元组：二维 16 个张量，三维 31 个张量。记录形状 `[B,R,T]`，未请求分量为空张量；下方给出精确顺序。
:rtype: `Tuple[torch.Tensor, ...]`
```

**Torch 路径：** 同一主入口设置 `backend="torch"` 并显式 `memory="full"` 或 `"checkpoint"`，可使用 CPU/CUDA，无需原生构建；导入仍会初始化已有可用的原生库。下方历史存储/原生 stream 限制只描述原生路径。完整选项和内存表见 [PyTorch 后端](pytorch-backend.md)。

(elastic-axes)=
### 模型轴序与交错网格

二维的 API 轴名为 `[y,x]`，三维为 `[z,y,x]`，名称对应输入张量的空间轴，不能脱离数组约定固定解释为物理垂直方向。若二维数组已经是 `[depth,horizontal]`，直接输入时 y 就是深度，垂直力使用 source_amplitudes_y；若显式转置为 `[horizontal,depth]`，深度就对应 x，源/接收分量、坐标、grid_spacing、PML 各面和 origin 均须一致变换。两种材料转换函数不做转置。

材料可共享或按炮给出，源和记录分别为 `[B,S,T]` 与 `[B,R,T]`；坐标最后一维严格遵循同一空间轴序。交错网格中分量位于不同半格位置，不能把不同分量当成同一点的同一种量。速度源/接收点在自身分量轴上不能取模型最后一个网格下标。参见 [Deepwave 弹性交错网格](https://ausargeo.com/deepwave/elastic.html)。

源 y/x（以及三维 z）是力密度，源 p 是压力变化率；接收 y/x/z 是速度，接收 p 是法向应力负平均。名义上压力对应 `t*dt`、速度对应 `(t-0.5)*dt`。内部 CFL 细分时使用内部半步与 FFT 重采样，不额外提供半步重新对齐操作；不要把 scalar 的 forcing 或记录时钟直接套用在这里。

顶面为零时，第一模型轴为深度轴。改进真空准备将第 0 行 lambda、mu 和 buoyancy 替换为固定零，不修改调用者张量；这些材料在该行的梯度为零。第 1 行为首个固体应力行，法向速度界面的 buoyancy 采用真空/固体平均。交错界面的法向体力源仍允许，真空行压力源被拒绝。这不是把首个固体行所有牵引同点钳零，也不支持任意地形。顶面为零要求第 0 行以下 buoyancy 为正，并拒绝 `survey_pad`、`origin` 及所有已给初始场。原生 CUDA full/boundary、原生 CPU full 和 Torch full/checkpoint 各自的限制仍适用。全正 PML 沿用原路径，包括调用者自行给出的真空材料；源/接收下标不额外平移。

(elastic-returns)=
### 完整返回顺序

二维返回 16 个张量，最后三个固定为 p/y/x 记录：

```text
(vy, vx, sigmayy, sigmaxy, sigmaxx,
 m_vyy, m_vyx, m_vxy, m_vxx,
 m_sigmayyy, m_sigmaxyy, m_sigmaxyx, m_sigmaxxx,
 receiver_amplitudes_p, receiver_amplitudes_y, receiver_amplitudes_x)
```

三维返回 31 个张量，最后四个固定为 p/z/y/x 记录：

```text
(vz, vy, vx, sigmazz, sigmayz, sigmaxz, sigmayy, sigmaxy, sigmaxx,
 m_vzz, m_vzy, m_vzx, m_vyz, m_vxz, m_vyy, m_vyx, m_vxy, m_vxx,
 m_sigmazzz, m_sigmayzy, m_sigmaxzx, m_sigmayzz, m_sigmaxzz,
 m_sigmayyy, m_sigmaxyy, m_sigmaxyx, m_sigmaxxx,
 receiver_amplitudes_p, receiver_amplitudes_z,
 receiver_amplitudes_y, receiver_amplitudes_x)
```

最终状态形状为 `[B,*传播域（含 PML）]`，不含差分 halo；取决于 survey_pad、origin 与初始状态指定的实际域。可按对应的 `_0` 参数继续传播。每个已请求记录为 `[B,R,T]`；省略的记录仍占形状 `(0,)` 的空张量槽位，所以 `outputs[-1]` 始终是 x 分量，不能通用地理解成“我请求的记录”。张量 dtype/device 与材料一致。

(elastic-memory)=
### 存储与梯度采样

边界重建的时间顺序、材料梯度和存储公式见{ref}`波场反传重建 <reconstruction>`。

- `memory="full"` 是默认值，二维/三维支持 CPU/CUDA、float32/float64。原生 `storage_mode="device"` 在当前设备保存历史；CUDA 下 `"cpu"` 使用主机存储，CPU 模型下会归一为 device；`"disk"` 使用 storage_path。`storage_compression=True` 是有损压缩。
- `storage_mode="none"` 会关闭传播历史对材料梯度的贡献，不能用于完整材料梯度反演；力源的材料缩放链仍可能产生部分梯度。存在可训练材料时会提示，源/初始状态梯度是另一条路径。
- `memory="boundary"` 是 StarWave 扩展：仅 CUDA，要求 `python_backend=False`、`storage_mode="device"`、`storage_compression=False`。裁剪后每个物理维度须大于 accuracy；请求 buoyancy 梯度时，有效交错 buoyancy 不能为零（顶面为零准备中固定的顶部真空切向槽位除外）。无 CPU/offload/压缩或自动 full 回退。内存/速度收益随域形状、PML 与采样而变。
- 默认 `model_gradient_sampling_interval=1`，CFL 比 q>1 时材料梯度步幅仍为 q，并非每个内部步都累计。若设为大于 1 的整数，当前原生 full/boundary 仅执行完整采样组：非整组尾部不传播；正时长不足一组会报错（有回调时可能在反传发生）。保持默认值最直接；需要采样时，让间隔不超过用户 nt 且整除 nt，并核对采样梯度的适用性。
- 普通反传释放 autograd 保存的张量历史；保留计算图可延长历史寿命。disk 临时文件随相应输出/loss 图释放；日志只保留 detached 数值，仍需反传的文件不要提前删除。

Torch 支持 full/checkpoint，要求 `model_gradient_sampling_interval=1`、`forward_callback=None`、`storage_mode="device"`、`storage_compression=False`，`python_backend` 仅 False 或 eager。它对每个内部步求导；原生按 CFL step_ratio × sampling_interval 抽样材料梯度，所以 native/Torch 精确材料梯度对照还需 step_ratio=1。

(elastic-examples)=
### 最小调用示例

已安装 [StarWave 4.0.0](installation.md) 且逻辑 CUDA 设备 0 可用时，以下构造 `[depth,horizontal]` 模型并沿深度施力。它演示 vp/vs/rho → 显式转换 → elastic → 一阶 Vs 梯度；此处未执行 GPU 实验，也不作为正确性或收敛验收。

```python
import torch
import starwave

# The input array axes are [depth, horizontal], mapped to elastic [y, x].
device = torch.device("cuda:0")
starwave.prepare_elastic([0])  # Optional explicit device check.
vp = torch.full((48, 64), 2200.0, device=device)
vs = torch.full_like(vp, 1100.0, requires_grad=True)
rho = torch.full_like(vp, 2000.0)
lamb, mu, buoyancy = starwave.common.vpvsrho_to_lambmubuoyancy(vp, vs, rho)

dt, nt, freq = 0.001, 100, 15.0
t = torch.arange(nt, device=device) * dt
phase = torch.pi * freq * (t - 0.04)
wavelet = (1.0 - 2.0 * phase.square()) * torch.exp(-phase.square())
force_y = wavelet.reshape(1, 1, nt)  # [shot, source, time], force density
source_y = torch.tensor([[[8, 32]]], device=device, dtype=torch.long)
receiver_y = torch.tensor([[[8, 20], [8, 32], [8, 44]]],
                          device=device, dtype=torch.long)
outputs = starwave.elastic(
    lamb, mu, buoyancy, grid_spacing=(10.0, 10.0), dt=dt,
    source_amplitudes_y=force_y, source_locations_y=source_y,
    receiver_locations_y=receiver_y, pml_freq=freq, accuracy=4,
    memory="boundary",
)
record_y = outputs[-2]  # [1, 3, 100]; the final slot is x, even when absent
loss = record_y.square().mean()
loss.backward()
assert vs.grad is not None
```

原生 CPU 应先构建匹配库，将 device 改为 `torch.device("cpu")` 并使用 `memory="full"`；`starwave.prepare_elastic()` 为可选。三维按 `[z,y,x]` 构造模型；第一轴为深度时垂直力用 z，坐标末维为 3，p/z/y/x 记录为 `outputs[-4:]`。

(elastic-notes)=
### 使用边界

- 原生传播提供一阶 AD，不承诺高阶导数。转换函数保留 PyTorch 链，但不增加传播器的高阶导数能力。原始材料没有物理符号修复；负开方、病态材料或分辨率不足仍可能导致无效结果。
- 没有公开 backward_callback、free_surface、boundary_buffer、source_fields 或 receiver_fields 参数；使用实际分量关键字。这里的 nt/storage 选项也不能直接加到 scalar、VRZ、VTI。
- V17 导入会在 DataParallel 前初始化可用弹性库；{py:func}`starwave.prepare_elastic` 仍可用于可选的显式设备检查。将完整材料放入 Module，只沿炮维拆分采集；初始化成功不代表多卡数值验收。
- 本次文档维护核对发布 wheel 的接口及 Python 3.10 语法，不执行 GPU。已有用户提供的 RTX 4060 源码测试报告与公开 wheel 的 CPU/主机测试具有各自范围，不据此宣称 A30、多卡、长程 FWI 或目标 GPU 性能全部通过。

(elastic-conversions)=
### 材料参数转换

```{py:function} starwave.common.vpvsrho_to_lambmubuoyancy(vp: torch.Tensor, vs: torch.Tensor, rho: torch.Tensor, eps: float=1e-15) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

:param vp: **必填。** P 波速度，典型单位 m/s。
:type vp: `torch.Tensor`
:param vs: **必填。** S 波速度，典型单位 m/s。
:type vs: `torch.Tensor`
:param rho: **必填。** 密度，典型单位 kg/m³。
:type rho: `torch.Tensor`
:param eps: **默认 `1e-15`。** 逆密度/密度分母的正则项；保留该值，不做自动单位缩放。
:type eps: `float`
:returns: `(lamb, mu, buoyancy)`；各表达式独立遵循 PyTorch 广播与类型提升，保留空间轴序和 autograd 链。传播前应提供形状匹配的完整模型。
:rtype: `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
```

```{py:function} starwave.common.lambmubuoyancy_to_vpvsrho(lamb: torch.Tensor, mu: torch.Tensor, buoyancy: torch.Tensor, eps: float=1e-15) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

:param lamb: **必填。** 第一 Lamé 参数，典型单位 Pa。
:type lamb: `torch.Tensor`
:param mu: **必填。** 剪切模量，典型单位 Pa。
:type mu: `torch.Tensor`
:param buoyancy: **必填。** 逆密度，典型单位 m³/kg。
:type buoyancy: `torch.Tensor`
:param eps: **默认 `1e-15`。** 逆密度/密度分母的正则项；保留该值，不做自动单位缩放。
:type eps: `float`
:returns: `(vp, vs, rho)`；各表达式独立遵循 PyTorch 广播与类型提升，保留空间轴序和 autograd 链。传播前应提供形状匹配的完整模型。
:rtype: `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
```

正向代数为 `lamb=(vp**2-2*vs**2)*rho`、`mu=vs**2*rho`、`buoyancy=rho/(rho**2+eps)`。逆向先取 `vs=sqrt(mu*buoyancy)`，再取 `vp=sqrt(lamb*buoyancy+2*vs**2)` 与 `rho=buoyancy/(buoyancy**2+eps)`。eps 使往返通常近似而非精确；负速度符号经平方丢失。支持 PyTorch 广播但不会校验/修复物理符号、裁剪开方输入、detach 或修改输入；传播器仍要求一致的空间形状。

(visco-gsls)=
## GSLS Function

本节对应 V17 源码 `0.1.0.dev17`，含已合并的顶面为零压力释放边界。二维 `starwave.visco_gsls` 是当前唯一黏声入口；单机制 SLS 通过显式 `mode="sls_compat", n_mechanisms=1` 使用。独立 `visco_sls`、`prepare_visco_sls` 和 `visco_sls_native_status` 已移除。已发布的 [PyPI 7.1.0](https://pypi.org/project/starwave/7.1.0/) 包含较早的 V16 GSLS API，不含 V17 边界/初始化变化；历史 `7.0.0` 不含 GSLS。Release V17 已发布，PyPI `8.0.0` 仍待发布。详见[文档状态](status.md)及[安装](installation.md)。

支持固定空间变密度、float32/float64、Vp/Q/source/provider 参数一阶梯度。原生 CPU/CUDA 支持 full/checkpoint；显式 Torch 支持 full/checkpoint 与 eager/compile。所有主传播入口均新增执行选项，原生默认行为保持；详见 [PyTorch 后端](pytorch-backend.md)。

```{py:function} starwave.visco_gsls(vp: torch.Tensor, q: torch.Tensor, rho: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, float], dt: float | int, *, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, f_ref: float | int, accuracy: int=4, pml_width: int | list[int] | tuple[int, int]=20, memory: str='full', max_vel: float | int | None=None, backend: str='cuda', mode: str='hao1', frequency_band: tuple[float, float] | list[float] | None=None, n_mechanisms: int | None=None, fit_tolerance: float | int | None=0.05, strict: bool=False, material_model: Callable | None=None, checkpoint_interval: int | None=None, execution: str='eager', compile_steps: int=1) -> tuple[torch.Tensor]

:param vp: **必填；位置或关键字。** `[nz,nx]` float32/float64 正有限张量，两个维度均至少 2；单位 m/s，严格为 `f_ref` 处相速度，可求导。
:type vp: `torch.Tensor`
:param q: **必填；位置或关键字。** 与 Vp 同形状/dtype/device，正有限或 +inf。Hao1 为名义 Q0，要求 Q0>S，不能称为精确 Qref；band_fit 为频带拟合目标；sls_compat 为参考频率精确模量 Q。可求导。
:type q: `torch.Tensor`
:param rho: **必填；位置或关键字。** 与模型匹配，正有限 kg/m³，倒数须可表示。支持空间变化，但 `requires_grad=True` 拒绝。
:type rho: `torch.Tensor`
:param grid_spacing: **必填；位置或关键字。** 正有限标量或 `(dz,dx)` tuple/list，单位 m。
:type grid_spacing: `float | int | list[float] | tuple[float, float]`
:param dt: **必填；位置或关键字。** 正有限秒；必须满足含未松弛速度和密度对比的保守 CFL 界，不自动重采样或子步进。
:type dt: `float | int`
:param source_amplitudes: **必填；仅关键字。** 与模型匹配的有限 `[B,S,T]` 张量，各维非空；单位 Pa/m²，二阶 forcing，可求导。
:type source_amplitudes: `torch.Tensor`
:param source_locations: **必填；仅关键字。** int32/int64 `[B,S,2]`，原始未扩边模型内 `[z,x]` 整数坐标；重复震源相加，无单源省略维度格式。顶面为零时要求源 z>=1。
:type source_locations: `torch.Tensor`
:param receiver_locations: **必填；仅关键字。** int32/int64 `[B,R,2]`，R≥1；原模型坐标；重复接收点重复记录，反传 cotangent 累加。顶面为零时，表面压力接收值为零。
:type receiver_locations: `torch.Tensor`
:param f_ref: **必填；仅关键字。** 正有限 Hz；Vp 的精确相速度参考频率，也参与固定 CPML setup。
:type f_ref: `float | int`
:param accuracy: **默认 `4`；仅关键字。** 空间 staggered FD 阶数 2/4/6/8，不是时间阶数。
:type accuracy: `int`
:param pml_width: **默认 `20`；仅关键字。** 正整数或两个正数的 `(z,x)` 对称宽度；也可用 `[0,bottom,left,right]` 选择平顶，其余各面须为正且可不同。拒绝单值/二元组中含零、全正四面列表、其它零面、bool、负值及非整数。replicate 扩边梯度仍完整累加。
:type pml_width: `int | list[int] | tuple[int, int]`
:param memory: **默认 `'full'`；仅关键字。** native CPU/CUDA 支持 `full` 或 `checkpoint`；Torch 也支持 `full` / `checkpoint`；不支持 boundary。
:type memory: `str`
:param max_vel: **默认 `None`；仅关键字。** 正有限 m/s 未松弛速度上界。任何可训练物性输出且启用非零 PML 时须显式固定；不能只覆盖 Vp。过小拒绝。
:type max_vel: `float | int | None`
:param backend: **默认 `'cuda'`；仅关键字。** 显式 `cuda`、`native_cpu` 或 `torch`；原生后端须匹配设备与已构建库，无回退或运行时编译。
:type backend: `str`
:param mode: **默认 `'hao1'`；仅关键字。** 默认 Hao 一阶 inverse-Q 模型，或显式 `band_fit`、`sls_compat`。不含论文二阶耦合模型。
:type mode: `str`
:param frequency_band: **默认 `None`；仅关键字。** 内置 Hao1 必须显式提供 `(scale,200*scale)` Hz 设计频带；band_fit 必须提供正递增拟合频带；不能把震源工作子频带当作 Hao1 任意带宽重新拟合。
:type frequency_band: `tuple[float, float] | list[float] | None`
:param n_mechanisms: **默认 `None`；仅关键字。** Hao1 默认 5，显式也只能 5；band_fit 默认 3，允许整数 2–6；sls_compat 须显式 1。
:type n_mechanisms: `int | None`
:param fit_tolerance: **默认 `0.05`；仅关键字。** 正相对 inverse-Q 容差；超过时默认 warning；None 跳过检查。采样诊断不构成全连续频率误差证明。
:type fit_tolerance: `float | int | None`
:param strict: **默认 `False`；仅关键字。** 超过拟合容差时是否由 warning 改为报错。
:type strict: `bool`
:param material_model: **默认 `None`；仅关键字。** 高级 callable `(vp_padded,q_padded,dt,f_ref)->(c_rel2,strength,A,B)`，接管内置物性选项，详见下文。
:type material_model: `Callable | None`
:param checkpoint_interval: **默认 `None`；仅关键字。** 仅 checkpoint 下可显式给正整数分段长度；原生 None 用形状/所需历史估计内存平衡；Torch None 固定为 32 个内部步，均不按 GPU 型号调参。full 下提供此参数会报错。
:type checkpoint_interval: `int | None`
:param execution: **默认 `'eager'`；仅关键字参数。** Torch 可选 `'eager'` 或 `'compile'`；compile 使用 PyTorch Inductor/AOTAutograd，失败不回退 eager。原生仅接受 eager。
:type execution: `str`
:param compile_steps: **默认 `1`；仅关键字参数。** 正整数，拒绝 bool；Torch 每个执行块包含的连续内部时间步数。compile 可尝试 4，完整处理尾块；不改变 dt、nt、炮数或损失。原生只接受 1。
:type compile_steps: `int`
:returns: 单元素元组 `(pressure_records,)`，形状 `[B,R,T]`、单位 Pa，与模型 dtype/device 一致。
:rtype: `tuple[torch.Tensor]`
```

<!-- api-doc:visco_gsls:signature:start -->
```text
starwave.visco_gsls(vp, q, rho, grid_spacing, dt, *, source_amplitudes,
           source_locations, receiver_locations, f_ref, accuracy=4,
           pml_width=20, memory="full", max_vel=None, backend="cuda",
           mode="hao1", frequency_band=None, n_mechanisms=None,
           fit_tolerance=0.05, strict=False, material_model=None,
           checkpoint_interval=None, execution="eager", compile_steps=1)
```
<!-- api-doc:visco_gsls:signature:end -->

前五项为位置或关键字参数；其余仅关键字。`source_amplitudes`、`source_locations`、`receiver_locations`、`f_ref` 必填。V17 四项顶面为零形式可用 list 或 tuple，须满足下方限制。下表覆盖完整参数与默认值。

<!-- api-doc:visco_gsls:parameters:start -->
| Parameter / 参数 | Calling kind | Default when omitted | Type, shape and units | Purpose and behavior | Allowed values, rejections and limits |
|---|---|---|---|---|---|
| `vp` | `positional-or-keyword` | `required` | Tensor [nz,nx]; m/s | Reference phase speed / 参考相速度 | `[nz,nx]` float32/float64 正有限张量，两个维度均至少 2；单位 m/s，严格为 `f_ref` 处相速度，可求导。 |
| `q` | `positional-or-keyword` | `required` | Tensor [nz,nx]; dimensionless | Mode-specific Q / 按模式定义的 Q | 与 Vp 同形状/dtype/device，正有限或 +inf。Hao1 为名义 Q0，要求 Q0>S，不能称为精确 Qref；band_fit 为频带拟合目标；sls_compat 为参考频率精确模量 Q。可求导。 |
| `rho` | `positional-or-keyword` | `required` | Tensor [nz,nx]; kg/m³ | Fixed density / 固定密度 | 与模型匹配，正有限 kg/m³，倒数须可表示。支持空间变化，但 `requires_grad=True` 拒绝。 |
| `grid_spacing` | `positional-or-keyword` | `required` | real or pair; m | Grid spacing / 网格间距 | 正有限标量或 `(dz,dx)` tuple/list，单位 m。 |
| `dt` | `positional-or-keyword` | `required` | real; s | Fixed time step / 固定时间步 | 正有限秒；必须满足含未松弛速度和密度对比的保守 CFL 界，不自动重采样或子步进。 |
| `source_amplitudes` | `keyword-only` | `required` | Tensor [B,S,T]; Pa/m² | Second-order forcing / 二阶 forcing | 与模型匹配的有限 `[B,S,T]` 张量，各维非空；单位 Pa/m²，二阶 forcing，可求导。 |
| `source_locations` | `keyword-only` | `required` | int32/int64 Tensor [B,S,2]; cells | Unpadded source indices / 原域源坐标 | int32/int64 `[B,S,2]`，原始未扩边模型内 `[z,x]` 整数坐标；重复震源相加，无单源省略维度格式。顶面为零时要求源 z>=1。 |
| `receiver_locations` | `keyword-only` | `required` | int32/int64 Tensor [B,R,2]; cells | Unpadded receiver indices / 原域接收坐标 | int32/int64 `[B,R,2]`，R≥1；原模型坐标；重复接收点重复记录，反传 cotangent 累加。顶面为零时，表面压力接收值为零。 |
| `f_ref` | `keyword-only` | `required` | real; Hz | Phase reference / 相速度参考频率 | 正有限 Hz；Vp 的精确相速度参考频率，也参与固定 CPML setup。 |
| `accuracy` | `keyword-only` | `4` | integer; dimensionless | Staggered spatial order / 空间阶数 | 空间 staggered FD 阶数 2/4/6/8，不是时间阶数。 |
| `pml_width` | `keyword-only` | `20` | integer, pair or top-zero face list; cells | Absorbing PML or flat top / 吸收 PML 或平顶 | 正整数或两个正数的 `(z,x)` 对称宽度；也可用 `[0,bottom,left,right]` 选择平顶，其余各面须为正且可不同。拒绝单值/二元组中含零、全正四面列表、其它零面、bool、负值及非整数。replicate 扩边梯度仍完整累加。 |
| `memory` | `keyword-only` | `'full'` | str | Native history strategy / 原生历史策略 | native CPU/CUDA 支持 `full` 或 `checkpoint`；Torch 也支持 `full` / `checkpoint`；不支持 boundary。 |
| `max_vel` | `keyword-only` | `None` | None or real; m/s | Unrelaxed envelope / 未松弛速度包络 | 正有限 m/s 未松弛速度上界。任何可训练物性输出且启用非零 PML 时须显式固定；不能只覆盖 Vp。过小拒绝。 |
| `backend` | `keyword-only` | `'cuda'` | str | Explicit implementation / 显式实现 | 显式 `cuda`、`native_cpu` 或 `torch`；原生后端须匹配设备与已构建库，无回退或运行时编译。 |
| `mode` | `keyword-only` | `'hao1'` | str | Material mapping / 物性映射 | 默认 Hao 一阶 inverse-Q 模型，或显式 `band_fit`、`sls_compat`。不含论文二阶耦合模型。 |
| `frequency_band` | `keyword-only` | `None` | None or positive pair; Hz | Design/fitting band / 设计或拟合频带 | 内置 Hao1 必须显式提供 `(scale,200*scale)` Hz 设计频带；band_fit 必须提供正递增拟合频带；不能把震源工作子频带当作 Hao1 任意带宽重新拟合。 |
| `n_mechanisms` | `keyword-only` | `None` | None or integer | Mechanism count / 机制数 | Hao1 默认 5，显式也只能 5；band_fit 默认 3，允许整数 2–6；sls_compat 须显式 1。 |
| `fit_tolerance` | `keyword-only` | `0.05` | None or positive real | Relative inverse-Q tolerance / 相对逆 Q 容差 | 正相对 inverse-Q 容差；超过时默认 warning；None 跳过检查。采样诊断不构成全连续频率误差证明。 |
| `strict` | `keyword-only` | `False` | bool | Fit warning/error policy / 拟合警告或报错 | 超过拟合容差时是否由 warning 改为报错。 |
| `material_model` | `keyword-only` | `None` | None or callable | Material provider / 物性提供函数 | 高级 callable `(vp_padded,q_padded,dt,f_ref)->(c_rel2,strength,A,B)`，接管内置物性选项，详见下文。 |
| `checkpoint_interval` | `keyword-only` | `None` | None or positive integer; time steps | Replay segment length / 重放分段长度 | 仅 checkpoint 下可显式给正整数分段长度；原生 None 用形状/所需历史估计内存平衡；Torch None 固定为 32 个内部步，均不按 GPU 型号调参。full 下提供此参数会报错。 |
| `execution` | `keyword-only` | `'eager'` | str | Torch execution | Torch eager/compile；原生仅 eager。 |
| `compile_steps` | `keyword-only` | `1` | positive integer | Steps per execution block | Torch 正整数（非 bool），尾块完整执行；原生仅 1。 |
<!-- api-doc:visco_gsls:parameters:end -->

(visco-gsls-modes)=
### 模式、Q 与参考速度

| mode | q 的物理含义 | 机制数与频带 |
|---|---|---|
| `hao1`（默认） | Hao & Greenhalgh 的名义 Q0，不是精确 Q(f_ref) | `None` 解析为 5；显式也必须 5；必须显式提供 `(scale,200*scale)` Hz 设计频带 |
| `band_fit` | 近似恒 Q 的频带拟合目标，不是精确 Q(f_ref) | `None` 解析为 3；支持整数 2–6；必须给正有限递增 `(fmin,fmax)` Hz |
| `sls_compat` | f_ref 处精确复体积模量 Q | 必须显式 `n_mechanisms=1`；无需 frequency_band，不使用频带拟合选项 |

所有内置模式的 `vp` 都是 f_ref 处精确相速度，不能直接以论文 v0 替代。正频率复模量采用 `exp(i*omega*t)` 约定；模量 Q 不与波数衰减定义混用。Q=+inf 是精确无耗散极限，相应 Q 灵敏度为零。

Hao1 使用论文 Table 4 五组权重及 Eq. (45) 的统一频率缩放。“一阶”指 inverse-Q 展开阶数，并非一个机制。设 `x_l=2*pi*f_ref*t_l`、`S=sum(k_l*x_l**2/(1+x_l**2))`，被动性要求 Q0>S；不合法值报错而不截断。震源工作频带可为设计频带的子集，例如 [1,50] Hz；不能把 (1,50) 当作 Hao1 设计频带。该实现不包含论文二阶耦合模型。

Hao1 和 band_fit 默认在稠密采样频率上检查相对 inverse-Q 误差，超过 0.05 会 warning；`strict=True` 改为报错，`fit_tolerance=None` 跳过诊断。容差不改变已定义的 Hao1 物性，也不保证任意 Q 的拟合精度。band_fit 活动集/秩变化处不可微，其他区域分段可微。sls_compat 不执行频带误差检查；仍校验 strict 为 bool。

从旧 SLS 迁移须同时指定 mode 与机制数，不能只替换函数名而保留默认 Hao1。其准备函数现按 backend 在前、device 在后的 GSLS 签名调用。

(visco-gsls-physics)=
### 源、坐标、采样与输入域

模型为 `[nz,nx]` 即 `[z,x]`，两维至少 2；三个模型及源的 dtype/device 必须一致。支持 CPU/CUDA、float32/64 strided 张量和非连续视图，连续化保留 autograd 链；不隐式转换模型/源的 dtype 或设备。rho 必须固定、正有限且倒数可表示。只有 Q 可含 +inf。

几何必须完整提供 `[B,S,2]` 和 `[B,R,2]`，按未扩边原域 `[z,x]` 整数网格坐标索引，不接受省略源维度的简写或域外点。坐标会复制到模型设备并转换为 int64。重合源相加，重复接收点重复记录，其反传余切量累加。B、S、R、T 均至少 1。

源为二阶 forcing，单位 Pa/m²；压力更新含 `-dt**2*c_rel2*forcing`。首次源更新乘 1/2，满足零初始压力与压力时间导数。没有隐藏网格体积因子；Pa/s 压力率源不能不经转换直接替代。

记录是更新前 `p[n]`，对应 `t=n*dt`，共 T 点。第一个样点严格为零，最后一个源样点只影响未返回的 `p[T]`，因此该样点对记录的 VJP 为零。T=1 时记录及模型/源梯度均为零。迁移观测与旧波形时须分别核对符号、幅度单位和时间原点。

(visco-gsls-gradients)=
### 梯度、扩边与 PML

- Vp、Q、源及 provider 闭包参数可分别或同时训练；只承诺一阶传播梯度。密度、几何、dt、间距、FD 与 CPML 设置固定。
- replicate 扩边形成 `[nz+top+bottom,nx+left+right]` 域；正 `(z,x)` 二元组使各轴对面宽度对称，顶面为零则不扩展顶部 PML。完整转置将扩边敏感度累加回原始边缘与角点，不以 crop-backward 替代。源处 c_rel2 的模型依赖也保留在梯度链中。
- 顶面为零的 `[0,bottom,left,right]` 施加 `p(z=0)=0`：压力采用奇延拓，垂直面场采用偶延拓，散度为梯度的精确负转置。松弛与 CPML 状态使用同一配对。源必须 z>=1，表面压力接收值为零；固定 rho 下所有材料模式仍可用。拒绝其它零面/无 PML 请求。CPML 阻尼、alpha 和速度包络仍为固定数值设置，不引入耗散边界重构。
- 开启梯度、PML 非零且 Vp/Q 或任意物性输出可训练时（含 provider 闭包参数），须给显式固定 max_vel，覆盖 `sqrt(c_rel2*(1+strength.sum(0)))`，不能仅覆盖 Vp。训练迭代及配对有限差分共用同一包络。
- 仅正演或仅源求导时可用 max_vel=None，自动上界与 autograd 分离；显式上界过小会报错。
- 每次 forward 拥有独立状态/历史。loss、归一化、mask、批次与优化器由调用者管理。不隐式截断参数、缩炮、修改 dt 或替换 NaN；非法系数、非有限输入/接收余切量直接报错。

(visco-gsls-stability)=
### 时间步检查

令 D 为所选交错差分权重绝对值之和，v_bound 覆盖未松弛速度，则保守检查为：

```text
dt <= 0.8 / (v_bound * sqrt(max(rho)/min(rho))
             * D * sqrt(1/dz**2 + 1/dx**2))
```

超限直接报错并给出允许的最大 dt；调用者重新选择 dt 并生成对应采样的源，不隐式重采样或子步进。dt² 和逆间距须能以模型 dtype 表示为有限正 normal 数。此界不是任意强反差、极端 Q、异质 CPML 或长时运行的稳定性证明；还须检查实际实验有限性和边界反射。accuracy=2/4/6/8 是空间阶数，压力时间离散为二阶。

(visco-gsls-memory)=
### full 与 checkpoint 内存

原生 native_cpu/CUDA 与 Torch 均支持 full 和 checkpoint。不支持 boundary。full 下 `checkpoint_interval` 必须为 None；checkpoint 显式间隔为正整数时间步。原生默认 None 依据形状与所需历史估算；Torch 默认为 32 个内部步。均不按 GPU 型号自动调速。

原生 full 按所请求的 primitive 梯度选择历史：c_rel2 保存 force；strength/B 保存空间驱动；A 保存全部 M 个旧物性记忆（例如可训练松弛时间）。仅源求导或 no_grad 观测不保存这些体积历史。固定 Q 的 Hao1 Vp 梯度存 T*V 项，联合 Vp/Q 存 2*T*V，通用 provider 全 primitive 梯度存 (M+2)*T*V；五个物理记忆仍实际演化。

以下公式仅描述原生历史，不能用于 Torch autograd 峰值估算。令 N 为扩边网格点数，V=B*N，Fx=B*nz_padded*(nx_padded+1)，Fz=B*(nz_padded+1)*nx_padded，M 为机制数。完整 checkpoint 重启状态含 p、前一 p、全部物性记忆、节点与面 CPML，共 S=(4+M)*V+Fx+Fz 项。分段长度 K 的 tape/replay 历史约为 `ceil(T/K)*S + K*H*V` 项；H 为请求的 scalar 历史数，旧物性记忆贡献 M。默认 `K=ceil(sqrt(S*T/(H*V)))`（需要历史时），限制于 [1,T]。这些是张量项数，乘 dtype 字节数后也不等于完整峰值显存：还包括输入、输出、图、每炮梯度、优化器及工作区。

原生 checkpoint 恢复完整状态并正向重放每段，再按反序应用离散伴随；不是逆衰减。首次半权重、全局源时钟和尾段均保留，伴随状态跨段连续。需要 checkpoint 重放历史时，每个时间步额外重算一次，用计算换历史内存。没有边界逆重建或压缩。示例和内存公式不构成 GPU 性能测量。

Torch checkpoint 保存完整段起始状态（含 PML/物性记忆），由非重入 checkpoint 正向重算并由 autograd 反传，不使用逆衰减或上述原生伴随历史。短记录/小模型不保证降低峰值显存。

(visco-gsls-example)=
### 最小调用与完整 CPU 示例

以下最小调用假定已准备匹配 CUDA 库、固定模型、源与几何均在同一 CUDA 设备，且 dt 合法。frequency_band 虽默认 None，但默认 Hao1 语义要求显式提供。

<!-- api-doc:visco_gsls:minimal:start -->
```python
records, = starwave.visco_gsls(vp, q, rho, 10.0, 0.0005,
    source_amplitudes=source, source_locations=src, receiver_locations=rec,
    f_ref=18.0, frequency_band=(1.0, 200.0))
```
<!-- api-doc:visco_gsls:minimal:end -->

下面完整接线使用 CPU Torch full，不需原生库，全部可选参数显式提供。1000 为示例 Pa/m² 波形标定；能量 loss 只检查求导接线，不表示 FWI 收敛。Q0=40 的 Hao1 可能触发默认 5% 误差 warning；保留并诊断它，不以放宽容差宣称物性更准确。

```python
import torch
import starwave

# V17 source; explicit CPU Torch reference, no native build required.
dtype = torch.float64
nz, nx, nt = 12, 16, 80
vp = torch.full((nz, nx), 1800.0, dtype=dtype, requires_grad=True)
q = torch.full((nz, nx), 40.0, dtype=dtype, requires_grad=True)
depth = torch.linspace(0.0, 1.0, nz, dtype=dtype)
rho = (1800.0 + 200.0 * depth[:, None]).expand(nz, nx).contiguous()
dt = 0.0005
t = torch.arange(nt, dtype=dtype) * dt
a = torch.pi * 18.0 * (t - 0.02)
source = (1000.0 * (1.0 - 2.0 * a.square()) * torch.exp(-a.square()))
source = source.reshape(1, 1, nt).requires_grad_()
src = torch.tensor([[[3, 8]]], dtype=torch.int64)
rec = torch.tensor([[[3, 4], [3, 8], [3, 12]]], dtype=torch.int64)

records, = starwave.visco_gsls(
    vp, q, rho, (10.0, 10.0), dt,
    source_amplitudes=source, source_locations=src,
    receiver_locations=rec, f_ref=18.0,
    accuracy=4, pml_width=(4, 4), memory="full",
    max_vel=4000.0, backend="torch", mode="hao1",
    frequency_band=(1.0, 200.0), n_mechanisms=5,
    fit_tolerance=0.05, strict=False,
    material_model=None, checkpoint_interval=None,
    execution="eager", compile_steps=1,
)
assert records.shape == (1, 3, nt)
assert torch.isfinite(records).all()
assert torch.count_nonzero(records[..., 0]) == 0
records.square().mean().backward()
for grad in (vp.grad, q.grad, source.grad):
    assert grad is not None and torch.isfinite(grad).all()
assert torch.count_nonzero(source.grad[..., -1]) == 0
```

完整原生 checkpoint 调用（先构建匹配 CUDA 库，并将模型/源/几何移到同一 CUDA 设备；V17 导入会初始化库，也可对 CPU 张量显式选择 native_cpu）：

<!-- api-doc:visco_gsls:full:start -->
```python
records, = starwave.visco_gsls(vp, q, rho, (10.0, 12.0), 0.0005,
    source_amplitudes=source, source_locations=src, receiver_locations=rec,
    f_ref=18.0, accuracy=4, pml_width=(8, 10), memory="checkpoint",
    max_vel=4000.0, backend="cuda", mode="hao1",
    frequency_band=(1.0, 200.0), n_mechanisms=5, fit_tolerance=0.05,
    strict=False, material_model=None, checkpoint_interval=32,
    execution="eager", compile_steps=1)
```
<!-- api-doc:visco_gsls:full:end -->

示例 max_vel 仅用于此配置；实际模型必须重新核对未松弛速度与 CFL。Torch 例可保持 backend="torch"，改为 memory="checkpoint" 并设置 checkpoint_interval=32；execution="compile" 与 compile_steps=4 另行控制编译。

(visco-gsls-coefficients)=
### 物性与频谱辅助函数

辅助函数导入自 `starwave.visco_gsls_coefficients`，不使用独立 starwave_gsls 命名空间。下列为签名清单（非执行语句）：

```text
from starwave.visco_gsls_coefficients import (
    coefficients, hao1_weighting, hao1_relaxation_times,
    coefficients_from_relaxation, band_relaxation_times,
    quality_spectrum, normalized_modulus, fit_diagnostics,
)

coefficients(vp, q, dt, f_ref, *, mode="hao1", frequency_band=None,
             n_mechanisms=None, fit_tolerance=0.05, strict=False)
hao1_weighting(frequency_band=(1., 200.), *, dtype=torch.float64, device=None)
hao1_relaxation_times(frequency_band=(1., 200.), *, dtype=torch.float64, device=None)
coefficients_from_relaxation(vp, strength, relaxation_times, dt, f_ref)
band_relaxation_times(frequency_band, n_mechanisms=3, *,
                      dtype=torch.float64, device=None)
quality_spectrum(strength, relaxation_times, frequencies)
normalized_modulus(strength, relaxation_times, frequencies)
fit_diagnostics(q, strength, relaxation_times, frequency_band, *,
                sample_count=1001, chunk_size=1024,
                target_convention="band_fit_target")
```


- coefficients 返回 `(c_rel2,strength,A,B)`，首项 `[z,x]`、单位 m²/s²，其余 `[M,z,x]`、无量纲，保留 Vp/Q 完整链。该 helper 可接受标量 Q；传播入口仍要求 Q 与模型同形状。
- hao1_weighting 返回 `[5]` 秒时间与无量纲权重 `(times,k)`；hao1_relaxation_times 只返回时间。helper 的默认设计频带 (1,200) 不代表传播器自动采用该频带。
- coefficients_from_relaxation 接受非负强度 `[M,z,x]` 和正有限时间 `[M]` 或 `[M,z,x]`，保留 Vp、强度、时间梯度，并校准参考相速度。band_relaxation_times 仅适用于 band_fit，不能与 Hao1 权重混配。
- quality_spectrum 返回 `[frequency,*model_shape]` 实际模量 Q，零衰减为 +inf；normalized_modulus 返回同形状复数 F=M/M_R。频率是 Hz，时间是秒。
- fit_diagnostics 返回 detached 字典，包括 max_relative_inverse_q_error、max_relative_q_error、minimum_strength、all_nonnegative、worst_frequency_hz、worst_model_flat_index、sample_count、frequency_band、target_convention。sample_count≥2，chunk_size≥1；包含频带两端的对数网格逐模型单元检查，不是连续全频率误差证明。Hao1 应指定 hao1_nominal_Q0，可只诊断设计频带内的工作子带。

下例接在完整 CPU 示例之后，检查 [1,50] Hz 工作子带：

```python
from starwave.visco_gsls_coefficients import (
    coefficients, hao1_relaxation_times, quality_spectrum, fit_diagnostics,
)
c_rel2, strength, A, B = coefficients(
    vp, q, dt, 18.0, mode="hao1", frequency_band=(1.0, 200.0),
)
times = hao1_relaxation_times((1.0, 200.0), dtype=vp.dtype, device=vp.device)
frequencies = torch.logspace(0, torch.log10(torch.tensor(50.0)).item(),
                            101, dtype=vp.dtype, device=vp.device)
actual_q = quality_spectrum(strength, times, frequencies)
report = fit_diagnostics(q, strength, times, (1.0, 50.0),
                        target_convention="hao1_nominal_Q0")
print(actual_q.shape, report["max_relative_inverse_q_error"])
```

(visco-gsls-provider)=
### 高级 material_model 协议

`provider(vp_padded,q_padded,dt,f_ref)` 在每次 forward 完成 replicate 扩边后调用一次。它接管内置 mode、frequency_band、n_mechanisms、fit_tolerance、strict；这些参数不自动传给 provider 或诊断它。原始输入合法性、系数和 CFL 检查仍生效。频带、额外参数由闭包持有，provider 自行负责 Q 意义、相速度归一化与实际频谱准确性。

返回四个同 dtype/device 的 strided Tensor：c_rel2 `[zp,xp]`，strength/A/B `[M,zp,xp]`，M≥1。全部有限，c_rel2>0，strength≥0，B≥0，−1<A≤1，且 B≈strength*(1−A)，一致性容差为 32 倍 dtype eps。它是被动梯形 GSLS 递推的物性接口，不能通过任意系数替换波动方程。

原生反向返回全部四个物性输出的 VJP，再经 Torch 链式传到 provider 参数。避免 detach、NumPy、对可训练值使用 item 或 torch.tensor 重新包装。闭包中的原域空间张量不会自动扩边，必须自行按完全相同域扩展并保留梯度。非零 PML 下所有可训练物性输出仍要求固定显式 max_vel。

以下接在 CPU 示例后定义每机制可训练权重和时间；修改后属于实验物性，不再是固定发表的 Hao1 谱。优化器必须保持 Q0>S，并重新诊断实际谱。外部 provider 变更本身无需重编译；修改包内指纹跟踪文件须重建并重启。

```python
from starwave.visco_gsls_coefficients import (
    coefficients_from_relaxation, hao1_weighting,
)

base_times, base_k = hao1_weighting(
    (1.0, 200.0), dtype=vp.dtype, device=vp.device,
)
log_weight = torch.nn.Parameter(torch.zeros_like(base_k))
log_time = torch.nn.Parameter(torch.zeros_like(base_times))

def provider(vp_padded, q_padded, dt, f_ref):
    times = base_times * log_time.exp()
    k = base_k * log_weight.exp()
    x = 2 * torch.pi * f_ref * times
    S = (k * x.square() / (1 + x.square())).sum()
    if not bool(torch.all(q_padded > S)):
        raise ValueError("Learned material requires Q0 > S")
    invq = q_padded.reciprocal()
    strength = k[:, None, None] * invq[None] / (1 - S * invq)[None]
    return coefficients_from_relaxation(vp_padded, strength, times, dt, f_ref)

# Pass material_model=provider and a covering fixed max_vel to visco_gsls.
# Add log_weight and log_time to the caller's optimizer when learning them.
```

(visco-gsls-runtime)=
### 原生准备、构建与重启

```{py:function} starwave.prepare_visco_gsls(backend: str='native_cpu', device: str | torch.device | None=None) -> dict

可选地检查显式设备并取得已有 GSLS 库。V17 导入已加载和校验可用库；后续调用复用就绪句柄，不重新计算已载入文件的哈希。不编译或传播。

:param backend: **默认 `'native_cpu'`；位置或关键字。** 仅 'native_cpu' 或 'cuda'；不接受 'torch'。
:type backend: `str`
:param device: **默认 `None`；位置或关键字。** CUDA 必须为有效显式逻辑设备，如 'cuda:0' 或 torch.device('cuda:0')；不能只写 'cuda'。native_cpu 忽略此项。
:type device: `str | torch.device | None`
:returns: 状态字典；成功仅表示加载与兼容核验通过。
:rtype: `dict`
```

```{py:function} starwave.visco_gsls_native_status(backend: str='native_cpu') -> dict

只读查询库与 metadata 是否存在及是否载入；不载库、不编译。

:param backend: **默认 `'native_cpu'`；位置或关键字。** 仅 'native_cpu' 或 'cuda'；不接受 'torch'。
:type backend: `str`
:returns: 状态字典，含 backend、library_path、library_exists、metadata_exists、library_loaded、cuda_available、gpu_numerically_verified、supported_memory、gradient_order。gpu_numerically_verified 恒为 False，不代表 GPU 验收。
:rtype: `dict`
```

注意：prepare/status 默认 native_cpu，传播默认 cuda；prepare 两项都不是 keyword-only。建议显式写关键字防止与旧 SLS 的参数顺序混淆。

```python
import starwave

print(starwave.visco_gsls_native_status(backend="native_cpu"))
starwave.prepare_visco_gsls(backend="native_cpu", device=None)
# After a matching CUDA build, with an available logical CUDA device:
starwave.prepare_visco_gsls(backend="cuda", device="cuda:0")
```

在已获准取得的完整 V17 源码根目录显式构建，使用已安装且与 Torch 匹配的工具链：

```bash
# 冻结 V17 的 core/elastic/GSLS 三库；按实际目标 SM 修改 80。
python compile_all.py --arch 80
# GSLS CUDA only:
python compile_visco_gsls.py --arch 80
# Separate native CPU library:
python compile_visco_gsls.py --backend cpu
```


冻结 V17 的 compile_all 构建/核验 core、elastic、GSLS 三库，Born 另用 compile_born.py；当前开发版 `3f2b24d` 的 compile_all 已包含 core/elastic/GSLS/Born 四库，VRR 仍单独构建。combined_build 仅 core/elastic。源树 GSLS 默认在 native/build/gsls，安装模式使用 StarWave cache。可选 STARWAVE_BUILD_DIR 下的 gsls 子目录必须与运行时一致；不用独立 STARWAVE_GSLS_BUILD_DIR。完整源树运行不要求 pip 注册或 export，也不下载编译器；可选 editable 安装与编译是两步。

GSLS 在导入时首次加载已有库时核验源码/库/metadata 指纹与 ABI，随后复用就绪句柄，不逐次传播重算源码哈希或轮询文件。不能借用其它树的库或绕过身份检查。更新已载入代码后须重建并重启 Python/Jupyter。普通首次传播或 DataParallel 无需手工 GSLS 准备；prepare_native 与 prepare_elastic 仍是各自独立的可选辅助函数。不进行运行时编译或后端回退。根 compile_all 不重建可能保留的 GSLS CPU 库；旧 CPU 库须单独重建。

CUDA 数值、实际多卡、长 FWI 和性能均须在目标设备单独验证；库存在、编译或预加载成功不能替代这些证据。本页示例不提供新的 GPU 或性能验收结果。

(visco-gsls-references)=
### 范围与参考

不支持 3D、rho 梯度、PML/dt 训练、boundary 重建、初始状态输入、终态返回、任意面自由表面/地形、压缩、照明、AMP、CUDA graphs 或受支持的高阶传播梯度。Torch 图的偶然高阶行为不构成契约。

- Hao & Greenhalgh (2021), “Nearly constant Q models of the generalized standard linear solid type and the corresponding wave equations”, GEOPHYSICS 86(4), T239–T260. [DOI](https://doi.org/10.1190/geo2020-0548.1).

论文描述物理模型；本 API 的负源、参考相速度归一化、离散采样、PML、扩边与伴随遵循本页契约，不声称与其他实现逐位相同。常规目标函数组织见[反演](inversion/fwi.md)。历史 SLS 实验不是当前 GSLS 的运行证据。

(native-runtime)=
## 原生运行库

(native-status)=
### starwave.native_status

```{py:function} starwave.native_status() -> dict

查询当前运行库状态，不执行编译或传播。

:returns: 状态字典；常用条目包括 `library_exists`、`library_loaded`、`torch_version`、`torch_cuda_version` 和 `cuda_available`。存在或加载成功均不代表数值验收通过。
:rtype: `dict`
```

(prepare-native)=
### starwave.prepare_native

```{py:function} starwave.prepare_native(device_ids: list[int] | tuple[int, ...]) -> dict

可选地验证显式逻辑 CUDA 设备并返回已有核心原生库状态。V17 导入在返回前已初始化可用核心库，普通首次传播或 DataParallel 无需此辅助步骤；不执行编译。

:param device_ids: 必填。非空、无重复、非负的可见逻辑 CUDA 编号；拒绝布尔值。编号按当前进程的设备可见性映射填写。
:type device_ids: `list[int] | tuple[int, ...]`
:returns: 包含原生状态、`selected_device_ids` 与准备消息的字典。初始化成功不是 GPU 数值测试。
:rtype: `dict`
```

(native-notes)=
## 原生库注意事项

显式构建后，普通 V17 `import starwave` 初始化可用 core、elastic、Born、GSLS 库，以及可导入的独立 VRR 扩展。共享分派在首次并行调用前已就绪，无需串行预热或强制 `prepare_*` 步骤。导入不编译、不初始化 CUDA、不选择设备、不分配波场，也不传播。尚未构建的库会跳过，以支持符合条件的纯 Torch 路径；已有但失效的构建会报错，不静默回退。构建路径应在导入前配置，重建已载入代码后重启 Python/Jupyter。就绪句柄会复用，不逐次传播重算源码哈希或轮询库文件。

`native_status()` 用于诊断，可选的 `prepare_native()` 用于检查显式逻辑 CUDA 设备；二者不编译，也不能替代 GPU 数值验收。示例中的设备 `0` 必须可见。core、elastic、GSLS 和 VRR 保留首次加载身份核验；Born 的完整哈希/符号审计仅在显式构建或开发审计时执行，运行时采用常规一次性加载。

本 scalar 开发整合保留导入时原生绑定；core 现在仅一次加载/绑定，不执行运行时源码/库哈希、指纹或版本探测，加载/绑定错误仍会报错。导入成功不等于源码/构建身份核验或数值检查。更新后按匹配源码重编译并重启 Python/Jupyter，prepare_native 仍是可选显式设备检查。

(native-examples)=
## 原生库示例

使用已审查 V17 源码、显式构建的匹配原生库与兼容 PyTorch，可在导入后查询状态。可选 prepare 调用要求可用 CUDA 设备。待发布 8.0.0 wheel 须独立核验，已发布 7.1.0 保留旧生命周期。

```python
import starwave

status = starwave.native_status()
print(status)
prepared = starwave.prepare_native([0])  # Optional device validation.
```

(prepare-elastic)=
## starwave.prepare_elastic

```{py:function} starwave.prepare_elastic(device_ids: list[int] | tuple[int, ...] | None=None) -> dict

可选地取得独立弹性库并校验显式逻辑设备。V17 导入已初始化可用弹性绑定，不再要求 CUDA/DataParallel 前必须手工预加载。不编译或执行传播测试。

:param device_ids: **默认 `None`。** None 加载弹性库但不选择或准备 CUDA 设备列表，适用于 CPU 使用；非空 list/tuple 指定无重复、非负的可见逻辑 CUDA 整数编号（拒绝 bool）。有 CUDA 请求时校验设备及构建能力。
:type device_ids: `list[int] | tuple[int, ...] | None`
:returns: 弹性库状态、selected_device_ids 与准备消息组成的字典；加载成功不等于数值或 DataParallel 验收。
:rtype: `dict`
```

`starwave.elastic_native_status()` 可只读查询弹性库状态。原有 `prepare_native()` 负责 scalar/VRZ/VTI；仅运行 elastic 不需要为它们额外准备。

(other-exports)=
## 其它导出名称与范围

`ScalarIllumination`、`IlluminationFields`、`precondition_gradient` 是已核验导出的二维 scalar 照明相关名称。本版暂不提供它们的完整生命周期教程；传播函数页会解释 `illumination` 参数的使用边界。

StarWave V17 没有公开 `starwave.Scalar` 包装类；`starwave.ScalarBorn` 是上文独立介绍的 Born 模块。教程中的 Module wrapper 由教程定义；不能将其它库的类名直接用于 StarWave，也不能把 elastic 的状态、`nt` 或存储选项添加到 scalar/VRZ/VTI 调用中。

{ref}`genindex`
