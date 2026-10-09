# Usage

SLS: [visco_sls](visco-sls.md).

```{container} sw-page-toc

**本页内容**

- [Scalar Function](#scalar)
- [VRZ Function](#vrz)
- [VTI Function](#vti)
- [Elastic Function](#elastic)
- [材料参数转换](#elastic-conversions)
- [starwave.native_status](#native-status)
- [starwave.prepare_native](#prepare-native)
- [starwave.prepare_elastic](#prepare-elastic)
- [其它导出名称与范围](#other-exports)
```

本参考以 StarWave **7.0.0** 公开 wheel（V14）为准，`starwave.scalar` 按模型维数支持二维与三维；二维 scalar 及 VRZ、VTI、elastic 的既有契约保留。每个传播函数提供真实签名、逐项参数、返回值、梯度范围、注意事项与调用示例。参数类型描述运行时接受的值；签名保留实际关键字边界和默认值。

6.0.0 的内部存储和 PML 转置维护见[发布说明](release-notes.md)，二维存储变化见[scalar 说明](modeling/scalar.md)。它们不引入新的公共参数；源码升级需重新构建配套原生库。

页面组织参考 [Deepwave 官方 Usage](https://ausargeo.com/deepwave/usage) 的 Sphinx Python API 风格，正文按 StarWave 的实际契约重新编写。两者的参数集合、源单位、返回结构和可微范围不能互换。

(propagators)=
## 传播函数速查

- {py:func}`starwave.scalar`：二维/三维标量声学；速度模型 `v`；返回单元素记录元组。

- {py:func}`starwave.vrz`：二维变密度声学；`v` 加恰好一种 `impedance` / `density` 参数化。

- {py:func}`starwave.vti`：二维/三维声学 VTI；`vp, epsilon, delta, rho`；按所选分量顺序返回记录。

- {py:func}`starwave.elastic`：二维/三维各向同性弹性；`lamb, mu, buoyancy`；返回完整最终状态及 p/速度记录。

符号约定：`B` 炮数、`S` 每炮源数、`R` 每炮接收点数、`T` 用户时间采样数、`D` 空间维数。源和接收点使用物理模型的整数网格下标，不是米坐标，不包含 PML 偏移。

(scalar)=
## Scalar Function

```{py:function} starwave.scalar(v: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, accuracy: int=8, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float | int=0.0, time_pad_frac: float | int=0.0, time_taper: bool=False, illumination: ScalarIllumination | None=None) -> tuple[torch.Tensor]

根据 `v.ndim` 在二维或三维网格上传播标量声学波，支持一批独立炮。两种维数都支持一次一阶速度梯度，三维还支持源波形梯度；坐标和数值设置不参与求导。每次调用使用新的传播状态，不返回最终波场。

:param v: **必填；位置或关键字参数。** 二维 `[N0,N1]` 或三维 `[N0,N1,N2]` 模型，每维至少为 2；CUDA float32，典型单位 m/s。由 `v.ndim` 选择传播维数。坐标 `[i,j]` / `[i,j,k]` 直接索引 `v[i,j]` / `v[i,j,k]`；三维示例采用 `[nx,ny,nz]`，不自动交换物理轴。所有值须有限；允许负值或零值，但不保证物理合理性。模型轴变换保留 autograd 链；`requires_grad=True` 时请求速度梯度。
:type v: `torch.Tensor`
:param grid_spacing: **必填；位置或关键字参数。** 正且有限的网格间距，典型单位 m。二维接受单值或两个相等的值；三维接受单值或按模型轴序排列的三个值（例如 `[dx,dy,dz]`），允许不等间距。拒绝布尔值、零/负值和长度不匹配。
:type grid_spacing: `float | int | list[float] | tuple[float, ...]`
:param dt: **必填；位置或关键字参数。** 正且有限的用户输入/输出采样间隔，典型单位 s。内部可能按 CFL 采用更小的时间步，返回间隔仍为此值。不能传入 Tensor 或布尔值，不对其求导。
:type dt: `float | int`
:param source_amplitudes: **必填；仅关键字参数。** 实数张量 `[B,S,T]`，三个维度均非空；数据须有限、可表示为 float32，会转换到模型设备和 float32。二维须固定（不能设置 `requires_grad=True`）；三维支持一阶源梯度，可单独或同时训练源与速度。它是归一化 forcing `f`，不是每步压力增量；若波场用 Pa、长度用 m，单位为 Pa/m²。不能只传 `[T]` 或 `[B,T]`，共享波形应显式扩展炮/源维。
:type source_amplitudes: `torch.Tensor`
:param source_locations: **必填；仅关键字参数。** 固定整数张量 `[B,S,D]`，`D=v.ndim`；单源时也接受 `[B,D]`。建议 `torch.long`。坐标按输入模型轴序直接索引物理网格；三维 `[i,j,k]` 对应 `v[i,j,k]`。不能包含 PML 偏移或越界值。每个源有对应波形，同一炮内重合源的增量相加。
:type source_locations: `torch.Tensor`
:param receiver_locations: **必填；仅关键字参数。** 固定整数张量 `[B,R,D]`，`D=v.ndim`、`R>0`，建议 `torch.long`。炮数与源一致，使用输入模型轴序的物理网格下标。允许重复位置；接收点不可省略，不能用空张量请求无记录传播。
:type receiver_locations: `torch.Tensor`
:param accuracy: **默认 `8`；仅关键字参数。** 空间有限差分阶数，可选 `2,4,6,8`，不是误差容差。影响模板范围、计算量和 boundary buffer 最小值；拒绝布尔值及其它阶数。
:type accuracy: `int`
:param pml_freq: **默认 `25.0`；仅关键字参数。** 正且有限的 PML 设置频率，典型单位 Hz，也用于空间采样诊断。不是波形生成器，不会自动测量源信号频率；应按实际实验选择。
:type pml_freq: `float | int`
:param pml_width: **默认 `20`；仅关键字参数。** 正整数网格厚度，单值应用到所有面；也接受二维四个、三维六个相等整数，按各模型轴前/后成对排列。零宽、非对称面宽和以零面宽请求自由表面均不支持。
:type pml_width: `int | list[int] | tuple[int, ...]`
:param boundary_buffer: **默认 `5`；仅关键字参数。** 非负空间网格 buffer，不是时间步数或 checkpoint 间隔。boundary 模式要求至少 `accuracy // 2 + 1`，8 阶时为 5；full 模式允许 0。每侧总填充为 `pml_width + boundary_buffer + accuracy // 2`。
:type boundary_buffer: `int`
:param memory: **默认 `'boundary'`；仅关键字参数。** 反传历史策略。`"boundary"` 保存压力边界条带与两个终态压力场并重构；三维六个面的保存宽度为 `M=accuracy//2`（Radius-M）。三维 `"full"` 保存逐内部时间步、完整填充体积的未缩放 `Lap(u)` 历史，适合小规模对照。三维只训练源也会保存所选历史；只有不需要速度或源梯度的正演才不保存。full 可能耗尽显存；没有自动降级、CPU/磁盘卸载或 `"checkpoint"` 选项。
:type memory: `str`
:param max_vel: **默认 `None`；仅关键字参数。** CFL/PML 速度包络，典型单位 m/s。`None` 时每次调用从当前模型 `max(abs(v))` 重新规划。显式值必须正且有限并覆盖该最大值；不是裁剪上限。全零模型需要显式正值。极值、时间子步与 PML 设置不参与求导。
:type max_vel: `float | int | None`
:param freq_taper_frac: **默认 `0.0`；仅关键字参数。** `[0,1]` 内有限比例，控制时间重采样中高频 FFT bin 的余弦 taper，数量按比例取整。重采样时即使为 0，最后一个正 rFFT bin 也会被抑制，奇数长度同样如此；不保证保留所有频率。内部重采样比为 1 时此设置不改变信号。
:type freq_taper_frac: `float | int`
:param time_pad_frac: **默认 `0.0`；仅关键字参数。** `[0,1]` 内有限比例；重采样前在尾部补零，补零数为 `int(time_pad_frac*T)` 用户样本，之后移除。不增加公开记录长度或实际传播时长；内部重采样比为 1 时不改变信号。
:type time_pad_frac: `float | int`
:param time_taper: **默认 `False`；仅关键字参数。** 是否在时间重采样中应用非周期 Hann 窗：上采样后、下采样前应用。改变信号及其反传转置；仅接受布尔值，不接受整数 0/1。内部重采样比为 1 时不改变信号。
:type time_taper: `bool`
:param illumination: **默认 `None`；仅关键字参数。** 二维可选的新 `ScalarIllumination` 收集器；需启用梯度且存在可训练模型。收集 detached 的源、接收与几何乘积统计；forward 后 seal，一次 backward 后 reduce。三维只接受 `None`，其它值会抛出 `NotImplementedError`。`None` 不分配照明 buffer；统计量不是精确 Hessian，也不自动预条件梯度。
:type illumination: `ScalarIllumination | None`
:returns: **`(receiver_amplitudes,)`**，恰好一个元素的元组。记录为连续 CUDA float32 张量 `[B,R,T]`，与模型在同一设备，名义时间 `0, dt, ..., (T-1)*dt`。为 pressure-like 记录，尺度取决于源和单位体系。用 `[0]` 或 `[-1]` 取出；不包含最终波场或 PML 状态。
:rtype: `tuple[torch.Tensor]`
```

(scalar-details)=
### 震源与自动微分

公开方程约定为 `u_tt = v² (Lap(u) - f)`。二维用户提供固定 `f`，三维还可训练 `f`；内部源增量为 `-v_source² * internal_dt² * U(f)`，其中 `U` 是内部上采样。不要在外部再次乘 `-v² dt²`。源位置的速度缩放链保留在模型梯度中。

步后记录在下采样前通过前置零、丢弃最后一步对齐用户时钟。时间重采样和移位的转置保留在 autograd 中；FFT 的非因果滤波可能使初始用户样本出现振铃。

模型梯度采用固定扩展模型、CFL 与 PML 的条件。重新生成物理边缘的 replicate padding 时，当前返回梯度不包含完整扩展链的转置累积，不能当作整个重建边界过程的全导数；切换 full 不改变这一限制。

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


(scalar-time-sampling)=
### CFL 与内部时间重采样

scalar 使用保守系数 0.6，按全部模型轴间距与 `max(abs(v))`（或经校验的 `max_vel`）计算时间上界。内部子步比 `r` 为满足该上界的正整数，`internal_dt=dt/r`、`internal_nt=T*r`。3D 不等间距会逐轴参与规划；`accuracy` 不会开启另一套公开 CFL 参数。时间规划、速度包络和 PML 系数是固定设置，不参与 autograd。

源波形先经 FFT 上采样，再施加源位置速度缩放；记录先做时间对齐，再降采样回 `[B,R,T]`。反传使用这些实际操作的转置，不能用简单重复或抽样代替。子步比为 1 时，合法 taper/padding 设置不改变信号；子步不能修复空间色散。

(scalar-memory)=
### 三维内存：full 与 Radius-M boundary

`M=accuracy//2` 是压力 Laplacian 的差分半径，和 `boundary_buffer`、`pml_width` 各司其职。boundary 保存六个宽度恰为 M 的压力面以及两个终态压力场；不保存完整时间体积，也不会失败后自动改为 full。full 保存逐内部步的未缩放 `Lap(u)` 体积历史。两种模式都保留传播/伴随工作场，三维 CPML 记忆使用方向条带；只训练源也会保存所选历史。显存还随内部时间步数和炮数增加，boundary 不保证大规模三维一定放得下。精确存储公式见{ref}`Scalar3D 重建说明 <reconstruction-scalar3d>`。

(scalar-3d-example)=
### 三维可运行调用：速度与源的一阶梯度

安装 7.0.0 并确保可见逻辑 CUDA 设备 0 可用后，可运行下面的独立小例子。模型采用 `[x,y,z]`，间距为 `[dx,dy,dz]`，坐标是网格下标。loss 仅检查求导接线。此代码已检查语法与接口；本次文档维护未运行 GPU，不把断言视为已通过的数值验收。

```python
import torch
import starwave

starwave.prepare_native([0])
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
    illumination=None,
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

- 仅 CUDA FP32 默认 stream；每次 forward 只支持一次一阶 backward。二维不支持源梯度；三维支持一次一阶源梯度。高阶导数、AMP、CUDA graphs、自定义 stream、CPU 传播、公开初始/最终状态不在支持范围。
- 时间 CFL 子步不能弥补空间采样不足；空间分辨率警告是诊断，不是精度或稳定性证书。
- DataParallel 应把完整模型放入 Module，只拆分炮维采集；参见[多卡入门](inversion/dataparallel.md)。
- 本页已核对接口契约，未完成目标 GPU 数值、性能或 FWI 验收。

(vrz)=
## VRZ Function

```{py:function} starwave.vrz(v: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, impedance: torch.Tensor | None=None, density: torch.Tensor | None=None, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, accuracy: int=8, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float | int=0.0, time_pad_frac: float | int=0.0, time_taper: bool=False, illumination: None=None) -> tuple[torch.Tensor]

使用速度和阻抗（或密度）进行二维声学传播，支持多炮批处理。恰好指定 `impedance` 或 `density` 一种参数化。输出可对所选模型参数求一次一阶梯度；源波形和采集坐标保持固定。

:param v: **必填；位置或关键字参数。** 二维模型 `[N0,N1]`，每个维度至少为 2；CUDA float32。典型单位 m/s。坐标 `[i,j]` 直接索引 `v[i,j]`；不自动推断物理轴序。 速度必须正且有限；不接受 scalar 的 signed/zero 模型域。设 `requires_grad=True` 时请求模型梯度。
:type v: `torch.Tensor`
:param grid_spacing: **必填；位置或关键字参数。** 正且有限的网格间距，典型单位 m。接受单值或两个相等的值；scalar/VRZ 不支持两轴不等间距。拒绝布尔值、零/负值和长度不匹配。
:type grid_spacing: `float | int | list[float] | tuple[float, ...]`
:param dt: **必填；位置或关键字参数。** 正且有限的用户输入/输出采样间隔，典型单位 s。内部可能按 CFL 采用更小的时间步，返回间隔仍为此值。不能传入 Tensor 或布尔值，不对其求导。
:type dt: `float | int`
:param impedance: **默认 `None`；仅关键字参数。** 声阻抗 `Z`，CUDA float32，与 `v` 形状和设备相同，正且有限。典型 SI 单位 kg/(m²·s)，与 `v*rho` 一致。与 `density` 恰好给出一个；两者同时提供或同时省略均不合法。训练此输入表示独立的速度/阻抗参数化。
:type impedance: `torch.Tensor | None`
:param density: **默认 `None`；仅关键字参数。** 密度 `rho`，CUDA float32，与 `v` 形状和设备相同，正且有限。选择一致单位体系；SI 时为 kg/m³。接口通过可微的 `Z=v*rho` 转换，梯度包含此链式法则，不隐式换算单位。与 `impedance` 恰好给出一个。
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
:param pml_width: **默认 `20`；仅关键字参数。** 正整数网格厚度，单值应用到所有面；也接受四个相等整数，按第一轴前/后、第二轴前/后排列。零宽、非对称面宽和以零面宽请求自由表面均不支持。
:type pml_width: `int | list[int] | tuple[int, ...]`
:param boundary_buffer: **默认 `5`；仅关键字参数。** 非负空间网格 buffer，不是时间步数或 checkpoint 间隔。boundary 模式要求至少 `accuracy // 2 + 1`，8 阶时为 5；full 模式允许 0。每侧总填充为 `pml_width + boundary_buffer + accuracy // 2`。
:type boundary_buffer: `int`
:param memory: **默认 `'boundary'`；仅关键字参数。** 模型反传历史策略。`"boundary"` 保存边界条带并重构，`"full"` 保存完整历史。full 可能耗尽显存；没有自动降级或 `"checkpoint"` 选项。没有模型梯度需求的正演不保留模型反传历史。
:type memory: `str`
:param max_vel: **默认 `None`；仅关键字参数。** CFL/PML 速度包络，典型单位 m/s。`None` 时每次根据当前 `max(v)` 重新规划；显式值须正且有限并覆盖当前最大速度，不是裁剪上限。规划与 PML 设置不参与求导。该主速度约束不保证任意阻抗反差的空间稳定性。
:type max_vel: `float | int | None`
:param freq_taper_frac: **默认 `0.0`；仅关键字参数。** `[0,1]` 内有限比例，控制时间重采样中高频 FFT bin 的余弦 taper，数量按比例取整。重采样时即使为 0，最后一个正 rFFT bin 也会被抑制，奇数长度同样如此；不保证保留所有频率。内部重采样比为 1 时此设置不改变信号。
:type freq_taper_frac: `float | int`
:param time_pad_frac: **默认 `0.0`；仅关键字参数。** `[0,1]` 内有限比例；重采样前在尾部补零，补零数为 `int(time_pad_frac*T)` 用户样本，之后移除。不增加公开记录长度或实际传播时长；内部重采样比为 1 时不改变信号。
:type time_pad_frac: `float | int`
:param time_taper: **默认 `False`；仅关键字参数。** 是否在时间重采样中应用非周期 Hann 窗：上采样后、下采样前应用。改变信号及其反传转置；仅接受布尔值，不接受整数 0/1。内部重采样比为 1 时不改变信号。
:type time_taper: `bool`
:param illumination: **默认 `None`；仅关键字参数。** 仅保留兼容关键字，必须为 `None`。VRZ 不支持照明，显式传入收集器或其它值会引发 `NotImplementedError`；不要沿用 scalar 的照明接线。
:type illumination: `None`
:returns: **`(receiver_amplitudes,)`**，单元素元组。唯一记录张量为 `[B,R,T]`，连续 CUDA float32，与模型在同一设备；名义时间 `0, dt, ..., (T-1)*dt`。是 pressure-like 记录，可用 `[0]` 或 `[-1]` 取得，不包含最终波场或边界状态。
:rtype: `tuple[torch.Tensor]`
```

(vrz-details)=
### 参数化、震源与梯度

使用 `impedance=Z` 时，速度和阻抗是两个独立输入；使用 `density=rho` 时，内部由 `v*rho` 得到阻抗。两种梯度的含义不同，不能不经过链式法则就互换。可训练输入用 `requires_grad=True`，其它介质值仍参与正演。

震源采用与 scalar 相同的归一化 forcing 协议，内部增量为 `-v_source² * internal_dt² * U(f)`，不是直接压力增量。不要额外乘时间步或速度缩放。记录在下采样前通过前置零并移去最后一步对齐用户时间；FFT 重采样可能出现端点振铃。

正且有限、系数可表示及 CFL 检查不保证任意强阻抗反差的稳定性；近零与强反差模型可能严重病态。

模型梯度采用固定扩展模型、CFL 与 PML 的条件。重新生成物理边缘的 replicate padding 时，当前返回梯度不包含完整扩展链的转置累积，不能当作整个重建边界过程的全导数；切换 full 不改变这一限制。

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

- 仅 CUDA FP32 默认 stream；每次 forward 只支持一次一阶 backward。源梯度、高阶导数、AMP、CUDA graphs、自定义 stream、公开初始/最终状态不在支持范围。
- 时间 CFL 子步不能弥补空间采样不足；空间分辨率警告是诊断，不是精度或稳定性证书。
- DataParallel 应把完整模型放入 Module，只拆分炮维采集；参见[多卡入门](inversion/dataparallel.md)。
- 本页已核对接口契约，未完成目标 GPU 数值、性能或 FWI 验收。

(vti)=
## VTI Function

```{py:function} starwave.vti(vp: torch.Tensor, epsilon: torch.Tensor, delta: torch.Tensor, rho: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, source_fields: str | list[str] | tuple[str, ...]=('sH', 'sV'), receiver_fields: str | list[str] | tuple[str, ...]=('vz',), accuracy: int=4, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float | int=0.0, time_pad_frac: float | int=0.0, time_taper: bool=False) -> tuple[torch.Tensor, ...]

传播 Duveneck 型一阶声学 VTI 系统，以 `vp.ndim` 选择二维或三维。输出按所选接收分量排列；四个模型可独立请求一次一阶梯度。固定模型仍参与全部物理计算；本接口不是完整弹性 VTI。

:param vp: **必填；位置或关键字参数。** 竖直 P 波速度，单位 m/s。CUDA float32，正且有限，形状 `[nx,nz]` 或 `[nx,ny,nz]`，每维至少为 2；最后一轴始终竖直。四个模型必须形状、dtype、设备一致；不会自动推断或转换物理轴序。
:type vp: `torch.Tensor`
:param epsilon: **必填；位置或关键字参数。** 无量纲 Thomsen epsilon，CUDA float32，与 `vp` 形状/设备相同；有限且严格大于 -0.5，并须满足系数和导数的数值可表示性。允许 `epsilon<delta`，不作隐式裁剪；存在增长模式的已知限制。
:type epsilon: `torch.Tensor`
:param delta: **必填；位置或关键字参数。** 无量纲 Thomsen delta，CUDA float32，与 `vp` 形状/设备相同；有限且严格大于 -0.5，并须满足系数和导数的数值可表示性。没有 `delta<=epsilon` 的输入强制约束，不会隐式改模型。
:type delta: `torch.Tensor`
:param rho: **必填；位置或关键字参数。** 密度，单位 kg/m³。CUDA float32，与 `vp` 形状/设备相同，正且有限；倒数、刚度及导数须可表示。无默认密度，不做隐式单位换算；若原数据单位已确认为 g/cm³，应在调用前显式乘 1000。
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
:param pml_width: **默认 `20`；仅关键字参数。** 正整数网格厚度，单值应用到所有面；或包含 `2*D` 个相等整数的序列，按各轴前/后成对排列。非对称、零宽面和自由表面均不支持。
:type pml_width: `int | list[int] | tuple[int, ...]`
:param boundary_buffer: **默认 `5`；仅关键字参数。** 非负空间网格 buffer，不是时间步数或 checkpoint 间隔。boundary 模式要求至少 `accuracy // 2 + 1`，8 阶时为 5；full 模式允许 0。每侧总填充为 `pml_width + boundary_buffer + accuracy // 2`。
:type boundary_buffer: `int`
:param memory: **默认 `'boundary'`；仅关键字参数。** 模型反传历史策略。`"boundary"` 保存边界条带并重构，`"full"` 保存完整历史。full 可能耗尽显存；没有自动降级或 `"checkpoint"` 选项。没有模型梯度需求的正演不保留模型反传历史。
:type memory: `str`
:param max_vel: **默认 `None`；仅关键字参数。** CFL/PML 各向异性速度包络，单位 m/s。`None` 时动态计算；显式值须正、有限并覆盖 `max(vp*sqrt(max(1,1+2*epsilon,sqrt(1+2*delta))))`。只覆盖 `max(vp)` 不足。密度反差仍重新计算；规划还取决于维数、间距和差分系数，不参与求导。
:type max_vel: `float | int | None`
:param freq_taper_frac: **默认 `0.0`；仅关键字参数。** `[0,1]` 内有限比例，控制时间重采样中高频 FFT bin 的余弦 taper，数量按比例取整。重采样时即使为 0，最后一个正 rFFT bin 也会被抑制，奇数长度同样如此；不保证保留所有频率。内部重采样比为 1 时此设置不改变信号。
:type freq_taper_frac: `float | int`
:param time_pad_frac: **默认 `0.0`；仅关键字参数。** `[0,1]` 内有限比例；重采样前在尾部补零，补零数为 `int(time_pad_frac*T)` 用户样本，之后移除。不增加公开记录长度或实际传播时长；内部重采样比为 1 时不改变信号。
:type time_pad_frac: `float | int`
:param time_taper: **默认 `False`；仅关键字参数。** 是否在时间重采样中应用非周期 Hann 窗：上采样后、下采样前应用。改变信号及其反传转置；仅接受布尔值，不接受整数 0/1。内部重采样比为 1 时不改变信号。
:type time_taper: `bool`
:returns: **`(records_0, ..., records_n)`**，每个 `receiver_fields` 条目对应一个 `[B,R,T]` 连续 CUDA float32 张量，顺序严格对应输入分量顺序。默认只有 `(vz_records,)`，为竖直质点速度 m/s，不是压力；`sH/sV` 为应力 Pa。张量与模型在同一设备，名义用户时间 `0, dt, ..., (T-1)*dt`，不包含最终状态。
:rtype: `tuple[torch.Tensor, ...]`
```

(vti-details)=
### 分量、时间与梯度

正的应力变化率对每个所选应力分量增加正应力；内部应用 `internal_dt*U(rate)`。没有 scalar 的 `-vp² dt²` 因子，也没有模型相关源缩放。旧的每步应力增量不能直接作为公开变化率输入。

应力记录在下采样前通过前置零、移去末项对齐时间；速度记录则将前后半时间步样本平均到整数时间（初始前半步取零）。两者输出都标在名义用户时间上，但物理量与空间位置不同。重采样可能出现端点振铃。

`vp`、`epsilon`、`delta`、`rho` 各自仅在 `requires_grad=True` 时请求梯度；固定字段仍参与正演和伴随。没有按模型数值自动冻结的规则，`nn.Parameter` 默认启用梯度；固定字段可注册为 buffer。

`epsilon<delta` 不被自动修复；较小 dt 不能保证消除负刚度增长或任意 PML 不稳定。速度包络的覆盖只是一项必要的输入条件。

模型梯度采用固定扩展模型、CFL 与 PML 的条件。重新生成物理边缘的 replicate padding 时，当前返回梯度不包含完整扩展链的转置累积，不能当作整个重建边界过程的全导数；切换 full 不改变这一限制。

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

- 仅 CUDA FP32 默认 stream；每次 forward 只支持一次一阶 backward。源梯度、高阶导数、AMP、CUDA graphs、自定义 stream、公开初始/最终状态不在支持范围。
- 时间 CFL 子步不能弥补空间采样不足；空间分辨率警告是诊断，不是精度或稳定性证书。
- DataParallel 应把完整模型放入 Module，只拆分炮维采集；参见[多卡入门](inversion/dataparallel.md)。
- 本页已核对接口契约，未完成目标 GPU 数值、性能或 FWI 验收。

(elastic)=
## Elastic Function

```{py:function} starwave.elastic(lamb: torch.Tensor, mu: torch.Tensor, buoyancy: torch.Tensor, grid_spacing: Union[float, Sequence[float]], dt: float, source_amplitudes_z: Optional[torch.Tensor]=None, source_amplitudes_y: Optional[torch.Tensor]=None, source_amplitudes_x: Optional[torch.Tensor]=None, source_amplitudes_p: Optional[torch.Tensor]=None, source_locations_z: Optional[torch.Tensor]=None, source_locations_y: Optional[torch.Tensor]=None, source_locations_x: Optional[torch.Tensor]=None, source_locations_p: Optional[torch.Tensor]=None, receiver_locations_z: Optional[torch.Tensor]=None, receiver_locations_y: Optional[torch.Tensor]=None, receiver_locations_x: Optional[torch.Tensor]=None, receiver_locations_p: Optional[torch.Tensor]=None, accuracy: int=4, pml_width: Union[int, Sequence[int]]=20, pml_freq: Optional[float]=None, max_vel: Optional[float]=None, survey_pad: Optional[Union[int, Sequence[Optional[int]]]]=None, vz_0: Optional[torch.Tensor]=None, vy_0: Optional[torch.Tensor]=None, vx_0: Optional[torch.Tensor]=None, sigmazz_0: Optional[torch.Tensor]=None, sigmayz_0: Optional[torch.Tensor]=None, sigmaxz_0: Optional[torch.Tensor]=None, sigmayy_0: Optional[torch.Tensor]=None, sigmaxy_0: Optional[torch.Tensor]=None, sigmaxx_0: Optional[torch.Tensor]=None, m_vzz_0: Optional[torch.Tensor]=None, m_vzy_0: Optional[torch.Tensor]=None, m_vzx_0: Optional[torch.Tensor]=None, m_vyz_0: Optional[torch.Tensor]=None, m_vxz_0: Optional[torch.Tensor]=None, m_vyy_0: Optional[torch.Tensor]=None, m_vyx_0: Optional[torch.Tensor]=None, m_vxy_0: Optional[torch.Tensor]=None, m_vxx_0: Optional[torch.Tensor]=None, m_sigmazzz_0: Optional[torch.Tensor]=None, m_sigmayzy_0: Optional[torch.Tensor]=None, m_sigmaxzx_0: Optional[torch.Tensor]=None, m_sigmayzz_0: Optional[torch.Tensor]=None, m_sigmaxzz_0: Optional[torch.Tensor]=None, m_sigmayyy_0: Optional[torch.Tensor]=None, m_sigmaxyy_0: Optional[torch.Tensor]=None, m_sigmaxyx_0: Optional[torch.Tensor]=None, m_sigmaxxx_0: Optional[torch.Tensor]=None, origin: Optional[Sequence[int]]=None, nt: Optional[int]=None, model_gradient_sampling_interval: int=1, freq_taper_frac: float=0.0, time_pad_frac: float=0.0, time_taper: bool=False, forward_callback: Optional[common.Callback]=None, callback_frequency: int=1, python_backend: Union[Literal['eager', 'jit', 'compile'], bool]=False, storage_mode: Literal['device', 'cpu', 'disk', 'none']='device', storage_path: str='.', storage_compression: bool=False, *, memory: Literal['full', 'boundary']='full') -> Tuple[torch.Tensor, ...]

二维/三维各向同性弹性波传播，使用 Deepwave 0.0.27 派生后端。输入是原始 Lamé 参数和浮力参数，支持多炮、材料/源/初始状态的一阶自动微分。除最后的 memory 外，所有参数均可按位置或关键字传入；只有前五项必填。

:param lamb: **必填。** 第一 Lamé 参数，典型单位 Pa。空间形状 `[Ny,Nx]`（二维）或 `[Nz,Ny,Nx]`（三维）；可选前导模型批次维为 1 或 B。CPU/CUDA float32 或 float64，值须有限。与 mu、buoyancy 的空间形状、dtype 和 device 相同；不会识别或自动转换 vp。
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
:param pml_width: **默认 `20`。** PML 网格厚度：非负整数或长度 2D 的序列，顺序为每个空间轴的低/高侧。零面宽移除该侧吸收层，不等于通用应力自由表面开关。
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
:param model_gradient_sampling_interval: **默认 `1`。** 模型梯度的时间采样间隔，整数 ≥1；内部累计步幅为 CFL 子步比 q 乘本值。默认 1 最直接；大于 1 是采样梯度，不能承诺与逐步累计相同。当前原生路径只处理完整采样组，详见存储与采样说明。
:type model_gradient_sampling_interval: `int`
:param freq_taper_frac: **默认 `0.0`。** CFL 引起 FFT 重采样时，对频谱高频端施加余弦 taper 的比例。影响源和记录，不改变用户输出长度。
:type freq_taper_frac: `float`
:param time_pad_frac: **默认 `0.0`。** FFT 重采样前追加零的长度比例，之后移除；作用于源和记录，不增加公开记录的时长。
:type time_pad_frac: `float`
:param time_taper: **默认 `False`。** 是否在重采样时对源和记录施加时间 Hann 窗。
:type time_taper: `bool`
:param forward_callback: **默认 `None`。** 接收 `starwave.common.CallbackState` 的观测函数。原生回调按后端外循环组触发；state.dt 是内部 h，state.step 是外循环索引。boundary 只允许只读快照。
:type forward_callback: `Optional[common.Callback]`
:param callback_frequency: **默认 `1`。** 正整数；两次回调之间的后端外循环组数。原生路径一组跨越 q × model_gradient_sampling_interval 个内部步，不是固定的用户记录采样间隔。
:type callback_frequency: `int`
:param python_backend: **默认 `False`。** False 选择 C/CUDA；`"eager"`、`"jit"`、`"compile"` 显式选择 PyTorch 路径。True 在已核验构建声明 OpenMP 时选择 compile，否则 jit；能力依赖 PyTorch/工具链。仅 full 接受 Python 路径，且要求 device 存储、无压缩。
:type python_backend: `Union[Literal['eager', 'jit', 'compile'], bool]`
:param storage_mode: **默认 `'device'`。** 中间数据存储，可选 `"device"`、`"cpu"`、`"disk"`、`"none"`。只在 full 原生路径支持 offload/disk/none；none 不提供完整材料梯度，详见下方说明。
:type storage_mode: `Literal['device', 'cpu', 'disk', 'none']`
:param storage_path: **默认 `'.'`。** disk 模式临时历史文件的目录。保留计算图时也可能保留这些文件；反传仍需使用的文件不能提前删除。
:type storage_path: `str`
:param storage_compression: **默认 `False`。** full 原生路径的有损中间数据压缩，可能改变模型梯度。boundary 和 Python 后端不接受 True。
:type storage_compression: `bool`
:param memory: **默认 `'full'`。** StarWave 的仅关键字扩展。`"full"` 保存体积历史；`"boundary"` 使用 CUDA 边界存储与重构。二者支持二维/三维及 2/4/6/8 阶；限制见下方，未支持组合不会自动退回 full。
:type memory: `Literal['full', 'boundary']`
:returns: 完整最终波场/PML 状态与记录元组：二维 16 个张量，三维 31 个张量。记录形状 `[B,R,T]`，未请求分量为空张量；下方给出精确顺序。
:rtype: `Tuple[torch.Tensor, ...]`
```

(elastic-axes)=
### 模型轴序与交错网格

二维的 API 轴名为 `[y,x]`，三维为 `[z,y,x]`，名称对应输入张量的空间轴，不能脱离数组约定固定解释为物理垂直方向。若二维数组已经是 `[depth,horizontal]`，直接输入时 y 就是深度，垂直力使用 source_amplitudes_y；若显式转置为 `[horizontal,depth]`，深度就对应 x，源/接收分量、坐标、grid_spacing、PML 各面和 origin 均须一致变换。两种材料转换函数不做转置。

材料可共享或按炮给出，源和记录分别为 `[B,S,T]` 与 `[B,R,T]`；坐标最后一维严格遵循同一空间轴序。交错网格中分量位于不同半格位置，不能把不同分量当成同一点的同一种量。速度源/接收点在自身分量轴上不能取模型最后一个网格下标。参见 [Deepwave 弹性交错网格](https://ausargeo.com/deepwave/elastic.html)。

源 y/x（以及三维 z）是力密度，源 p 是压力变化率；接收 y/x/z 是速度，接收 p 是法向应力负平均。名义上压力对应 `t*dt`、速度对应 `(t-0.5)*dt`。内部 CFL 细分时使用内部半步与 FFT 重采样，不额外提供半步重新对齐操作；不要把 scalar 的 forcing 或记录时钟直接套用在这里。

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
- `memory="boundary"` 是 StarWave 扩展：仅 CUDA，要求 `python_backend=False`、`storage_mode="device"`、`storage_compression=False`。裁剪后每个物理维度须大于 accuracy；请求 buoyancy 梯度时，有效交错 buoyancy 不能为零。无 CPU/offload/压缩或自动 full 回退。内存/速度收益随域形状、PML 与采样而变。
- 默认 `model_gradient_sampling_interval=1`，CFL 比 q>1 时材料梯度步幅仍为 q，并非每个内部步都累计。若设为大于 1 的整数，当前原生 full/boundary 仅执行完整采样组：非整组尾部不传播；正时长不足一组会报错（有回调时可能在反传发生）。保持默认值最直接；需要采样时，让间隔不超过用户 nt 且整除 nt，并核对采样梯度的适用性。
- 普通反传释放 autograd 保存的张量历史；保留计算图可延长历史寿命。disk 临时文件随相应输出/loss 图释放；日志只保留 detached 数值，仍需反传的文件不要提前删除。

(elastic-examples)=
### 最小调用示例

已安装 [StarWave 4.0.0](installation.md) 且逻辑 CUDA 设备 0 可用时，以下构造 `[depth,horizontal]` 模型并沿深度施力。它演示 vp/vs/rho → 显式转换 → elastic → 一阶 Vs 梯度；此处未执行 GPU 实验，也不作为正确性或收敛验收。

```python
import torch
import starwave

# 输入数组为 [depth, horizontal]，对应 elastic 的 [y, x]。
device = torch.device("cuda:0")
starwave.prepare_elastic([0])
vp = torch.full((48, 64), 2200.0, device=device)
vs = torch.full_like(vp, 1100.0, requires_grad=True)
rho = torch.full_like(vp, 2000.0)
lamb, mu, buoyancy = starwave.common.vpvsrho_to_lambmubuoyancy(vp, vs, rho)

dt, nt, freq = 0.001, 100, 15.0
t = torch.arange(nt, device=device) * dt
phase = torch.pi * freq * (t - 0.04)
wavelet = (1.0 - 2.0 * phase.square()) * torch.exp(-phase.square())
force_y = wavelet.reshape(1, 1, nt)  # [shot, source, time]，力密度
source_y = torch.tensor([[[8, 32]]], device=device, dtype=torch.long)
receiver_y = torch.tensor([[[8, 20], [8, 32], [8, 44]]],
                          device=device, dtype=torch.long)
outputs = starwave.elastic(
    lamb, mu, buoyancy, grid_spacing=(10.0, 10.0), dt=dt,
    source_amplitudes_y=force_y, source_locations_y=source_y,
    receiver_locations_y=receiver_y, pml_freq=freq, accuracy=4,
    memory="boundary",
)
record_y = outputs[-2]  # [1, 3, 100]；最后一项始终是 x，即使未请求
loss = record_y.square().mean()
loss.backward()
assert vs.grad is not None
```

对于 CPU 示例，将 device 改为 `torch.device("cpu")`，调用 `starwave.prepare_elastic()`，并改用 `memory="full"`。三维模型按 `[z,y,x]` 构造；若第一轴是深度，垂直力对应 z，坐标最后一维为 3，记录 p/z/y/x 为 `outputs[-4:]`。

(elastic-notes)=
### 使用边界

- 原生传播提供一阶 AD，不承诺高阶导数。转换函数保留 PyTorch 链，但不增加传播器的高阶导数能力。原始材料没有物理符号修复；负开方、病态材料或分辨率不足仍可能导致无效结果。
- 没有公开 backward_callback、free_surface、boundary_buffer、source_fields 或 receiver_fields 参数；使用实际分量关键字。这里的 nt/storage 选项也不能直接加到 scalar、VRZ、VTI。
- DataParallel 前在主线程调用 {py:func}`starwave.prepare_elastic`，完整材料保存在 Module 中，仅拆分炮维采集；elastic 与原有传播器使用独立准备入口。初始化成功不等于多 GPU 数值验收。
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

在主线程验证并预加载已有的原生库，供后续传播或 DataParallel 使用，不执行编译。

:param device_ids: 必填。非空、无重复、非负的可见逻辑 CUDA 编号；拒绝布尔值。编号按当前进程的设备可见性映射填写。
:type device_ids: `list[int] | tuple[int, ...]`
:returns: 包含原生状态、`selected_device_ids` 与准备消息的字典。初始化成功不是 GPU 数值测试。
:rtype: `dict`
```

(native-notes)=
## 原生库注意事项

`native_status()` 用于排查安装与加载状态；`prepare_native()` 应在主线程、传播或 DataParallel 开始前调用。两者都不编译库，也不替代 GPU 数值验收。示例中的逻辑设备 `0` 必须对当前进程可见。

(native-examples)=
## 原生库示例

安装公开 wheel 和匹配的 PyTorch 后，可查询状态；后续准备调用要求有可用 CUDA 设备。

```python
import starwave

status = starwave.native_status()
print(status)
prepared = starwave.prepare_native([0])
```

(prepare-elastic)=
## starwave.prepare_elastic

```{py:function} starwave.prepare_elastic(device_ids: list[int] | tuple[int, ...] | None=None) -> dict

在主线程加载独立弹性原生库；CUDA/DataParallel 使用前显式准备逻辑设备。不编译或执行传播测试。

:param device_ids: **默认 `None`。** None 加载弹性库但不选择或准备 CUDA 设备列表，适用于 CPU 使用；非空 list/tuple 指定无重复、非负的可见逻辑 CUDA 整数编号（拒绝 bool）。有 CUDA 请求时校验设备及构建能力。
:type device_ids: `list[int] | tuple[int, ...] | None`
:returns: 弹性库状态、selected_device_ids 与准备消息组成的字典；加载成功不等于数值或 DataParallel 验收。
:rtype: `dict`
```

`starwave.elastic_native_status()` 可只读查询弹性库状态。原有 `prepare_native()` 负责 scalar/VRZ/VTI；仅运行 elastic 不需要为它们额外准备。

(other-exports)=
## 其它导出名称与范围

`ScalarIllumination`、`IlluminationFields`、`precondition_gradient` 是已核验导出的二维 scalar 照明相关名称。本版暂不提供它们的完整生命周期教程；传播函数页会解释 `illumination` 参数的使用边界。

StarWave 7.0.0 没有公开 `starwave.Scalar` 包装类。教程中的 Module wrapper 由教程定义；不能将其它库的类名直接用于 StarWave，也不能把 elastic 的状态、`nt` 或存储选项添加到 scalar/VRZ/VTI 调用中。

{ref}`genindex`
