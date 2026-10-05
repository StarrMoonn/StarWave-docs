# vti：二维与三维声学 VTI

```{contents} 本页内容
:local:
:depth: 2
```

## 函数

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

## 分量、时间与梯度

正的应力变化率对每个所选应力分量增加正应力；内部应用 `internal_dt*U(rate)`。没有 scalar 的 `-vp² dt²` 因子，也没有模型相关源缩放。旧的每步应力增量不能直接作为公开变化率输入。

应力记录在下采样前通过前置零、移去末项对齐时间；速度记录则将前后半时间步样本平均到整数时间（初始前半步取零）。两者输出都标在名义用户时间上，但物理量与空间位置不同。重采样可能出现端点振铃。

`vp`、`epsilon`、`delta`、`rho` 各自仅在 `requires_grad=True` 时请求梯度；固定字段仍参与正演和伴随。没有按模型数值自动冻结的规则，`nn.Parameter` 默认启用梯度；固定字段可注册为 buffer。

`epsilon<delta` 不被自动修复；较小 dt 不能保证消除负刚度增长或任意 PML 不稳定。速度包络的覆盖只是一项必要的输入条件。

模型梯度采用固定扩展模型、CFL 与 PML 的条件。重新生成物理边缘的 replicate padding 时，当前返回梯度不包含完整扩展链的转置累积，不能当作整个重建边界过程的全导数；切换 full 不改变这一限制。

## 示例

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

本例为二维输入；三维需同时使用 `[nx,ny,nz]` 四模型、三维坐标和相应间距。省略 `receiver_fields` 时只返回 `vz`，应使用单变量元组解包。VTI 没有 `illumination` 参数。完整独立 GPU 算例仍待补充，见[VTI 建模说明](../modeling/vti.md)。

## 注意事项

- 仅 CUDA FP32 默认 stream；每次 forward 只支持一次一阶 backward。源梯度、高阶导数、AMP、CUDA graphs、自定义 stream、公开初始/最终状态不在支持范围。
- 时间 CFL 子步不能弥补空间采样不足；空间分辨率警告是诊断，不是精度或稳定性证书。
- DataParallel 应把完整模型放入 Module，只拆分炮维采集；参见[多卡入门](../inversion/dataparallel.md)。
- 本页已核对接口契约，未完成目标 GPU 数值、性能或 FWI 验收。
