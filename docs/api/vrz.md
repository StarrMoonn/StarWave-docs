# vrz：二维变密度声学

```{contents} 本页内容
:local:
:depth: 2
```

## 函数

```{py:function} starwave.vrz(v, grid_spacing, dt, *, impedance=None, density=None, source_amplitudes, source_locations, receiver_locations, accuracy=8, pml_freq=25.0, pml_width=20, boundary_buffer=5, memory='boundary', max_vel=None, freq_taper_frac=0.0, time_pad_frac=0.0, time_taper=False, illumination=None)

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

## 参数化、震源与梯度

使用 `impedance=Z` 时，速度和阻抗是两个独立输入；使用 `density=rho` 时，内部由 `v*rho` 得到阻抗。两种梯度的含义不同，不能不经过链式法则就互换。可训练输入用 `requires_grad=True`，其它介质值仍参与正演。

震源采用与 scalar 相同的归一化 forcing 协议，内部增量为 `-v_source² * internal_dt² * U(f)`，不是直接压力增量。不要额外乘时间步或速度缩放。记录在下采样前通过前置零并移去最后一步对齐用户时间；FFT 重采样可能出现端点振铃。

正且有限、系数可表示及 CFL 检查不保证任意强阻抗反差的稳定性；近零与强反差模型可能严重病态。

模型梯度采用固定扩展模型、CFL 与 PML 的条件。重新生成物理边缘的 replicate padding 时，当前返回梯度不包含完整扩展链的转置累积，不能当作整个重建边界过程的全导数；切换 full 不改变这一限制。

## 示例

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

若选择阻抗参数化，将 `density=rho` 替换为 `impedance=Z`，不要同时保留两者。更多背景见[VRZ 建模说明](../modeling/vrz.md)。完整独立 GPU 算例仍待补充。

## 注意事项

- 仅 CUDA FP32 默认 stream；每次 forward 只支持一次一阶 backward。源梯度、高阶导数、AMP、CUDA graphs、自定义 stream、公开初始/最终状态不在支持范围。
- 时间 CFL 子步不能弥补空间采样不足；空间分辨率警告是诊断，不是精度或稳定性证书。
- DataParallel 应把完整模型放入 Module，只拆分炮维采集；参见[多卡入门](../inversion/dataparallel.md)。
- 本页已核对接口契约，未完成目标 GPU 数值、性能或 FWI 验收。
