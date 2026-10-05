# scalar：二维标量声学

```{contents} 本页内容
:local:
:depth: 2
```

## 函数

```{py:function} starwave.scalar(v: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, accuracy: int=8, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float | int=0.0, time_pad_frac: float | int=0.0, time_taper: bool=False, illumination: ScalarIllumination | None=None) -> tuple[torch.Tensor]

在二维等间距网格上传播标量声学波，支持一批独立炮。输出可对速度模型求一次一阶梯度；源波形、坐标和数值设置不作为可训练输入。每次调用使用新的传播状态，不返回最终波场。

:param v: **必填；位置或关键字参数。** 二维模型 `[N0,N1]`，每个维度至少为 2；CUDA float32。典型单位 m/s。坐标 `[i,j]` 直接索引 `v[i,j]`；不自动推断物理轴序。 所有值须有限；允许负值或零值，但这不保证物理合理性。模型轴变换保留 autograd 链；`requires_grad=True` 时请求速度梯度。
:type v: `torch.Tensor`
:param grid_spacing: **必填；位置或关键字参数。** 正且有限的网格间距，典型单位 m。接受单值或两个相等的值；scalar/VRZ 不支持两轴不等间距。拒绝布尔值、零/负值和长度不匹配。
:type grid_spacing: `float | int | list[float] | tuple[float, ...]`
:param dt: **必填；位置或关键字参数。** 正且有限的用户输入/输出采样间隔，典型单位 s。内部可能按 CFL 采用更小的时间步，返回间隔仍为此值。不能传入 Tensor 或布尔值，不对其求导。
:type dt: `float | int`
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
:param max_vel: **默认 `None`；仅关键字参数。** CFL/PML 速度包络，典型单位 m/s。`None` 时每次调用从当前模型 `max(abs(v))` 重新规划。显式值必须正且有限并覆盖该最大值；不是裁剪上限。全零模型需要显式正值。极值、时间子步与 PML 设置不参与求导。
:type max_vel: `float | int | None`
:param freq_taper_frac: **默认 `0.0`；仅关键字参数。** `[0,1]` 内有限比例，控制时间重采样中高频 FFT bin 的余弦 taper，数量按比例取整。重采样时即使为 0，最后一个正 rFFT bin 也会被抑制，奇数长度同样如此；不保证保留所有频率。内部重采样比为 1 时此设置不改变信号。
:type freq_taper_frac: `float | int`
:param time_pad_frac: **默认 `0.0`；仅关键字参数。** `[0,1]` 内有限比例；重采样前在尾部补零，补零数为 `int(time_pad_frac*T)` 用户样本，之后移除。不增加公开记录长度或实际传播时长；内部重采样比为 1 时不改变信号。
:type time_pad_frac: `float | int`
:param time_taper: **默认 `False`；仅关键字参数。** 是否在时间重采样中应用非周期 Hann 窗：上采样后、下采样前应用。改变信号及其反传转置；仅接受布尔值，不接受整数 0/1。内部重采样比为 1 时不改变信号。
:type time_taper: `bool`
:param illumination: **默认 `None`；仅关键字参数。** 可选的新 `ScalarIllumination` 收集器；需启用梯度且存在可训练模型。收集 detached 的源、接收与几何乘积统计；forward 后 seal，一次 backward 后 reduce。`None` 不分配照明 buffer；本参数不自动预条件梯度，统计量也不是精确 Hessian。
:type illumination: `ScalarIllumination | None`
:returns: **`(receiver_amplitudes,)`**，恰好一个元素的元组。记录为连续 CUDA float32 张量 `[B,R,T]`，与模型在同一设备，名义时间 `0, dt, ..., (T-1)*dt`。为 pressure-like 记录，尺度取决于源和单位体系。用 `[0]` 或 `[-1]` 取出；不包含最终波场或 PML 状态。
:rtype: `tuple[torch.Tensor]`
```

## 震源与自动微分

公开方程约定为 `u_tt = v² (Lap(u) - f)`。用户提供固定 `f`；内部源增量为 `-v_source² * internal_dt² * U(f)`，其中 `U` 是内部上采样。不要在外部再次乘 `-v² dt²`。源位置的速度缩放链保留在模型梯度中。

步后记录在下采样前通过前置零、丢弃最后一步对齐用户时钟。时间重采样和移位的转置保留在 autograd 中；FFT 的非因果滤波可能使初始用户样本出现振铃。

模型梯度采用固定扩展模型、CFL 与 PML 的条件。重新生成物理边缘的 replicate padding 时，当前返回梯度不包含完整扩展链的转置累积，不能当作整个重建边界过程的全导数；切换 full 不改变这一限制。

## 示例

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

完整、无数据文件依赖的输入构造与一次 FWI 更新见[快速入门](../quickstart.md)和[FWI](../inversion/fwi.md)。默认 `pml_freq=25.0` 不会随源频率自动调整。

## 注意事项

- 仅 CUDA FP32 默认 stream；每次 forward 只支持一次一阶 backward。源梯度、高阶导数、AMP、CUDA graphs、自定义 stream、公开初始/最终状态不在支持范围。
- 时间 CFL 子步不能弥补空间采样不足；空间分辨率警告是诊断，不是精度或稳定性证书。
- DataParallel 应把完整模型放入 Module，只拆分炮维采集；参见[多卡入门](../inversion/dataparallel.md)。
- 本页已核对接口契约，未完成目标 GPU 数值、性能或 FWI 验收。
