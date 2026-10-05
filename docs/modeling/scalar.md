# 标量声学 scalar

`starwave.scalar` 接受二维速度模型 `v`，返回单元素元组 `(receiver_amplitudes,)`。使用 `[0]` 取得形状为 `[B,R,T]` 的 pressure-like 记录。

模型允许有限正值、零值和负值，但这不是对所有值的物理合理性背书；常规速度模型通常选择正值。CFL 自动规划使用 `max(abs(v))`；全零模型必须显式给出正的 `max_vel`。

## 震源与调用

公开震源是归一化 forcing，满足 `u_tt = v² (Lap(u) - f)`。用户提供固定 `f`，不要额外手动乘以 `-v² dt²`。如果 `u` 以 Pa、长度以 m 计，`f` 对应 Pa/m²。

```python
records = starwave.scalar(
    v, grid_spacing=10.0, dt=0.001,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
)[0]
```

此片段依赖已准备的输入；完整独立程序见[快速入门](../quickstart.md)。逐项参数与默认值见 [scalar API 参考](../api/scalar.md)。

## 梯度与照明

设 `v.requires_grad=True`，从记录构造标量 loss 后调用一次 `backward()`。源位置速度因子也属于模型梯度链；震源波形本身不可训练。可选 `ScalarIllumination` 只适用于 scalar，统计量不能称为精确 Hessian；其完整使用教程暂未纳入首版。

减小时间步不能解决空间采样不足。进行反演前需单独检验波形、空间分辨率、梯度与[边界梯度限制](conventions.md)。
