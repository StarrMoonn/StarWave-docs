# VRZ：速度与阻抗或密度

`starwave.vrz` 提供二维变密度声学传播。模型参数是正且有限的速度 `v`，以及**恰好一个** `impedance`（阻抗 Z）或 `density`（密度 rho）。两者不能同时传入，也不能同时省略；在同一单位体系下 `Z = v * rho`。

```python
records = starwave.vrz(
    v, grid_spacing=10.0, dt=0.001,
    density=rho,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
)[0]
```

`rho` 与 `v` 应具有相同形状、dtype 和设备。另一种参数化是将 `density=rho` 替换为 `impedance=Z`。这是调用片段，输入与原生库需预先准备。

## 与 scalar 的相同点和差别

VRZ 的输入采集形状、等间距限制和单元素返回元组与 scalar 相同；震源仍是归一化 forcing，内部应用速度相关缩放。VRZ 要求正模型值，不能套用 scalar 的 signed/zero 容许范围。

梯度回到调用时选择的参数化：训练 `v` 与 `Z` 和训练 `v` 与 `rho` 是不同的优化问题。不得不经链式法则就互换两种梯度。

## 限制

正值检查、数值可表示性检查与 CFL 规划不能保证任意强阻抗反差的空间稳定性。近零值及强反差可能导致病态。VRZ 不提供照明；签名中的 `illumination` 仅保留兼容位置，须为 `None`。实机 VRZ 数值、梯度与 FWI 验收待完成。
