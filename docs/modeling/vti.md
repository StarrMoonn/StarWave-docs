# 声学 VTI：二维与三维

`starwave.vti` 是 Duveneck 型声学 VTI（垂直对称轴的横向各向同性）接口，不是完整弹性波接口。四个模型参数是 `vp`、`epsilon`、`delta`、`rho`，可独立通过 `requires_grad` 选择训练；固定参数仍参与正演。

| 模型 | 条件 | 典型单位 |
|---|---|---|
| `vp` | 正且有限 | m/s |
| `rho` | 正且有限 | kg/m³ |
| `epsilon` | 有限且 > -0.5 | 无量纲 |
| `delta` | 有限且 > -0.5 | 无量纲 |

二维轴序为 `[x,z]`，三维为 `[x,y,z]`。最后一轴必须竖直；网格间距按相同轴序填写，可不相等。

## 源和记录分量

震源 `source_amplitudes` 是 **Pa/s 应力变化率**，默认同时注入 `sH` 与 `sV`；内部执行时间积分，不要在外部再次乘 `dt`，也不要套 scalar 的 `-vp² dt²` 归一化。

默认 `receiver_fields=("vz",)`，返回 `vz`（m/s）。二维可选 `vx,vz,sH,sV`，三维另有 `vy`。应力为 Pa；返回元组按 `receiver_fields` 顺序排列，每项形状 `[B,R,T]`。

```python
stress_h, velocity_z = starwave.vti(
    vp, epsilon, delta, rho,
    grid_spacing=[10.0, 10.0], dt=0.001,
    source_amplitudes=stress_rate,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    receiver_fields=("sH", "vz"),
)
```

此调用片段需预先准备有效输入。速度位于对应轴的正向半网格位置，应力位于网格中心；接口不替用户做空间插值。输出对齐名义用户时间，但不能据此把应力与速度当成同一物理量。逐项参数与返回顺序见 {ref}`VTI API 参考 <vti>`。

## 各向异性与稳定性

接口不强制 `epsilon >= delta`，也不会隐式截断参数。`epsilon < delta` 存在已知稳定性限制，减小 `dt` 不保证消除增长。显式 `max_vel` 必须覆盖各向异性速度包络，仅覆盖 `max(vp)` 不足以保证合法。

V17 源码通过二维 `[A,A,0,A]` 或三维 `[A,A,A,A,0,A]` 支持平顶，其余面均为同一正整数 A。在 `z=0` 将法向牵引投影为零；该处含 `sV` 的源被拒绝，包括默认源，表面 `sV` 接收为零。`sH` 源与速度接收仍遵循原生交错位置。这是声学闭合，不是弹性 VTI，也不保证表面达到内部差分阶数。原生 full/boundary 保留顶部条带，Torch full/checkpoint 采用相同闭合。投影后的材料链参与求导，但冻结扩边/裁剪梯度与 `epsilon<delta` 限制仍在。公开 7.1.0 不含此 V17 行为。

无 VTI 照明、公开重启状态或高阶自动微分。VTI 边界重构与梯度的目标 GPU 验收、完整正演示例仍为待完善项；同时适用[边界梯度限制](conventions.md)。
