# 模型、采集与时间约定

本页未注明时描述原生默认路径。V16 新增显式 Torch CPU/CUDA full/checkpoint，边界、设备和限制见 [PyTorch 后端](../pytorch-backend.md)。

## 张量形状

`B` 是炮数，`S` 是每炮震源数，`R` 是每炮接收点数，`T` 是用户时间采样数，`D` 是空间维数。

| 输入/输出 | 形状 | 约定 |
|---|---|---|
| scalar 模型 | `[N0,N1]` 或 `[N0,N1,N2]` | CUDA float32；由 `v.ndim` 选择维数 |
| VRZ 模型 | `[N0,N1]` | CUDA float32 |
| VTI 模型 | `[nx,nz]` 或 `[nx,ny,nz]` | 竖直轴固定为最后一轴 |
| `source_amplitudes` | `[B,S,T]` | 三维 scalar 支持一阶源梯度；二维 scalar、VRZ、VTI 源固定 |
| `source_locations` | `[B,S,D]` | 整数网格下标，建议 torch.long |
| `receiver_locations` | `[B,R,D]` | 整数网格下标 |
| 每个记录张量 | `[B,R,T]` | CUDA float32 |

坐标直接索引传入的物理模型，**不是米坐标，不含 PML 偏移**。scalar / VRZ 不会自动推断用户的物理轴序；三维 scalar 坐标 `[i,j,k]` 对应 `v[i,j,k]`。若模型存为 `[nz,nx]`，采集坐标使用 `[x,z]`，应先显式转置。VTI 的四个模型必须使用相同轴序、形状、设备和 dtype。

## 采样与吸收边界

`grid_spacing` 和 `dt` 为正数，典型单位是米和秒。内部会按 CFL 规划更小时间步，但返回长度仍为 `T`，名义用户时间为 `0, dt, ..., (T-1)dt`。二维 scalar / VRZ 只支持等间距；三维 scalar 和 VTI 可按模型轴序逐轴给出间距。

`pml_freq` 用于 PML 设置及空间采样诊断，不是自动生成震源波形的函数。`pml_width` 是每个面相等的正整数网格厚度；不支持通过设为零来请求自由表面。

`boundary_buffer` 是空间网格数，默认 5。boundary 模式要求至少 `accuracy // 2 + 1`；full 模式允许 0，但仍保留配置的边界填充。

## 内存与自动微分

本页的 scalar、VRZ、VTI 均支持 `memory="boundary"`（默认）或 `"full"`。full 保存时间历史，可能占用更多显存；没有自动切换模式。三维 scalar 的 Radius-M 六面历史、终态与工作区见{ref}`重建说明 <reconstruction-scalar3d>`；源单独求导同样保存历史。每次 forward 只支持一次一阶 backward。CPU 传播、AMP、高阶自动微分、自定义 CUDA stream、公开 checkpoint/restart 状态接口均不在当前支持范围。

```{warning}
模型梯度以固定扩展模型、CFL 与 PML 设置为条件。物理边缘的 replicate padding 在前向重新生成时，返回梯度不保证包含完整扩展链的转置累积。切换 full 模式不能修复这一限制，不能把梯度描述为整个重建边界过程的全导数。
```

更具体的源单位、记录分量和稳定性限制见各方程页面。
