# 模型、采集与时间约定

未特别注明时，下方张量表与 scalar/VRZ/VTI 限制描述原生默认路径。V17 源码新增方程专属平顶边界与 Born；公开 7.1.0 仍是较早的 V16 wheel，8.0.0 尚未发布。显式 Torch CPU/CUDA full/checkpoint 有独立的设备与功能限制，见 [PyTorch 后端](../pytorch-backend.md)。

## 张量形状

`B` 是炮数，`S` 是每炮震源数，`R` 是每炮接收点数，`T` 是用户时间采样数，`D` 是空间维数。

| 输入/输出 | 形状 | 约定 |
|---|---|---|
| scalar 模型 | `[N0,N1]` 或 `[N0,N1,N2]` | CUDA float32；由 `v.ndim` 选择维数 |
| VRZ 模型 | `[N0,N1]` | CUDA float32 |
| VTI 模型 | `[nx,nz]` 或 `[nx,ny,nz]` | 竖直轴固定为最后一轴 |
| `source_amplitudes` | `[B,S,T]` | 当前 dev scalar 二维/三维均支持源梯度；发布版二维 scalar 与 VRZ/VTI 源固定 |
| `source_locations` | `[B,S,D]` | 整数网格下标，建议 torch.long |
| `receiver_locations` | `[B,R,D]` | 整数网格下标 |
| 每个记录张量 | `[B,R,T]` | CUDA float32 |

坐标直接索引传入的物理模型，**不是米坐标，不含 PML 偏移**。scalar / VRZ 不会自动推断用户的物理轴序；三维 scalar 坐标 `[i,j,k]` 对应 `v[i,j,k]`。若模型存为 `[nz,nx]`，采集坐标使用 `[x,z]`，应先显式转置。VTI 的四个模型必须使用相同轴序、形状、设备和 dtype。

## 采样与吸收边界

`grid_spacing` 和 `dt` 为正数，典型单位是米和秒。内部会按 CFL 规划更小时间步，但返回长度仍为 `T`，名义用户时间为 `0, dt, ..., (T-1)dt`。二维 scalar / VRZ 只支持等间距；三维 scalar 和 VTI 可按模型轴序逐轴给出间距。

`pml_freq` 用于 PML 设置与空间采样诊断，不生成震源。V17 的 `pml_width` 与表面几何依方程区分：

- 仅当前 dev scalar：二维四面/三维六面的任意组合可为零，包括全零；所有正宽度仍须相等。每个零面仅保留固定零外部 ghost，物理样点活动，不含 PML/buffer，也不宣称高阶奇反射。两种维度均支持源梯度；可选终态 `[B,*v.shape]` 对应 `T*dt`，记录末样点为 `(T-1)*dt`；支持空接收几何 `[B,0,D]`。它不扩展下方发布 V17 或 VRZ/VTI 契约。
- 冻结 V17 Scalar2D/3D 与 VRZ2D：正整数或逐面相等的正宽度；只允许最后一模型轴的低侧面为零，其余宽度必须是同一正整数 A。`[x,z]` 使用 `[A,A,0,A]`，`[x,y,z]` 使用 `[A,A,A,A,0,A]`。外部 `accuracy//2` 格 halo 固定为零，首个物理行仍活动，顶部不加 PML/buffer。它是截断模板，不是奇反射；二阶的等效压力释放面在首个采样上方一格，不据此宣称高阶表面精度。scalar 顶部零宽时拒绝 illumination。
- 独立 VRR2D：轴序 `[x,z]`，接受正标量/等正逐面宽度或 `[A,A,0,A]`；独立扩展契约见 [VRR Function](../usage.md#vrr-function)。
- VTI2D/3D：与 scalar 相同的最后一轴顶部零宽排列，其余宽度等正，但采用投影的零法向牵引闭合。`z=0` 含 `sV` 的源被拒绝，包括默认源；该处 `sV` 接收为零。它是声学闭合，不是弹性 VTI。
- Elastic2D/3D：第一模型轴竖直（`[y,x]` / `[z,y,x]`）。`[0,A,B,C]` / `[0,A,B,C,D,E]` 选择改进真空准备，其余宽度为正且可不同。第 0 行固定真空，第 1 行为首个固体应力行。顶部零宽时拒绝传入初始状态、`origin` 与 `survey_pad`，不新增坐标偏移。
- GSLS2D：轴序 `[z,x]`；正标量或正 `(z,x)` 对表示对称吸收边界。`[0,bottom,left,right]` 选择压力释放，其他宽度为正且可不同：奇压力/偶竖直面闭合、表面压力为零，压力源必须 `z>=1`。其余零宽排列和全正四面列表被拒绝。
- Born1D/2D/3D：轴序 `[x]`、`[y,x]`、`[z,y,x]`；可用非负标量或 `2*ndim` 个逐面宽度。零面保留继承的固定零 FD ghost，物理格点仍活动。这一较宽取值域不能套给 `starwave.scalar`，也不代表奇镜像高阶表面精度或弹性无牵引物理。

7.1.0 wheel 不含上述源码变更。各传播器完整源、接收、后端与导数限制见 [Usage](../usage.md)。

`boundary_buffer` 是空间网格数，默认 5。boundary 模式要求至少 `accuracy // 2 + 1`；full 模式允许 0，但仍保留配置的边界填充。

## 内存与自动微分

本页的 scalar、VRZ、VTI 均支持 `memory="boundary"`（默认）或 `"full"`。full 保存时间历史，可能占用更多显存；没有自动切换模式。三维 scalar 的 Radius-M 六面历史、终态与工作区见{ref}`重建说明 <reconstruction-scalar3d>`；源单独求导同样保存历史。每次 forward 只支持一次一阶 backward。这些原生 scalar/VRZ/VTI 路径不支持 CPU 传播、AMP、高阶自动微分、自定义 CUDA stream 或公开重启状态。显式 Torch 可在 CPU 使用 full/checkpoint，Born/elastic 则有自己的终态返回契约；这些限制不代表所有 StarWave 入口。

```{warning}
模型梯度以固定扩展模型、CFL 与 PML 设置为条件。物理边缘的 replicate padding 在前向重新生成时，返回梯度不保证包含完整扩展链的转置累积。切换 full 模式不能修复这一限制，不能把梯度描述为整个重建边界过程的全导数。
```

更具体的源单位、记录分量和稳定性限制见各方程页面。
