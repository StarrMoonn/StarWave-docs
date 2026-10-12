# 标量声学 scalar

本章当前契约对应冻结 V19。完整参数与后端限制见 [Usage](../usage.md)，发布渠道见[安装](../installation.md)。历史推导与实测例子保留其注明版本。

本页的 CUDA 设备、full/boundary 历史布局与原生构建说明描述原生默认路径。V19 显式 Torch 路径支持 CPU/CUDA full/checkpoint；见 [PyTorch 后端](../pytorch-backend.md)。

V19 scalar 支持任意零面组合（正宽度相等）、二维/三维可训练源、可选可微最终物理波场及诊断前向回调。23 参数完整契约见 [Scalar Function](../usage.md#scalar-function)。默认返回 `(records,)`；`return_wavefield=True` 在记录前增加终态，兼容两种形式时用 `[-1]` 取记录。

## 模型、坐标与采样

模型是 CUDA float32，每个维度至少为 2。坐标 `[i,j]` / `[i,j,k]` 直接索引输入模型 `v[i,j]` / `v[i,j,k]`，不含 PML 偏移。采用 `[nx,ny,nz]` 模型时，三维间距按 `[dx,dy,dz]` 给出；不会自动交换物理轴。二维只支持等间距，三维支持逐轴不等间距。

模型允许有限正值、零值和负值，但这不是对所有值的物理合理性背书；常规速度模型通常选择正值。CFL 使用 `max(abs(v))` 与全部轴间距规划内部子步；全零模型必须显式给出正的 `max_vel`。输出仍为 T 个用户采样点，间隔为 `dt`。

支持 `accuracy=2,4,6,8`。V19 scalar 接受任意零面组合或 `pml_width=0`，所有正面宽仍须相等。面顺序按公开模型轴低/高端排列。零面仅含 `M=accuracy//2` 个固定零 ghost，不加 PML/buffer，物理边缘样点仍活动。二阶等效零平面在边缘外一格，高阶为截断模板而非奇镜像。即使全零面，boundary 的 buffer 最小值仍为 `accuracy//2+1`。冻结 V17 仅支持最后一轴顶部零宽；7.1.0 既不含 dev 扩展，也不含 V17 顶部零宽功能。

## 震源与调用

公开震源是归一化 forcing，满足 `u_tt = v² (Lap(u) - f)`。V19 二维和三维均可对源求一阶梯度，历史 V17 及更早二维源须固定；不要额外手动乘以 `-v² dt²`。如果 `u` 以 Pa、长度以 m 计，`f` 对应 Pa/m²。

```python
records = starwave.scalar(
    v, grid_spacing=10.0, dt=0.001,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
)[0]
```

此片段依赖已准备的输入。二维独立程序见[快速入门](../quickstart.md)，三维速度与源联合求导的独立程序见{ref}`三维调用示例 <scalar-3d-example>`；完整参数见 {ref}`scalar API 参考 <scalar>`。文档维护没有执行这些 GPU 示例。

可选最终物理波场可微，但缺少前一时间层与 CPML 状态，不能单独重启传播。必填 receiver_locations 可为空 `[B,0,D]`。前向回调在用户步间隔及最后一次更新后提供独立 detached 快照，秒时间使用 state.time；CFL 细分时不能把 step*dt 当作物理时间。回调过程中修改快照会报错，checkpoint 反传不重放回调。Usage 提供可运行 CPU 示例；完整目标波场示例在授权V19 源码的 `notebook/Scalar/Scalar2D_TargetWavefield_CUDA.ipynb`。

## 梯度与照明

设 `v.requires_grad=True`，从记录构造标量 loss 后调用一次 `backward()`。源位置速度因子也属于模型梯度链。V19 二维/三维均支持单独或同时请求速度与源梯度，loss 可来自记录、终态或二者之和。可选 `ScalarIllumination` 仅支持二维 scalar，三维必须传 `None`。任意零面 scalar 与 Torch 均拒绝照明；照明统计量不能称为精确 Hessian。

6.0.0 引入的二维全正 PML 布局使用方向紧凑 PML 状态，boundary 保存宽度 `M=accuracy//2` 的压力条带与两个终态压力场；普通目标反传（`illumination=None`）的 full 历史移除 PML/差分 halo、保留 `boundary_buffer`，每轴长度为原模型长度加 `2*boundary_buffer`；启用可选照明收集器时保留完整布局。省去填充区历史不等于省去传播、伴随或重建工作场。实际峰值显存仍取决于模型、PML、炮数、内部时间步和计算图；full/boundary 的公开选择与默认值不变。PML 转置修复不改变下述模型扩边梯度限制。

V19 Scalar2D 存在任意零面时，原生 full 使用完整网格历史；原生 boundary 保留必要面条带，不回退 full 历史。二者均保留冻结扩边模型梯度限制。

默认 `memory="boundary"`。三维保存六个宽度为 `M=accuracy//2` 的压力面和两个终态压力场，用于反向重建；`"full"` 保存逐内部步的完整填充体积 `Lap(u)` 历史。源单独求导也会保存历史。CPML 使用方向条带，但传播、伴随和其它工作区仍占显存，没有自动降级或 CPU/磁盘卸载。内存估计见{ref}`Scalar3D 重建 <reconstruction-scalar3d>`。

线性化散射见 [Born Function](../usage.md#born-function)：`scatter` 是速度扰动 δv，其背景算子与本 scalar 算子不同。应使用 Born 返回的背景记录构造兼容观测；Born 的一维/二维/三维轴序、dtype、状态、梯度与零面契约均独立。

减小时间步不能解决空间采样不足。进行反演前需单独检验波形、空间分辨率、梯度与[边界梯度限制](conventions.md)。full/boundary 都采用原有的 replicate-forward/crop-backward 梯度约定，不能把它描述为完整模型扩边链的全导数。
