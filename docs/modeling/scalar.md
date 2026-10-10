# 标量声学 scalar

本页的 CUDA 设备、full/boundary 历史布局与原生构建说明描述原生默认路径。V16 显式 Torch 路径支持 CPU/CUDA full/checkpoint；见 [PyTorch 后端](../pytorch-backend.md)。

`starwave.scalar` 在公开 **6.0.0** 中接受二维或三维速度模型 `v`，由 `v.ndim` 自动选择，返回单元素元组 `(receiver_amplitudes,)`。使用 `[0]` 取得形状 `[B,R,T]` 的 pressure-like 记录。

## 模型、坐标与采样

模型是 CUDA float32，每个维度至少为 2。坐标 `[i,j]` / `[i,j,k]` 直接索引输入模型 `v[i,j]` / `v[i,j,k]`，不含 PML 偏移。采用 `[nx,ny,nz]` 模型时，三维间距按 `[dx,dy,dz]` 给出；不会自动交换物理轴。二维只支持等间距，三维支持逐轴不等间距。

模型允许有限正值、零值和负值，但这不是对所有值的物理合理性背书；常规速度模型通常选择正值。CFL 使用 `max(abs(v))` 与全部轴间距规划内部子步；全零模型必须显式给出正的 `max_vel`。输出仍为 T 个用户采样点，间隔为 `dt`。

支持 `accuracy=2,4,6,8`。`pml_width` 为正整数格数，二维四面或三维六面必须相等。默认 `boundary_buffer=5`；boundary 模式要求至少 `accuracy//2+1`。零 PML、非对称 PML 和自由表面不在本接口范围。

## 震源与调用

公开震源是归一化 forcing，满足 `u_tt = v² (Lap(u) - f)`。二维提供固定 `f`，三维还可对源求一阶梯度；不要额外手动乘以 `-v² dt²`。如果 `u` 以 Pa、长度以 m 计，`f` 对应 Pa/m²。

```python
records = starwave.scalar(
    v, grid_spacing=10.0, dt=0.001,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
)[0]
```

此片段依赖已准备的输入。二维独立程序见[快速入门](../quickstart.md)，三维速度与源联合求导的独立程序见{ref}`三维调用示例 <scalar-3d-example>`；完整参数见 {ref}`scalar API 参考 <scalar>`。文档维护没有执行这些 GPU 示例。

## 梯度与照明

设 `v.requires_grad=True`，从记录构造标量 loss 后调用一次 `backward()`。源位置速度因子也属于模型梯度链。二维源仍不可训练；三维支持单独或同时请求速度与源梯度。可选 `ScalarIllumination` 仅支持二维 scalar，三维必须传 `None`。照明统计量不能称为精确 Hessian。

二维在 6.0.0 中使用方向紧凑 PML 状态，boundary 保存宽度 `M=accuracy//2` 的压力条带与两个终态压力场；普通目标反传（`illumination=None`）的 full 历史移除 PML/差分 halo、保留 `boundary_buffer`，每轴长度为原模型长度加 `2*boundary_buffer`；启用可选照明收集器时保留完整布局。省去填充区历史不等于省去传播、伴随或重建工作场。实际峰值显存仍取决于模型、PML、炮数、内部时间步和计算图；full/boundary 的公开选择与默认值不变。PML 转置修复不改变下述模型扩边梯度限制。

默认 `memory="boundary"`。三维保存六个宽度为 `M=accuracy//2` 的压力面和两个终态压力场，用于反向重建；`"full"` 保存逐内部步的完整填充体积 `Lap(u)` 历史。源单独求导也会保存历史。CPML 使用方向条带，但传播、伴随和其它工作区仍占显存，没有自动降级或 CPU/磁盘卸载。内存估计见{ref}`Scalar3D 重建 <reconstruction-scalar3d>`。

减小时间步不能解决空间采样不足。进行反演前需单独检验波形、空间分辨率、梯度与[边界梯度限制](conventions.md)。full/boundary 都采用原有的 replicate-forward/crop-backward 梯度约定，不能把它描述为完整模型扩边链的全导数。
