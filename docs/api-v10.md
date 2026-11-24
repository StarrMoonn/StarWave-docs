# StarWave v10.0.0 API

本页对应 GitHub [v10.0.0 Release](https://github.com/StarrMoonn/StarWave/releases/tag/v10.0.0)，源码提交 `5cc7c5aa9d86e50cb366d2e2ea2b09e57404d079`，Python 包版本 `3.3.0`。这里重点说明当前接口容易混淆的轴序、采样及后端约定。[历史 Usage（V19 基线）](usage.md)。

## 安装与验收范围

从**私有 GitHub Release** 下载 `starwave-3.3.0-py3-none-manylinux_2_35_x86_64.whl`（需仓库权限），在已安装目标 PyTorch 的环境中执行：

```bash
python -m pip install ./starwave-3.3.0-py3-none-manylinux_2_35_x86_64.whl
```

该 wheel **未发布到 PyPI**，不要把不带路径的 `pip install starwave` 当作 v10 安装。发行目标为 Linux x86-64 / glibc ≥ 2.35，元数据支持 Python 3.10–3.12；本次实际安装与 CPU 正反传验收使用 Python 3.12、PyTorch 2.5.1+cpu 和 2.6.0+cpu。11 个 CPU 库和 11 个 CUDA 库均从源码构建并通过主机加载检查。CUDA 架构覆盖 sm70/75/80/86/89/90 和 compute80 PTX；本次没有运行 GPU 数值测试。GPU 传播仍需兼容驱动、GPU 和 CUDA 版 PyTorch。

## 网格间距、炮检坐标与吸收边界

`grid_spacing` 以米为单位。传标量表示各轴同间距；传序列则按**实际传给算子的模型轴序**排列。炮检坐标最后一维按相同轴序填写原模型上的整数网格索引，不是米；逐面 PML 每个轴依次填写低端、高端宽度，单位是网格点。不同算子对自由表面及边界宽度有独立限制。

| 算子 | 传入模型与间距 | 炮检位置最后一维 |
|---|---|---|
| Scalar2D、声学 VRR/VRZ | `[x,z]`，`(dx,dz)` | `[ix,iz]` |
| Scalar3D | `[x,y,z]`，`(dx,dy,dz)` | `[ix,iy,iz]` |
| VTI-VRR 2D | 默认 `[x,z]`，`(dx,dz)`；或 `[z,x]`，`(dz,dx)` 且显式 `depth_axis=0` | 分别为 `[ix,iz]` 或 `[iz,ix]` |
| GSLS 2D | `[z,x]` 配 `(dz,dx)`，或 `[x,z]` 配 `(dx,dz)`；没有 `depth_axis` 参数 | 跟随模型轴序 |
| Elastic 2D / 3D | `[y,x]` 配 `(dy,dx)`，或 `[z,y,x]` 配 `(dz,dy,dx)` | 跟随模型轴序 |

例如磁盘模型是 `vp_zx[z,x]`，横向 12.5 m、纵向 20 m：调用 Scalar2D 时传 `vp_zx.T`、`grid_spacing=(12.5,20.0)`，同时将原为 `[iz,ix]` 的炮检坐标翻转为 `[ix,iz]`。若仍用 `[z,x]` 模型调用 GSLS，则传 `(20.0,12.5)`，不用翻转已经是 `[iz,ix]` 的坐标。CPML 仍按原方法计算，非等距时两个方向的物理厚度与系数可以不同；更细网格可能改变 CFL 子步或触发固定时间步校验。

## Scalar2D：各向网格间距与梯度采样

`starwave.scalar(v, grid_spacing, dt, ...)` 的二维速度为 `[x,z]`。源振幅为 `[shot,source,time]`，炮检位置为 `[shot,source,2]` 和 `[shot,receiver,2]`；常规调用取 `outputs[-1]` 作为 `[shot,receiver,time]` 记录。以下假定输入均在 CPU，且坐标已是 `[ix,iz]`：

```python
import starwave

records = starwave.scalar(
    vp_zx.T, (12.5, 20.0), dt,
    source_amplitudes=source, source_locations=src_xz,
    receiver_locations=rec_xz, backend="native_cpu",
    memory="full", model_gradient_sampling_interval=1,
)[-1]
```

`model_gradient_sampling_interval` 默认 `1`，表示完整离散速度梯度。设为整数 `s>1` 时按**内部 CFL 时间步**取样：从零起每隔 `s` 步取一次，权重 `min(s, internal_nt-t)`，末组可不足 `s` 步。若内部步/用户步比为 `r`，`s=r` 即每个用户步取一次，算子不会再乘一次 `r`。正演、伴随传播、接收记录和源梯度仍使用所有内部步；原生 full 少存速度梯度历史，boundary 仍保存完整重建数据。稀疏梯度是近似，先与 `s=1` 比较；不能把它当作数据降采样。

## 声学 VRR / VRZ

`starwave.vrr` 接受速度与独立的 `rx, rz`（单位 1/m，允许有符号），按 `[x,z]` 排列。 `starwave.vrz` 从阻抗 `Z`（也可由速度与密度得到）通过匹配的中心差分构造 `R_i=-Z D_i(1/Z)/2`，再调用 VRR；离散上不以直接差分 `0.5 log(Z)` 替代。两个接口均返回单元素元组 `(records,)`。原生 CPU/CUDA 支持 full/boundary；显式 Torch 支持 full/checkpoint。顶面零宽的示例是 `pml_width=[w,w,0,w]`，其中 `w` 为正整数；不代表任意零宽面或任意不等吸收宽度均支持。

## VTI-VRR：模型转置、分量与时钟

`starwave.vti_vrr` 默认深度在末轴；将所有模型改成 `[z,x]` 后，同时调整 `depth_axis=0`、间距、炮检坐标与 PML 面顺序。若仍传兼容的 `dx`、`dz`，这两个名称是**物理方向**，不会因转置交换名称。

返回值为按 `receiver_fields` 排列的元组；默认 `("sV",)`，用 `sV, = ...` 取出单分量。`sH/sV` 是中心点拉应力（Pa），`vx/vz` 是对应交错面上的速度（m/s），没有自动插值到中心。读取速度须提供物理 `density`（kg/m³），它影响速度读出，不参与 VRR 传播参数化。输出对齐完步时刻 `(n+1)*dt`，与常规 VTI 的 `n*dt` 不同。CFL 必要时重采样；`cfl_resample=False` 时遇到过大步长会拒绝。

## GSLS：模型方向、固定时间步与材料

`starwave.visco_gsls(vp, q, rho, grid_spacing, dt, ...)` 返回 `(records,)`。PML 面宽按模型轴的低端、高端成对排列：`[z,x]` 的平顶零宽是 `[0,bottom,left,right]`，`[x,z]` 则为 `[left,right,0,bottom]`；其余面须为正。自由表面上源不能放在 `z=0`，该面的压力接收值为零。保持固定 `dt` 并检查 CFL，不会自动重采样。`rho` 为固定正密度；可微的材料参数遵守供给函数及固定速度包络规则。原生 CPU/CUDA 和 Torch 均支持 full/checkpoint，不支持 boundary。默认 `mode="hao1"` 使用内置材料模型时需显式给定合法 `frequency_band`（`(scale, 200*scale)`）；`q` 的物理含义随所选材料模式而变。

## Elastic：梯度采样使用用户步

Elastic 的同名 `model_gradient_sampling_interval=m` **不是 Scalar 的内部步语义**。`m` 必须为正整数；当需要弹性材料梯度时，必须整除用户采样点数 `user_nt`，原生计算使用间隔 `r*m`（`r` 是 CFL 子步比）。没有材料梯度时，有效间隔退回 `1`，包括只对震源或初始状态求导的情形。设置不会删减正演时间步。

## 完整接口

以下为精确发行源码的参考页，需要私有仓库访问权限。旧示例保留原来的历史版本与参数，不能代替本页契约。

- [完整 API（中文）](https://github.com/StarrMoonn/StarWave/blob/5cc7c5aa9d86e50cb366d2e2ea2b09e57404d079/docs/reference/API_REFERENCE_ZH.md)
- [VRR / VRZ](https://github.com/StarrMoonn/StarWave/blob/5cc7c5aa9d86e50cb366d2e2ea2b09e57404d079/docs/reference/VRR_API.md) · [VTI-VRR](https://github.com/StarrMoonn/StarWave/blob/5cc7c5aa9d86e50cb366d2e2ea2b09e57404d079/docs/reference/VTI_VRR_RESEARCH_API.md)
- [GSLS](https://github.com/StarrMoonn/StarWave/blob/5cc7c5aa9d86e50cb366d2e2ea2b09e57404d079/docs/reference/VISCO_GSLS_API_ZH.md) · [Elastic](https://github.com/StarrMoonn/StarWave/blob/5cc7c5aa9d86e50cb366d2e2ea2b09e57404d079/docs/reference/DEEPWAVE_ELASTIC_API_ZH.md)
