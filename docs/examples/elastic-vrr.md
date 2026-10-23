(elastic-vrr)=
# 弹性 VRR（V21）

本章说明 [Release V21](https://github.com/StarrMoonn/StarWave/releases/tag/V21) 新增的 `starwave.elastic_vrr` 接口，对应源码 `0.1.0.dev21`、提交 `fa1eab2ba64771d9df1823b009aad5e3d72a6aa4`。这是二维局部展开的动量/应力求解器，使用六个独立材料坐标，不是声学 `starwave.vrr`、普通 `starwave.elastic`，也没有新增 `elastic_vrz` 接口。其余参考页保留明确标注的 V19 基线；历史测试不重新标作 V21 结果。

## 安装匹配源码

请在 PyTorch 环境中使用完整、已获授权的 V21 源码及其依赖。源码仓库为私有，需要访问权限。本公开手册仅分发接口说明和合成示例，不分发求解器实现或模型数据。本章不宣称已有 V21 PyPI wheel；无版本限定的 `pip install starwave` 或旧 wheel 不能代替这份源码。

```bash
# 在已获授权的 V21 源码根目录执行：
python -m pip install --no-deps -e .
# 可选：编译原生 CPU
python compile_all.py --backend cpu --equation elastic_vrr
# 可选：编译原生 CUDA，架构按实际显卡选择
python compile_all.py --backend cuda --equation elastic_vrr --arch 80,89
```

安装或重新编译后重启 Python/Jupyter。Torch eager CPU 不需要原生传播库。统一 `all` 编译包含弹性 VRR；安装后的 `starwave-build` 接受相同参数。编译或导入成功不等于数值验证。

## 接口与坐标约定

```text
starwave.elastic_vrr(
    vp, vs, rp_x, rp_z, rs_x, rs_z, grid_spacing, dt, *,
    source_amplitudes, source_locations, receiver_locations,
    source_component="pz", receiver_components=("pz",),
    source_type="momentum_rate", density=None, accuracy=4,
    pml_width=20, pml_freq=25.0, max_vel=None,
    backend="torch", execution="eager", memory="full",
    checkpoint_interval=20, forward_callback=None, callback_frequency=1,
    return_state=False, free_surface=False, reflectivity_layout="edge",
    cfl_resample=True, return_time_sampling=False, compile_steps=1,
)
starwave.elastic_vrr_parameters(vp, vs, rho, grid_spacing, *, accuracy=4)
```

- 六个模型均为不含批次维的 `[nx, nz]`，各维至少 3，使用相同 CPU/CUDA 设备及 float32/float64 类型。数值必须有限，满足 `vp > 0`、`vs > 0` 和 `vp**2 > 4*vs**2/3`。不支持零 Vs 流体、MPS 或 autocast。
- `grid_spacing` 为正标量或两个正间距 `[dx, dz]`；`dt` 为用户采样间隔；`accuracy` 支持 2、4、6、8。
- 反射率单位为 1/m，采样在正向边：`rp_x[i,j]` 位于 `(i+1/2,j)`，`rp_z[i,j]` 位于 `(i,j+1/2)`，S 同理。仅支持 `reflectivity_layout="edge"`。每个方向最后一条法向边进入常值延拓，其值固定为零，该输入边不影响结果且梯度为零。
- `elastic_vrr_parameters` 返回 `(vp, vs, rp_x, rp_z, rs_x, rs_z)`，符号为 `R = +0.5 D_edge log(rho V)`。所有传播精度均使用最近邻正向差分，除以该方向网格间距，外侧边为零。`rho` 必须为正且类型/设备匹配。该辅助函数保留物理模型的自动微分链；六坐标独立优化时也可直接提供 R。
- 源形状为 `[shot, source, time]`，类型/设备与模型一致。源点和检波点坐标为相同设备上的整数 `[shot, point, 2]`，直接索引未加 PML 的物理 `[x,z]` 网格，不是米制位置。
- 动量率源使用 `px` 或 `pz`；应力率源使用 `sxx`、`szz` 或 `sxz`；速度率源需要正且匹配的 `density`。源振幅是率，不是预积分增量；每个完整应力/动量步之后注入，第一次记录在首次注入后。
- 按请求顺序返回各分量的 `[shot, receiver, time]` 张量元组。`px,pz` 是动量而非速度；`vx,vz` 接收需要 `density` 做单位标定，不强制独立 R 共享同一个密度。
- `return_state=True` 追加 13 个带 padding 的交错网格终态（五个动量/应力场及八个 CPML 历史场）的元组；`return_time_sampling=True` 在可选终态之后追加实际采样计划。

## 可运行 CPU 正演与梯度示例

完整示例将六个坐标设为独立优化叶子，没有切断波传播梯度，也没有跳过内部时间步。可下载相同的 {download}`Python 脚本 <../../examples/elastic_vrr_cpu.py>`。

```python
import starwave
import torch

# 2D local-expanded momentum/stress equation; model axes [x, z].
vp = torch.full((24, 24), 2500.0, dtype=torch.float64)
vs = torch.full_like(vp, 1200.0)
rho = torch.full_like(vp, 2000.0)
models = starwave.elastic_vrr_parameters(vp, vs, rho, 10.0)
# Independent six-coordinate optimization leaves (not physical-model AD).
models = tuple(x.detach().clone().requires_grad_() for x in models)
source = torch.zeros((1, 1, 32), dtype=vp.dtype)
source[0, 0, 0] = 1.0  # momentum rate, integrated by the solver
src = torch.tensor([[[12, 10]]], dtype=torch.long)
rec = torch.tensor([[[12, 10], [13, 10]]], dtype=torch.long)
(pz,) = starwave.elastic_vrr(
    *models, 10.0, 0.0002,
    source_amplitudes=source, source_locations=src, receiver_locations=rec,
    source_component="pz", receiver_components=("pz",),
    source_type="momentum_rate", reflectivity_layout="edge",
    accuracy=4, pml_width=4, pml_freq=25.0, max_vel=3000.0,
    backend="torch", execution="eager", memory="full",
)
assert pz.shape == (1, 2, 32)
pz.square().sum().backward()
assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in models)
assert torch.isfinite(pz).all()
print("Elastic VRR CPU forward/backward passed:", tuple(pz.shape))
```

若反演物理 Vp/Vs/rho，应让物理模型可训练，并保留辅助函数输出的计算图，不要 detach，否则会丢失密度/速度链式求导。支持源波形梯度；使用密度单位标定时，其梯度也可传播。冻结某个独立叶子应关闭其梯度、清理旧 `.grad`，并从优化器和投影中排除，而不是从耦合传播中移除该材料值。

## 原生与 Torch 执行

| 选择 | 设备 | 内存模式 | 执行方式 |
|---|---|---|---|
| `backend="torch"` | CPU/CUDA | `full`、`checkpoint` | `eager`、`compile` |
| `backend="cpu"` | CPU | `full`、`boundary` | `eager` |
| `backend="cuda"` | CUDA | `full`、`boundary` | `eager` |

运行原生 CPU 示例时先按上文编译，再把 `backend="torch"` 改为 `backend="cpu"`。CUDA 还需要在调用前把所有模型、源波形及位置张量移到 CUDA。不会静默替换内存模式。原生一阶导数覆盖六个模型、源和密度标定；不支持原生二阶导数。

Torch checkpoint 使用 `checkpoint_interval`，默认 20。编译模式设 `execution="compile"`；`compile_steps` 控制每块内部时间步数，默认 1，与 checkpoint 间距独立。原生只接受 `compile_steps=1`。编译耗时与热运行应分别测量；更大块不保证更快。

## 边界与数值范围

CPML 由 `pml_width`、`pml_freq` 及覆盖模型速度的 `max_vel` 设计；求导时应固定该设计上界。默认开启可微 CFL 重采样，所有内部步参与梯度；`cfl_resample=False` 则拒绝过大的时间步。该时间步筛查不等于任意独立反射率下的能量稳定性或收敛性证明。

`free_surface=True` 是平坦顶面的改进真空边界：物理 z=0 为真空，z=1 为首个固体层。z=0 仅允许 pz 源及 pz/vz 接收。不支持任意地形；仅把顶部 PML 设为零并不构成自由表面。真空材料梯度为零。

局部展开模板是特定离散选择。物理参数转换不保证在尖锐界面与普通弹性方程精确有限网格等价。这个小型合成正反传示例不代表长期稳定性、现场数据精度、GPU 性能或反演收敛验收。

## 正式反演示例

已获授权的 V21 源码 `notebook/Elastic_VRR/` 下提供 CUDA / TorchCompile 成对 notebook 和脚本：

- `Elastic2D_Marmousi2_VpVsRpRs_{CUDA,TorchCompile}`：六个独立 Vp/Vs/R 坐标，使用动量观测。
- `Elastic2D_Marmousi2_VpVsZpZs_VRR_{CUDA,TorchCompile}`：四个独立 Vp/Vs/Zp/Zs 坐标，在计算图内构造 `Rp=0.5 D_edge log Zp` 和 `Rs=0.5 D_edge log Zs`；`rho_aux=0.5*(Zp/Vp+Zs/Vs)` 仅标定速度单位，不强制共享物理密度。
- P-only 的 `VpZp` 与 `VpRp` 对照在固体裁剪区把 S 参数固定为真值，不增加水层（`vs=0`）支持。应保留各 notebook 的观测单位、初始化及目标函数，不直接比较不同参数化的原始目标值。

准确文件名和实验设置见授权源码的 `notebook/Elastic_VRR/README.md`。这些私有 notebook 及模型数据不在公开手册中镜像分发。
