# StarWave 使用手册

StarWave 为 PyTorch 提供 CUDA 波传播接口，覆盖二维标量声学、二维 VRZ 与二维/三维声学 VTI。本手册面向使用公开 **2.0.0 wheel** 的用户，从环境安装、小规模正演开始，再介绍梯度与反演接线。

```{important}
本手册的 API 对应公开 2.0.0 wheel。三份原创教程已有 NVIDIA A30 上 StarWave 0.1.0.dev9 的实际运行记录，涵盖小规模 smoke check、梯度与 25 次 FWI 更新；这不等于公开 wheel、完整伴随、收敛或性能验收。请按各页标注的版本和证据范围阅读。
```

第一次使用：阅读[安装](installation.md) → [快速入门](quickstart.md) → [模型与采集约定](modeling/conventions.md)。Windows 用户从 [WSL 2](wsl.md) 开始；查看实际实验：{ref}`安装运行检查 <installation-smoke>` → [简单梯度计算](modeling/gradient.md) → [简单 FWI 示例](inversion/fwi.md)。

| 接口 | 模型与维数 | 默认记录 | 主要区别 |
|---|---|---|---|
| `starwave.scalar` | `v`，2D | pressure-like | 等间距；标量声学 |
| `starwave.vrz` | `v` 与 `impedance` 或 `density`，2D | pressure-like | 必须恰好选择一种介质参数 |
| `starwave.vti` | `vp, epsilon, delta, rho`，2D/3D | `vz` | 声学 VTI；支持各轴不同网格间距 |

```{toctree}
:maxdepth: 1
:caption: 开始使用

installation
wsl
quickstart
```

```{toctree}
:maxdepth: 1
:caption: 正演建模

modeling/conventions
modeling/scalar
modeling/gradient
modeling/vrz
modeling/vti
```

```{toctree}
:maxdepth: 1
:caption: 反演与神经网络

inversion/fwi
inversion/dataparallel
inversion/inr
```

```{toctree}
:maxdepth: 1
:caption: Usage

usage
```

```{toctree}
:maxdepth: 1
:caption: 参考与维护

faq
release-notes
status
```

导航结构参考 [PyFWI 使用手册](https://pyfwi.readthedocs.io/en/latest/) 的入门、正演、反演分类；本手册内容针对 StarWave 编写，不表示两者接口或数值实现相同。
