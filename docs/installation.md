# 安装

[V19 Release](https://github.com/StarrMoonn/StarWave/releases/tag/V19) 已发布；本手册对应冻结源码 `0.1.0.dev19`、提交 `46d32877e3336d531520a696d5f2e074e6f4ef74`。[PyPI **9.1.0**](https://pypi.org/project/starwave/9.1.0/) 已发布，对应 V19 源码契约并包含集成 VRR。现有 [9.0.0 wheel](https://pypi.org/project/starwave/9.0.0/) 仍对应 V18，不含集成 VRR。

V19 将 {ref}`VRR <vrr>` 纳入主包，入口为 `starwave.vrr`，根目录安装与 `compile_all.py` 已覆盖 VRR，无需独立安装 `starwave_vrr`。VRR 显式 Torch 支持 CPU/CUDA float32/64、full/checkpoint 与 eager/compile；原生 CUDA 支持 float32 full/boundary。各方程的设备、梯度和边界限制仍独立。

V19 保留 V18 的 23 参数 scalar 契约：任意零宽面组合（正宽度仍须相等）、二维/三维可训练源、可微最终物理波场与诊断前向回调。终态可用于 loss，回调快照为 detached 观测，不能传递 loss 梯度。见 [Scalar Function](usage.md#scalar-function)。

## 安装 V19 源码

使用已获授权的完整 V19 源码，在目标 PyTorch 环境中安装源码要求的依赖，再从源码根目录执行：

```bash
python -m pip install --no-deps -e .
```

根目录注册包含 VRR，不会隐式编译原生库。完整源码也可不经注册直接运行。源码仓库仍为私有仓库，本手册不分发实现或模型数据。

显式 `backend="torch"` 支持 CPU/CUDA full/checkpoint，不依赖传播原生库或 nvcc；`execution="eager"` / `"compile"` 与模型设备分别选择。compile 使用当前 PyTorch Inductor 工具链，失败明确报错，不回退 eager。MPS 仅支持 Scalar2D accuracy=4 float32、Torch full/checkpoint 与 eager/compile；Born、Scalar3D、VRZ、VRR、VTI、elastic、GSLS 及其他 MPS 组合均拒绝。AMP 仍不支持。可从 {ref}`CPU 独立示例 <torch-example>` 开始。

## V19 原生构建

使用兼容的 CUDA Toolkit/C++ 工具链与完整匹配源码，显式构建五组原生库：

```bash
python compile_all.py --arch 80
```

该命令覆盖 core（scalar/VRZ/VTI）、elastic、GSLS、Born、VRR。A30 使用 SM `80`，RTX 4060 使用 `89`，或选择实际 GPU 架构。较早的 `combined_build.py` 只覆盖 core/elastic。可选的独立构建如下：

```bash
# Born 原生 CPU full：
python compile_born.py --cpu-only
# GSLS CUDA 或原生 CPU：
python compile_visco_gsls.py --arch 80
python compile_visco_gsls.py --backend cpu
# 仅 VRR CUDA：
python -m starwave._vrr.build --arch 80
```

VRR 无需另装包。其原生构建使用 nvcc/C++17，不依赖 Torch C++ 头文件或 libtorch 链接；GPU 运行仍需支持 CUDA 的 Torch 以及兼容驱动/运行库。纯 Torch Born、VRR 无需原生构建。

`import starwave` 返回前绑定已有原生库和共享 dispatch，包含集成 VRR；不编译、不初始化 CUDA、不选设备、不分配波场，也不传播。未构建的库允许受支持的显式 Torch 使用，加载/绑定错误仍会报错。导入不执行运行时源码/库哈希、指纹、ABI/capability 版本或产物身份审计。源码变更后须重建匹配库并重启 Python/Jupyter；手动 `prepare_*` 仍为可选设备检查。不能混用旧库与新源码；编译或导入成功不等于 GPU 数值验收。

GSLS 输出为 `native/build/gsls/`，或 `STARWAVE_BUILD_DIR` 的 `gsls/` 子目录。导入前设置构建路径，构建和运行使用同一路径。既有教程、图件与实测结果保留历史版本。

(runtime-requirements)=
## 已发布 V19 二进制渠道

以下安装已发布的 V19 / 9.1.0，包含集成 VRR、Born 和 23 参数 scalar。Mac MPS 必须使用上方完整源码安装；此 Linux wheel 不能在 macOS 安装。源码接受 compile 模式，不保证每个 PyTorch 版本或设备均有可用的编译器。

**安装已发布 V19 / 9.1.0 wheel**

要求 Linux x86_64 / glibc ≥ 2.35、Python ≥ 3.10 且 < 3.13、PyTorch ≥ 2.5 且 < 2.6、NumPy ≥ 1.23。没有 macOS wheel 或 sdist。预编译安装不需要 nvcc 或本地 CUDA Toolkit；原生 CUDA 运行仍需支持的 NVIDIA 驱动与设备。下面使用发布核验采用的 PyTorch 2.5.1 / CUDA 11.8 配置；仅用 CPU 时可选择配套的 PyTorch 2.5.x CPU 构建。这不扩大后端或设备的支持范围。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --only-binary=starwave starwave==9.1.0
python -m pip check
```

升级后重启 Python/Jupyter。公开 wheel 已通过 Python 3.10–3.12 安装与 smoke 检查；目标 GPU 数值验收另行记录。旧 wheel 条件与迁移命令已归档至[发布说明](release-notes.md)。

## 核心库检查（已安装 wheel 或已注册源码）

```python
import torch
import starwave
from importlib.metadata import version

print(version("starwave"))
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available())
status = starwave.native_status()
print({key: status[key] for key in ("library_exists", "library_loaded")})
# V19 可选的显式设备检查：
starwave.prepare_native([0])
```

V19 的 `import starwave` 在返回前准备已有 core/elastic/Born/GSLS 原生库与共享 dispatch；集成 VRR 也会准备已有的原生库。导入不编译、不初始化 CUDA、不选设备、不分配波场，也不传播。导入前配置构建路径，更新后重建并重启。未构建的库允许跳过并使用受支持的显式 Torch；保留的无效构建会报错。手动 prepare 是可选设备检查，不再要求串行预热。导入或 prepare 均不能代替正演/梯度数值检查，接着运行[快速入门](quickstart.md)。

文档站本身的构建环境独立于计算环境；仅阅读或构建文档不需要 GPU。


弹性传播保留 V19 可选的独立准备入口（较早原生工作流仍需准备）：GPU 主线程调用 `starwave.prepare_elastic([0])`，CPU 调用 `starwave.prepare_elastic()` 并选择 `memory="full"`。完整参数与 vp/vs/rho 转换见 {ref}`Elastic Function <elastic>`。原生 scalar3D 没有 CPU 路径；Torch scalar3D 可在 CPU/CUDA 上运行。GPU、DataParallel 与长程 FWI 验收范围见[文档状态](status.md)。

## GSLS 准备与后端选择

GSLS 保留 V19 可选的独立准备入口（较早原生工作流仍需准备）：CUDA 主线程调用 `starwave.prepare_visco_gsls(backend="cuda", device="cuda:0")`，CPU 调用 `starwave.prepare_visco_gsls(backend="native_cpu")`。`starwave.visco_gsls_native_status(backend="cuda")` 只查看状态。准备默认 CPU，传播默认 CUDA，建议始终显式指定；不自动回退。`backend="torch"` 支持 full/checkpoint 与 eager/compile，无需原生构建。见 {ref}`GSLS API <visco-gsls>` 的完整契约与 CPU 示例。

(installation-smoke)=
## Notebook 01：实际运行检查

下载 {download}`01_installation_environment_smoke.ipynb <../examples/tutorials/01_installation_environment_smoke.ipynb>`，或 {download}`配套 Python 脚本 <../examples/tutorials/01_installation_environment_smoke.py>`。Notebook 还需要 NumPy、Matplotlib ≥ 3.6 和 Jupyter；在选定环境的独立终端安装，不在运行中的 kernel 内升级软件：

```bash
python -m pip install numpy "matplotlib>=3.6" jupyterlab ipykernel
python -m jupyter lab
```

选择对应的 CUDA kernel，从头运行全部单元。每份下载均是独立、清空输出的公开版；保留原始中文教学说明。数值实验未改动，环境来源字段及私有安装说明的整理范围见{ref}`来源记录 <tutorial-evidence>`。

```{admonition} 实测环境 · 2026-10-05
:class: sw-run-note
NVIDIA A30 · Python 3.10.18 · PyTorch 2.5.1 / CUDA 11.8 · 安装的 StarWave 版本 **0.1.0.dev9**。以下服务器结果验证的是这个开发版环境的小规模流程，不能据此宣称公开 **2.0.0、4.0.0、5.0.0、6.0.0、7.0.0 或 7.1.0 wheel** 已通过实机验收。
```

## 小规模实验设计

均匀速度 1800 m/s，模型 48 × 40、间距 10 m；一炮位于 (240, 50) m，12 个接收点位于 x=80…410 m、间隔 30 m、z=50 m。15 Hz Ricker 峰值时刻 0.055 s，dt=1 ms、160 个样点，最后样点在 0.159 s。

配置为 float32、accuracy=4、PML=12 格、boundary_buffer=5、memory="boundary"，固定 max_vel=2200 m/s。该运行的时间规划器返回内部步长 1 ms、子步比 1。四侧均使用 PML，没有自由表面。参数区为 `active[4:-4,8:-4]`，浅部和外缘保持固定。

```{figure} /_static/tutorials/installation_model.png
:alt: 1800 m/s 均匀模型、一炮十二道采集和固定边带，坐标以米表示
:class: sw-science-image
:figclass: sw-science-figure

最小实验的模型与几何。固定的源 forcing 由传播器施加内部注入系数，无需手动重复乘以 −v² dt²。
```

先执行一次 `scalar`，再取 `objective = record.square().mean()` 并执行 `objective.backward()`。这个能量型标量只用于检查反传接线，不是 FWI 数据失配或物理解的误差基准。

## 已得到的记录与检查结果

```{figure} /_static/tutorials/installation_gather.png
:alt: 一炮十二道实测合成记录，时间以秒、接收位置以米表示，振幅为任意单位
:class: sw-science-image
:figclass: sw-science-figure

服务器导出的炮集，采用完整零对称振幅范围；不截幅或逐道归一化，幅度单位不是校准后的 Pa。
```

| 检查项 | 实测结果 |
|---|---:|
| 记录形状 `[shot,receiver,time]` | `[1,12,160]` |
| 最大绝对振幅 / RMS | 25.1871452 / 4.5169382 |
| 记录平方均值 objective | 20.4027290 |
| 速度梯度 L2 范数 | 0.0019382348 |
| 活动区最大绝对梯度 | 0.0010059283 |
| 固定区最大绝对梯度 | 0 |

记录和梯度均为有限值，活动区梯度非零。**这是一项通过的小规模 smoke check，不是梯度数值正确性、FWI 收敛或多 GPU 验证。** 原生库准备成功、编译架构覆盖和上述数值实验属于不同证据层级。

继续运行[简单梯度计算](modeling/gradient.md)与[简单 FWI 示例](inversion/fwi.md)，或从[快速入门](quickstart.md)下载更短的命令行接线脚本。完整实验设置与精确数值见 {download}`结果摘要 <../examples/tutorials/results_summary.json>`。
