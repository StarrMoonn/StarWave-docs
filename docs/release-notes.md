# 发布说明

## V19 / 0.1.0.dev19 · 2026-10-12

[V19 Release](https://github.com/StarrMoonn/StarWave/releases/tag/V19) 已发布；本手册对应冻结源码 `0.1.0.dev19`、提交 `46d32877e3336d531520a696d5f2e074e6f4ef74`。[PyPI **9.1.0**](https://pypi.org/project/starwave/9.1.0/) 已发布，对应 V19 源码契约并包含集成 VRR。现有 [9.0.0 wheel](https://pypi.org/project/starwave/9.0.0/) 仍对应 V18，不含集成 VRR。

公开 wheel：`starwave-9.1.0-py3-none-manylinux_2_35_x86_64.whl`，29,014,423 字节，上传时间 2026-10-12 06:05:56 UTC。SHA-256：`07e9bfe00425e8a28b67805c1e300a05f9f60a4d9b963fa7d4f7b1c0425d0df7`。公开 PyPI 元数据与已审产物一致。打包和 CPU 检查不代表 GPU、多卡、MPS、长程 FWI 或性能验收。

V19 将 {ref}`VRR <vrr>` 纳入主包，入口为 `starwave.vrr`，根目录安装与 `compile_all.py` 已覆盖 VRR，无需独立安装 `starwave_vrr`。VRR 显式 Torch 支持 CPU/CUDA float32/64、full/checkpoint 与 eager/compile；原生 CUDA 支持 float32 full/boundary。各方程的设备、梯度和边界限制仍独立。

V19 保留 V18 的 23 参数 scalar 契约：任意零宽面组合（正宽度仍须相等）、二维/三维可训练源、可微最终物理波场与诊断前向回调。终态可用于 loss，回调快照为 detached 观测，不能传递 loss 梯度。见 [Scalar Function](usage.md#scalar-function)。

V19 冻结已审阅的集成基线，不改变科学内核或公开签名。根目录安装/构建覆盖 core、elastic、GSLS、Born、VRR 五组原生库。正式示例按方程组织，采用 CUDA/TorchCompile 脚本与 notebook 配对，并包含示例启动及 INR memory 传递修复。源码发布不等于 9.1.0 wheel 身份核验、GPU/多卡、MPS、长程 FWI 收敛或性能验收；下列历史图件、实测教程与哈希保留原证据范围。


## V18 / PyPI 9.0.0 · 2026-10-12

V18 scalar（审阅基于 dev `e040aa9b8225ab387f7e4853bfe6162f5351f5d5`） 新增任意零面组合（正宽度仍须相等）、二维/三维可训练源、可选可微最终物理波场与诊断前向回调。这些源码功能，不在冻结 V17/其 8.0.0 发布基线或 7.1.0 中。23 参数契约见 [Scalar Function](usage.md#scalar-function)，较早的 20 参数发布快照仍单独固定。

[Release V18](https://github.com/StarrMoonn/StarWave/releases/tag/V18) 已发布，冻结提交为 `fba1f496b890f2dbaf3fbb3011d39f6d723cfea1`，保留已审 dev `e040aa9` 的 scalar API。[PyPI 9.0.0](https://pypi.org/project/starwave/9.0.0/) 已发布，受支持的 Linux x86_64 主机可安装包含这些功能的 wheel，条件和命令见[安装](installation.md)。V17 与未发布的 8.0.0 基线保持不变；7.1.0 为历史 V16 wheel。

公开文件为 `starwave-9.0.0-py3-none-manylinux_2_35_x86_64.whl`（27,975,674 字节），2026-10-12 上传，SHA-256 为 `d290ea0a7fdc2a74229d989525aa2bff1cdc9d675fb715bd6f43a6e17b1ddd0b`。没有 sdist 或 macOS wheel。要求 Linux x86_64 / glibc ≥ 2.35、Python ≥ 3.10 且 < 3.13、PyTorch ≥ 2.5 且 < 2.6、NumPy ≥ 1.23。公开下载文件已逐字节核对哈希。[发布工作流](https://github.com/StarrMoonn/StarWave/actions/runs/38199707015) 八项作业全部通过，包括 Python 3.10、3.11、3.12 的公开 PyPI 干净下载、安装与 smoke 检查。这些是打包和主机/CPU 检查，不代表公开 wheel 的 GPU、多卡、长程 FWI 或性能验收。Mac MPS 仍需源码，且只限已说明的 Scalar2D 配置。

## V17 源码、Born 与平顶边界

[Release V17](https://github.com/StarrMoonn/StarWave/releases/tag/V17) 已发布，冻结源码版本为 `0.1.0.dev17`，提交 `27c2e14b184b94d8031a5b3a83c25b1ed3d4ac67`。PyPI **8.0.0 尚未发布**。历史 **7.1.0 wheel 仍是下文记录的 V16 二进制**，不包含 Born 或 V17 的平顶边界、导入初始化变更。

- 主包新增 `starwave.scalar_born` 与 `starwave.ScalarBorn`：一维/二维/三维常密度声学 Born，散射参数为速度扰动 δv，返回背景/散射记录及终态。原生 CPU full、原生 CUDA full/boundary 与显式 Torch CPU/CUDA full/checkpoint 各有独立契约，见 [Born Function](usage.md#born-function)。
- 平顶行为依方程区分：scalar/VRZ 为最后一轴顶部零宽 PML 的固定零 halo；VTI 约束法向牵引为零；elastic 采用第一轴的改进真空自由表面；GSLS 采用二维压力释放闭合。Born 保留独立的逐面非负宽度与零 ghost 约定，不能相互套用，见[模型约定](modeling/conventions.md)。
- 显式编译后，导入会准备已有原生库与共享 dispatch；不编译、不初始化 CUDA、不选择设备、不分配波场，也不传播。手动 prepare 可用于显式检查，但不是首次传播的必需步骤。未构建的库可跳过并使用受支持的显式 Torch 路径；保留的无效构建会报错，不静默忽略。
- 冻结 V17 归档的 `compile_all.py` 构建 core/elastic/GSLS，Born 另需 `python compile_born.py --arch 80`。后续已审 dev `3f2b24d` 将 Born 纳入 `compile_all.py`，一次构建/核验四个库；仍保留 `python compile_born.py --cpu-only`。独立 `starwave_vrr` 扩展继续单独安装/构建，见[安装](installation.md)。

本手册只发布公开契约与合成示例，不包含私有科学实现或源码归档。Release 发布、源码/构建身份、文档检查、CPU 检查与目标 GPU/多卡/FWI 验收分别记录。既有实测教程、图件、PDF 与哈希保留原版本和证据范围；以下均为历史快照，其中 V16 wheel 是历史公开二进制。

## V16 / 7.1.0 · 2026-10-11

[StarWave 7.1.0](https://pypi.org/project/starwave/7.1.0/) 已于 2026-10-11 发布，来自冻结的 V16 源码 `0.1.0.dev16`，提交 `5a224dbb1e86154ff034bbedaf67c0e4cf7a1b6a`，包含 GSLS 和新增 [PyTorch 后端](pytorch-backend.md)。唯一发行文件为 `starwave-7.1.0-py3-none-manylinux_2_35_x86_64.whl`（17,628,561 字节），无 sdist。运行平台为 Linux x86_64 / glibc ≥ 2.35，要求 Python ≥ 3.10 且 < 3.13、PyTorch ≥ 2.5 且 < 2.6。历史 [7.0.0](https://pypi.org/project/starwave/7.0.0/) 仍为不含 GSLS 或统一 Torch 后端的 V14 二进制。

独立下载的公开 wheel 与已核验发行文件的 SHA-256 一致：

```text
36a0be76af997e2d31de8313cab00ff59d79f7537055c829df2bfdee0ff61e88
```

receipt 固定的源码清单 SHA-256 为 `f682ff0a836c0c259cd03a2e760460d68426cc5688e4852f8c62a151804694dc`。[发布工作流](https://github.com/StarrMoonn/StarWave/actions/runs/38108561775) 已通过发行文件核验、Python 3.10/3.11/3.12 发布前主机/CPU 检查、发布及三个 Python 版本的公开 PyPI 回装。这些检查不能证明新二进制的 GPU 数值、多卡、长程 FWI 或性能验收通过。

本次没有 macOS wheel。Mac MPS 需使用已获授权的完整 V16 源码，且仅限下述 Scalar2D 配置。后续 dev 自由表面工作和独立打包的 {ref}`VRR 扩展 <vrr>` 不在 V16 / PyPI 7.1.0 中。此前 VRR 文档审阅覆盖 `904c9106bdb883325df8586b7c3412f7814a9737` 的经审阅源码整合；这是文档更新，不是新包发布。用户提供的设备报告与文档/CPU 检查、公开 wheel 验证分别记录。

- scalar/VRZ/VTI/elastic 新增 backend、execution、checkpoint_interval、compile_steps；GSLS 新增 execution、compile_steps 并扩展 Torch checkpoint。旧位置参数、原生默认行为和科学 CUDA/C++ 核保留。
- Torch 提供 CPU/CUDA full/checkpoint、eager/compile；没有 Torch boundary、AMP、elastic callbacks/offload 或抽样材料梯度。MPS 仅支持 Scalar2D accuracy=4 float32，使用 backend="torch"、full/checkpoint 与 eager/compile；Scalar3D、VRZ、VTI、elastic、GSLS 和其他 MPS 组合明确拒绝。本次不宣称最终 Mac 实机、500 轮 FWI 或性能验收通过。完整参数和限制见 [Usage](usage.md)、{ref}`GSLS <visco-gsls>` 和后端页。
- Torch elastic 对所有内部步求导；原生材料梯度按 CFL ratio × sampling_interval 抽样。精确跨后端比较须 ratio=1，不能把这个差别称作编译误差。
- 文档 gate 核对双语完整签名、默认值、参数覆盖和示例语法，不导入传播库。CPU/文档通过不能替代最终 GPU、DataParallel、长程 FWI 或性能验收；已有示例、图件、PDF 保留原版本与证据范围。

以下 V15 及更早记录仅描述各自历史快照。


## V15 / GSLS 源码 Release · 2026-10-10

[V15 源码 Release](https://github.com/StarrMoonn/StarWave/releases/tag/V15) 已发布，源码版本为 `0.1.0.dev15`，提交 `2508543ccc1dc3015b07806e6e240dd88c8e871e`。本次没有更新 PyPI：公开 7.0.0 仍是历史 V14 二进制，不能用于当前 GSLS 调用。

- {ref}`GSLS <visco-gsls>` 成为唯一黏声接口；独立 `visco_sls` 及其准备/状态入口移除。单机制使用 `mode="sls_compat", n_mechanisms=1`。
- 二维固定可变密度，原生 CPU/CUDA full/checkpoint、显式 Torch full；Vp/Q/source/provider 参数一阶梯度。默认 Hao1、显式 band_fit 与 sls_compat 各有不同 Q 语义，不能直接互换。
- 原有 scalar/VRZ/VTI/elastic 的科学实现、签名与默认值保持不变。GSLS ABI 2 / semantics 1 / capabilities 255 要求配套重建和重启；`compile_all.py` 覆盖 core、elastic、GSLS 三库，CPU GSLS 单独显式构建。
- 导航和当前 API 直接介绍 GSLS。旧独立 SLS API、实验页面和下载均从当前手册移除；原始材料保留在 Git 历史中，不改标为 GSLS。

本次文档更新不宣称最终集成版 GPU、多卡、长程 FWI 或性能验收。安装渠道、历史验证范围与来源见[安装](installation.md)和[文档状态](status.md)。

## 7.0.0 / V14（历史） · 2026-10-09

[StarWave 7.0.0](https://pypi.org/project/starwave/7.0.0/) 对应授权源码 `0.1.0.dev14`，首次公开提供新增并整合的 SLS 黏声学接口。Linux 二进制 wheel 包含核心、elastic、SLS CPU 和 SLS CUDA 四份原生库；不需要用户编译，也不提供 sdist。

- 新增 `starwave.visco_sls`、`starwave.prepare_visco_sls` 与 `starwave.visco_sls_native_status`；原有九个公开函数的签名和默认值不变。
- 二维 single-SLS、固定可变密度、full 体积历史；Vp/Q/source 可独立求一阶梯度。CPU 原生、CUDA 与 Torch 参考后端均显式选择，不自动回退。
- Vp 是 `f_ref` 处相速度，Q 是同频复模量品质因子。源单位为 Pa/m²，首步半权重，记录为更新前的 `p[n]`，不重采样。
- 训练 Vp/Q 且 PML 非零时须固定显式非松弛速度上界 `max_vel`；不支持 rho 梯度、3D 或 boundary。

既有 scalar、VRZ、VTI、elastic 科学调用与 Scalar3D Example 保留。升级后须重启 Python/Notebook kernel；SLS 原生库使用独立准备入口。安装与最终发行身份见[安装](installation.md)和[文档状态](status.md)。

## 6.0.0 / V13 · 2026-10-09

公开 [StarWave 6.0.0 wheel](https://pypi.org/project/starwave/6.0.0/) 对应授权源码 `0.1.0.dev13`，以内部存储和 CUDA 执行路径维护为主；发行文件身份与验收范围见[文档状态](status.md)：

- Scalar2D 使用方向紧凑 PML 状态、宽度 `M=accuracy//2` 的压力边界带，以及普通目标反传（`illumination=None`）时移除 PML/差分 halo、仍含 `boundary_buffer` 区域的 full 历史；CUDA 更新采用 32 × 8 线程块和直接加载。
- 修正 Scalar2D 转置更新中的 plain 区额外支撑及邻点 union-PML 贡献。此修复不补齐模型 replicate-padding 的完整反传转置。
- Elastic2D/3D 的 full/boundary 取消按 L2 缓存容量分组的炮轨迹调度，保留原有内核映射；三维 forward 仍在内核内遍历炮，不能据此理解为全炮 CUDA grid 并行。

公开 API 签名、默认值、源尺度、CFL 和既有调用方式不变。Scalar3D、VRZ 与 VTI 的科学实现不属于本次优化范围。源码用户需重建配套原生库；所有用户升级后均应重启 Python / Notebook kernel，见[安装与升级](installation.md)。

本次文档维护未独立重跑 GPU，未给出通用加速比、峰值显存或优于 SWEEP 的性能结论。原有教程与 Scalar3D Example 保留各自版本和证据范围，不能当作 6.0.0 的新验收。

## 5.0.0 / V12 · 2026-10-08

公开 [StarWave 5.0.0 wheel](https://pypi.org/project/starwave/5.0.0/) 在原签名上扩展 `starwave.scalar`：二维/三维由 `v.ndim` 选择，三维支持不等间距、2/4/6/8 阶、Radius-M 六面压力重建，以及一次一阶速度与源梯度。`full` 保留逐内部步 `Lap(u)` 体积历史。三维模型按自身轴序给出，坐标直接索引；输出仍为单元素 `(receiver_amplitudes,)`，记录 `[B,R,T]`。

二维 scalar 行为与固定源约定保留，VRZ、VTI、Deepwave elastic 及其示例没有随此版本改变。三维要求 CUDA FP32，illumination 仅支持二维 scalar。PML/扩边梯度限制仍适用，boundary 不保证生产规模三维一定放得下显存。公开版本为 5.0.0，不应安装私有源码编号作为 PyPI 版本。

本次维护同步双语 Usage、安装、模型约定、Scalar3D 重建和可运行调用片段，并以 5.0.0 wheel 核对 API。没有修改传播实现，也没有在文档维护中执行 GPU 数值、长程 FWI、多卡或性能测试。发布与历史教程证据见[文档状态](status.md)。

## 4.0.0 · 2026-10-07

公开 [StarWave 4.0.0 wheel](https://pypi.org/project/starwave/4.0.0/) 对应 V11，新增基于 Deepwave 0.0.27 派生后端的 `starwave.elastic(lamb, mu, buoyancy, ...)`，支持二维/三维、2/4/6/8 阶及完整状态/记录返回。材料转换通过 `starwave.common` 显式完成；elastic 使用独立的 `prepare_elastic`。原有 scalar 仍为二维，scalar/VRZ/VTI 的公开签名与默认值保留。

## Elastic API 文档维护 · 2026-10-08

在原 Usage 单页内新增 Elastic Function、62 项参数、16/31 返回顺序、轴序和最小示例，并补充两种材料转换与 prepare_elastic。核对对象为 4.0.0 发布 wheel；审阅明确区分 full/boundary、采样间隔、survey_pad 和已有验证范围。同步安装版本与双语 API 契约检查，保留原 scalar/VRZ/VTI 正文、历史教程数值、版权署名、字体和导航。

本次仅维护文档，不修改传播运行时或发行产物，也未执行 GPU 数值/性能测试。核对来源、wheel 哈希与验收边界见[文档状态](status.md)。

## 2.0.0 · 2026-10-05

公开发行 Linux x86_64 预编译 wheel。该版本改进 VTI 二维/三维 boundary backward 的融合路径；公开 Python API 与默认值不变，scalar/VRZ 不因版本号升为 2.0 而增加新能力。浮点运算顺序变化意味着不能要求与前版逐位相同。

升级后应重启解释器或 Notebook kernel。此版本号不表示新增了目标 GPU 数值、多卡、FWI 或性能验收。已知模型扩展梯度、VRZ 强反差稳定性和 VTI `epsilon < delta` 限制仍需遵守。

安装条件和完整发布范围以 [PyPI 2.0.0](https://pypi.org/project/starwave/2.0.0/) 为公开版本来源。

## 本使用手册首稿

新增安装/WSL、合成 scalar 示例、三类方程使用说明、FWI/DataParallel/INR 入门、API 索引与 FAQ。初稿采用 Sphinx + Read the Docs 主题，可输出到 GitHub Pages 或其它静态服务器；当前双语版保留这一初始模板。

本手册首稿不附带 CUDA 实现、传播 Python 实现、私人实验数据或内部记录。待完成的教程与验收见[文档状态](status.md)。

## API 参考完善

scalar、VRZ、VTI 各自提供独立参考页，以 Sphinx Python 域展示完整签名、逐项参数类型/默认值/形状/单位、返回结构、自动微分范围及示例。补充 54 项参数说明；保留原 API 索引中的函数入口锚点。此为文档改进，不改变 StarWave 2.0.0 的运行接口或验收状态。

## 中英双语版

完整手册提供中文与 English，包括导航、搜索、提示与 18 个内容页面。各语言独立构建和索引，语言切换保留对应页面与章节；旧中文链接与 API 锚点保持可访问。站点保留最初的 PyFWI / Read the Docs 模板，API 内容组织参考 Deepwave 的 Sphinx 结构，增加类型签名，并继续保留 StarWave 自身的返回与数值限制。


## 历史 V16 安装与迁移

以下归档命令用于复现 V16 / 7.1.0 渠道，不是 V19 安装路径。当前源码或已发布二进制的选择见[安装](installation.md)。

### 历史 7.1.0 wheel 运行条件

| 项目 | StarWave 7.1.0 wheel 要求 |
|---|---|
| 系统 | Linux x86_64，glibc ≥ 2.35 |
| C++ 运行库 | libstdc++ 提供 GLIBCXX_3.4.30 / CXXABI_1.3.13 或更新符号 |
| Python | ≥ 3.10 且 < 3.13（3.10–3.12） |
| PyTorch / NumPy | PyTorch ≥ 2.5 且 < 2.6；NumPy ≥ 1.23 |
| 后端 / 设备 | 原生 scalar 2D/3D、VRZ/VTI 与 elastic boundary 需要 NVIDIA CUDA；elastic full 与 GSLS native_cpu 支持 CPU。五种传播入口的显式 Torch full/checkpoint 均支持 CPU/CUDA |
| 编译目标 | SASS 70/75/80/86/89/90 与 compute80 PTX |

安装预编译 wheel 不需要 nvcc 或本地 CUDA Toolkit。Linux 驱动保守目标为 520.61.05 或更新，同时必须支持实际 GPU；WSL 使用 Windows 主机驱动，见 [WSL 说明](wsl.md)。A30 与 RTX 4060 Laptop 属于编译目标覆盖范围；公开 7.1.0 wheel 的实机数值与性能测试仍待完成。下文另列开发版 A30 实测记录。

以上版本信息来自 [PyPI 7.1.0 发布说明](https://pypi.org/project/starwave/7.1.0/)，发行文件身份及核验范围见[文档状态](status.md)。公开发行仅有 Linux wheel，没有 sdist；原生 Windows、macOS、ARM 主机不能直接使用这一 wheel。

### 安装历史 7.1.0 wheel

以下命令在符合条件的 Linux 或 WSL 2 Linux 终端中执行。选择 Python 3.10–3.12；示例使用 3.12。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --only-binary=starwave starwave==7.1.0
python -m pip check
```

PyTorch 安装命令依据 [PyTorch 历史版本页面](https://pytorch.org/get-started/previous-versions/)。`cu118` 是此版本的起始验证配置，不代表其它 PyTorch 版本已被验证。升级 StarWave 后重启 Python 或 Notebook kernel。

### 历史迁移：V16 / 7.1.0

V16 的唯一黏声入口为 `starwave.visco_gsls`；独立 `visco_sls` 及其准备/状态入口已移除。单机制计算须显式使用 `mode="sls_compat", n_mechanisms=1`，不能只更换函数名而沿用默认 Hao1。源码编号 `0.1.0.dev16` 是公开 `7.1.0` wheel 所依据的冻结源码版本；二者采用不同发行版本号。

源码用户必须使用这一版本配套的 Python 包与重新构建的原生库，按随源码提供的构建说明操作。从早于 6.0.0 的版本升级时，Scalar2D 的 PML、边界与 full 历史布局已改变；即使核心 ABI 布局相同，旧库也不能复用或通过绕过兼容检查加载。安装公开 wheel 不需要自行编译。升级或替换已加载的库后，重启 Python / Notebook kernel，再执行原生准备与小规模正演/梯度检查。

内部运算顺序与 PML 转置修复可能改变末位数值或原先有误的梯度，不要求与旧版本逐位一致。内存优化不改变模型扩边梯度限制，也不保证任意配置的速度或显存收益。



## 历史产物核验记录

以下保留原版本的文档与产物核验记录，均不作为 V19 运行验收结论。

### V18 / PyPI 9.0.0 · 2026-10-12

V18 scalar（审阅基于 dev `e040aa9b8225ab387f7e4853bfe6162f5351f5d5`） 新增任意零面组合（正宽度仍须相等）、二维/三维可训练源、可选可微最终物理波场与诊断前向回调。这些源码功能，不在冻结 V17/其 8.0.0 发布基线或 7.1.0 中。23 参数契约见 [Scalar Function](usage.md#scalar-function)，较早的 20 参数发布快照仍单独固定。

[Release V18](https://github.com/StarrMoonn/StarWave/releases/tag/V18) 已发布，冻结提交为 `fba1f496b890f2dbaf3fbb3011d39f6d723cfea1`，保留已审 dev `e040aa9` 的 scalar API。[PyPI 9.0.0](https://pypi.org/project/starwave/9.0.0/) 已发布，受支持的 Linux x86_64 主机可安装包含这些功能的 wheel，条件和命令见[安装](installation.md)。V17 与未发布的 8.0.0 基线保持不变；7.1.0 为历史 V16 wheel。

公开文件为 `starwave-9.0.0-py3-none-manylinux_2_35_x86_64.whl`（27,975,674 字节），2026-10-12 上传，SHA-256 为 `d290ea0a7fdc2a74229d989525aa2bff1cdc9d675fb715bd6f43a6e17b1ddd0b`。没有 sdist 或 macOS wheel。要求 Linux x86_64 / glibc ≥ 2.35、Python ≥ 3.10 且 < 3.13、PyTorch ≥ 2.5 且 < 2.6、NumPy ≥ 1.23。公开下载文件已逐字节核对哈希。[发布工作流](https://github.com/StarrMoonn/StarWave/actions/runs/38199707015) 八项作业全部通过，包括 Python 3.10、3.11、3.12 的公开 PyPI 干净下载、安装与 smoke 检查。这些是打包和主机/CPU 检查，不代表公开 wheel 的 GPU、多卡、长程 FWI 或性能验收。Mac MPS 仍需源码，且只限已说明的 Scalar2D 配置。

文档 CPU checkpoint 示例在 Torch 2.5.1+cpu 通过：终态 (1,12,10)、空记录 (1,0,7)、用户步 3/6/7 与内部步 6/12/14 回调、有限模型/源梯度，反传重算不重复回调。它只是一项小规模 CPU 接线检查。用户提供了 RTX 4060/A30 功能分支报告；本次文档更新未重跑 GPU 传播，也不把该结果转为合并构建或发布 wheel 的验收。

### V17 源码、Born 与平顶边界

[Release V17](https://github.com/StarrMoonn/StarWave/releases/tag/V17) 已发布，冻结源码版本为 `0.1.0.dev17`，提交 `27c2e14b184b94d8031a5b3a83c25b1ed3d4ac67`。PyPI **8.0.0 尚未发布**。历史 **7.1.0 wheel 仍是下文记录的 V16 二进制**，不包含 Born 或 V17 的平顶边界、导入初始化变更。

- 主包新增 `starwave.scalar_born` 与 `starwave.ScalarBorn`：一维/二维/三维常密度声学 Born，散射参数为速度扰动 δv，返回背景/散射记录及终态。原生 CPU full、原生 CUDA full/boundary 与显式 Torch CPU/CUDA full/checkpoint 各有独立契约，见 [Born Function](usage.md#born-function)。
- 平顶行为依方程区分：scalar/VRZ 为最后一轴顶部零宽 PML 的固定零 halo；VTI 约束法向牵引为零；elastic 采用第一轴的改进真空自由表面；GSLS 采用二维压力释放闭合。Born 保留独立的逐面非负宽度与零 ghost 约定，不能相互套用，见[模型约定](modeling/conventions.md)。
- 显式编译后，导入会准备已有原生库与共享 dispatch；不编译、不初始化 CUDA、不选择设备、不分配波场，也不传播。手动 prepare 可用于显式检查，但不是首次传播的必需步骤。未构建的库可跳过并使用受支持的显式 Torch 路径；保留的无效构建会报错，不静默忽略。
- 冻结 V17 归档的 `compile_all.py` 构建 core/elastic/GSLS，Born 另需 `python compile_born.py --arch 80`。后续已审 dev `3f2b24d` 将 Born 纳入 `compile_all.py`，一次构建/核验四个库；仍保留 `python compile_born.py --cpu-only`。独立 `starwave_vrr` 扩展继续单独安装/构建，见[安装](installation.md)。

本手册只发布公开契约与合成示例，不包含私有科学实现或源码归档。Release 发布、源码/构建身份、文档检查、CPU 检查与目标 GPU/多卡/FWI 验收分别记录。既有实测教程、图件、PDF 与哈希保留原版本和证据范围；以下均为历史快照，其中 V16 wheel 是历史公开二进制。

### V16 / 7.1.0 · 2026-10-11

[StarWave 7.1.0](https://pypi.org/project/starwave/7.1.0/) 已于 2026-10-11 发布，来自冻结的 V16 源码 `0.1.0.dev16`，提交 `5a224dbb1e86154ff034bbedaf67c0e4cf7a1b6a`，包含 GSLS 和新增 [PyTorch 后端](pytorch-backend.md)。唯一发行文件为 `starwave-7.1.0-py3-none-manylinux_2_35_x86_64.whl`（17,628,561 字节），无 sdist。运行平台为 Linux x86_64 / glibc ≥ 2.35，要求 Python ≥ 3.10 且 < 3.13、PyTorch ≥ 2.5 且 < 2.6。历史 [7.0.0](https://pypi.org/project/starwave/7.0.0/) 仍为不含 GSLS 或统一 Torch 后端的 V14 二进制。

独立下载的公开 wheel 与已核验发行文件的 SHA-256 一致：

```text
36a0be76af997e2d31de8313cab00ff59d79f7537055c829df2bfdee0ff61e88
```

receipt 固定的源码清单 SHA-256 为 `f682ff0a836c0c259cd03a2e760460d68426cc5688e4852f8c62a151804694dc`。[发布工作流](https://github.com/StarrMoonn/StarWave/actions/runs/38108561775) 已通过发行文件核验、Python 3.10/3.11/3.12 发布前主机/CPU 检查、发布及三个 Python 版本的公开 PyPI 回装。这些检查不能证明新二进制的 GPU 数值、多卡、长程 FWI 或性能验收通过。

本次没有 macOS wheel。Mac MPS 需使用已获授权的完整 V16 源码，且仅限下述 Scalar2D 配置。后续 dev 自由表面工作和独立打包的 {ref}`VRR 扩展 <vrr>` 不在 V16 / PyPI 7.1.0 中。此前 VRR 文档审阅覆盖 `904c9106bdb883325df8586b7c3412f7814a9737` 的经审阅源码整合；这是文档更新，不是新包发布。用户提供的设备报告与文档/CPU 检查、公开 wheel 验证分别记录。

- scalar/VRZ/VTI/elastic 新增 backend、execution、checkpoint_interval、compile_steps；GSLS 新增 execution、compile_steps 并扩展 Torch checkpoint。旧位置参数、原生默认行为和科学 CUDA/C++ 核保留。
- Torch 提供 CPU/CUDA full/checkpoint、eager/compile；没有 Torch boundary、AMP、elastic callbacks/offload 或抽样材料梯度。MPS 仅支持 Scalar2D accuracy=4 float32，使用 backend="torch"、full/checkpoint 与 eager/compile；Scalar3D、VRZ、VTI、elastic、GSLS 和其他 MPS 组合明确拒绝。本次不宣称最终 Mac 实机、500 轮 FWI 或性能验收通过。完整参数和限制见 [Usage](usage.md)、{ref}`GSLS <visco-gsls>` 和后端页。
- Torch elastic 对所有内部步求导；原生材料梯度按 CFL ratio × sampling_interval 抽样。精确跨后端比较须 ratio=1，不能把这个差别称作编译误差。
- 文档 gate 核对双语完整签名、默认值、参数覆盖和示例语法，不导入传播库。CPU/文档通过不能替代最终 GPU、DataParallel、长程 FWI 或性能验收；已有示例、图件、PDF 保留原版本与证据范围。

以下 V15 及更早记录仅描述各自历史快照。


### V15 / GSLS · 2026-10-10

V15 手册当时依据已发布的 [V15 源码 Release](https://github.com/StarrMoonn/StarWave/releases/tag/V15)，源码版本 `0.1.0.dev15`，对应提交 `2508543ccc1dc3015b07806e6e240dd88c8e871e`。Release 发布与 PyPI 二进制发行分开：原公开 **7.0.0** 未更新，不包含 GSLS，也不包含独立 SLS 的源码清理。

V15 黏声入口为 {ref}`GSLS <visco-gsls>`：二维、固定空间变密度、原生 CPU/CUDA full/checkpoint、显式 Torch full、一阶 Vp/Q/source/provider 梯度。独立 SLS API/生产源码/构建入口已移除，单机制显式选择 `mode="sls_compat", n_mechanisms=1`。本手册不发布传播实现、CUDA 源码或完整源码归档。

参数和物理说明以该 V15 源码契约为准。默认 Hao1 的 Q 为名义 Q0；Vp 是参考相速度；训练材料且 PML 非零时须固定覆盖非松弛速度的 `max_vel`。不支持三维、rho 梯度、boundary 或传播高阶导数。完整限制见 GSLS 页，显式构建和重启步骤见[安装](installation.md)。

提供的 A30/RTX 4060 记录属于集成前的源码指纹；host guard 与集成后的身份不同，不能作为最终集成版 GPU 重跑的证明。本次手册更新没有运行 GPU、DataParallel、长程 FWI 或目标设备性能测试。旧独立 SLS API、实验页面及下载已从当前手册移除；原始材料保留在 Git 历史中，不改标为 GSLS 结果。

以下各节保留历史发行核验事实，不能转移为 V15 的新验收结论。

### V14 / 7.0.0 SLS（历史）

[StarWave 7.0.0](https://pypi.org/project/starwave/7.0.0/) 已于 2026-10-09 发布，对应授权 V14 / `0.1.0.dev14`。唯一发行文件为 `starwave-7.0.0-py3-none-manylinux_2_35_x86_64.whl`（17,515,547 字节），无 sdist。官方 PyPI 文件与已审 CI 构建逐字节一致；Python 3.10–3.12 安装/主机检查与公开文件干净环境回装通过。

当时新增双语 SLS API，保留原有九函数契约并增加三个入口。十二份签名、默认值、关键字边界和返回契约已与真实 wheel 对齐；核心、elastic、SLS CPU、SLS CUDA 四份 ELF 及 receipt 哈希均已核验。文档检查器读取归档，不加载或执行原生库。

7.0.0 公开 wheel SHA-256：

```text
0fb2d66555ad19d4178738db93ea5d71c3b7dbfb456cf4fd82898f5d44725964
```

适用范围为二维 single-SLS、固定可变密度、full 历史以及 Vp/Q/source 一阶梯度。当时记录的源码 GPU 结果包括：两张 A30 的固定 Q / 联合 Vp/Q 各 100 轮，以及独立 RTX 4060 环形小模型。这些原运行记录与公开 wheel 回装核验属于不同阶段。本次文档维护没有独立重跑 GPU。既有教程、图件、Scalar3D 分卷下载与 PDF 保留原版本和证据范围。

### V13 / 6.0.0 维护范围 · 2026-10-09

本次维护面向 6.0.0 / 源码 `0.1.0.dev13`，同步安装版本、旧库兼容警告、Scalar2D 内部存储和 PML 转置修复，以及 elastic 取消 L2 炮组轨迹调度的说明。九个公开函数的参数签名与既有 5.0.0 契约保持一致；不增加 API 参数或改写示例调用。

[StarWave 6.0.0](https://pypi.org/project/starwave/6.0.0/) 已于 2026-10-09 发布。官方 PyPI 版本记录仅含 `starwave-6.0.0-py3-none-manylinux_2_35_x86_64.whl`（16,674,439 字节），无 sdist；从官方文件地址独立下载后的字节数与 SHA-256 均匹配构建产物。已核对版本与平台元数据、两份原生库的 receipt 哈希，以及九个公开签名、默认值和已记录的类型/返回契约。文档检查器只读取归档，不导入或执行 StarWave。

发布流程通过 Python 3.10–3.12 的安装及主机/CPU 检查，并通过公开 PyPI 的干净环境回装检查。这些结果不代表 CUDA 数值验收；scalar3D 仍没有 CPU 传播路径。

6.0.0 公开 wheel SHA-256：

```text
297a4e359d86003513452294e6384f78a6ab7029fdabefa936e0733228882cf7
```

本次没有独立执行 GPU 数值、性能、多卡或长程 FWI 测试。存储布局说明不是峰值显存实测，也不构成通用速度保证。现有教程和 Scalar3D Example 的图件、下载与实测记录保持不变，其原始版本和适用范围继续有效。

### V12 / 5.0.0 Scalar3D API 核对 · 2026-10-08

本节历史安装版本为公开 [StarWave 5.0.0](https://pypi.org/project/starwave/5.0.0/)。同一个 `starwave.scalar` 签名按 `v.ndim` 选择二维/三维。三维支持逐轴不等间距、2/4/6/8 阶、Radius-M 六面 boundary 或 full、一次一阶速度与源梯度；二维源仍固定，三维 illumination 不支持。原有 VRZ、VTI 和 Deepwave elastic 契约保留。

本次独立读取公开 wheel，并核对 SHA-256、九个公开签名、参数默认值、类型和返回契约；还将 scalar、三维包装/保存布局、输入验证和时间采样模块逐字节与本次授权源码核对。双语文档与调用片段按 Python 3.10 语法检查，不在文档构建时加载或运行 StarWave。

5.0.0 公开 wheel SHA-256：

```text
bf159a6544cdc1ab12c99e22578ec40ed56e0d3011ae71be9fc8ecb04f9c2bff
```

发布记录确认 Python 3.10–3.12 安装检查及公开 wheel 独立下载/安装核验。CPU/主机检查不能代表 scalar3D CPU 传播，因为该路径只支持 CUDA。用户提供的源版本 GPU 报告属于另行标明的历史证据；本次文档维护未重跑 GPU，不宣称公开 wheel 的 A30、多 GPU、长程 FWI、峰值显存或性能已验收。下列 4.0.0、2.0.0 与 0.1.0.dev9 记录保留各自原始版本，不能改标为 5.0.0 新实验。

### V11 / 4.0.0 Elastic API 核对 · 2026-10-08

[StarWave 4.0.0](https://pypi.org/project/starwave/4.0.0/) 的发布 wheel 为本次新增接口的依据。62 项 elastic 参数、两个转换函数和 prepare_elastic 已逐项核对类型、顺序、默认值与调用边界；原有五个 API 签名在该 wheel 中保持不变。双语构建与本地链接/锚点检查包含新接口，Python 示例按 3.10 语法检查。没有在文档构建中导入或执行 StarWave。

公开 wheel SHA-256：

```text
9f64f2677c54af5b2bd1c48e509a24c7d40eb0257d9e86267e721d3f61eaa954
```

发布阶段已通过 Python 3.10–3.12 安装检查及公开 wheel 的二维/三维 CPU/主机检查；用户提供的 RTX 4060 源码测试报告是独立证据。本次未重跑 GPU，不宣称公开 wheel 的目标 GPU 数值、A30、多卡、长程 FWI 或性能已验收。下面保留原有教程和 2.0.0 文档基线的记录，不能把历史结果改标为 4.0.0 的新实验。



### 历史 2.0.0 文档基线

以下为历史 2.0.0 文档基线：API 核对基于当时发布 wheel 的 Python 接口签名与授权使用说明；文档重新组织为入门内容，不包含软件实现。公开 wheel SHA-256：

```text
6622b863c76db1ba708622048377f3de4447609ff7295d1705b8145884ce06db
```

最后核对日期：2026-10-05。参考资料更新后应按实际发行版本重新审阅安装条件与 API。
