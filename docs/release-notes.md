# 发布说明

## V16 / PyTorch 后端 · 发布核验待完成

当前手册同步 V16 / `0.1.0.dev16` 的新增 [PyTorch 后端](pytorch-backend.md)。V16 源码 Release、开发分支提交和目标 PyPI `7.1.0` wheel 的远端身份尚待最终核验；本页不把计划描述为已发布。已有 [7.0.0](https://pypi.org/project/starwave/7.0.0/) 为历史 V14 二进制，不含 GSLS 或新增统一 Torch 后端。

- scalar/VRZ/VTI/elastic 新增 backend、execution、checkpoint_interval、compile_steps；GSLS 新增 execution、compile_steps 并扩展 Torch checkpoint。旧位置参数、原生默认行为和科学 CUDA/C++ 核保留。
- Torch 提供 CPU/CUDA full/checkpoint、eager/compile；没有 Torch boundary、AMP、elastic callbacks/offload 或抽样材料梯度。MPS 仅支持 Scalar2D accuracy=4 float32，使用 backend="torch"、full/checkpoint 与 eager/compile；Scalar3D、VRZ、VTI、elastic、GSLS 和其他 MPS 组合明确拒绝。本次不宣称最终 Mac 实机、500 轮 FWI 或性能验收通过。完整参数和限制见 [Usage](usage.md)、[GSLS](visco-gsls.md) 和后端页。
- Torch elastic 对所有内部步求导；原生材料梯度按 CFL ratio × sampling_interval 抽样。精确跨后端比较须 ratio=1，不能把这个差别称作编译误差。
- 文档 gate 核对双语完整签名、默认值、参数覆盖和示例语法，不导入传播库。CPU/文档通过不能替代最终 GPU、DataParallel、长程 FWI 或性能验收；已有示例、图件、PDF 保留原版本与证据范围。

以下 V15 及更早记录仅描述各自历史快照。


## V15 / GSLS 源码 Release · 2026-10-10

[V15 源码 Release](https://github.com/StarrMoonn/StarWave/releases/tag/V15) 已发布，源码版本为 `0.1.0.dev15`，提交 `2508543ccc1dc3015b07806e6e240dd88c8e871e`。本次没有更新 PyPI：公开 7.0.0 仍是历史 V14 二进制，不能用于当前 GSLS 调用。

- [GSLS](visco-gsls.md) 成为唯一黏声接口；独立 `visco_sls` 及其准备/状态入口移除。单机制使用 `mode="sls_compat", n_mechanisms=1`。
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
