# 文档状态与参考来源

## V16 / PyTorch 后端 · 发布核验待完成

当前手册同步 V16 / `0.1.0.dev16` 的新增 [PyTorch 后端](pytorch-backend.md)。V16 源码 Release、开发分支提交和目标 PyPI `7.1.0` wheel 的远端身份尚待最终核验；本页不把计划描述为已发布。已有 [7.0.0](https://pypi.org/project/starwave/7.0.0/) 为历史 V14 二进制，不含 GSLS 或新增统一 Torch 后端。

- scalar/VRZ/VTI/elastic 新增 backend、execution、checkpoint_interval、compile_steps；GSLS 新增 execution、compile_steps 并扩展 Torch checkpoint。旧位置参数、原生默认行为和科学 CUDA/C++ 核保留。
- Torch 提供 CPU/CUDA full/checkpoint、eager/compile；没有 Torch boundary、AMP、elastic callbacks/offload 或抽样材料梯度。MPS 仅支持 Scalar2D accuracy=4 float32，使用 backend="torch"、full/checkpoint 与 eager/compile；Scalar3D、VRZ、VTI、elastic、GSLS 和其他 MPS 组合明确拒绝。本次不宣称最终 Mac 实机、500 轮 FWI 或性能验收通过。完整参数和限制见 [Usage](usage.md)、[GSLS](visco-gsls.md) 和后端页。
- Torch elastic 对所有内部步求导；原生材料梯度按 CFL ratio × sampling_interval 抽样。精确跨后端比较须 ratio=1，不能把这个差别称作编译误差。
- 文档 gate 核对双语完整签名、默认值、参数覆盖和示例语法，不导入传播库。CPU/文档通过不能替代最终 GPU、DataParallel、长程 FWI 或性能验收；已有示例、图件、PDF 保留原版本与证据范围。

以下 V15 及更早记录仅描述各自历史快照。


## V15 / GSLS · 2026-10-10

V15 手册当时依据已发布的 [V15 源码 Release](https://github.com/StarrMoonn/StarWave/releases/tag/V15)，源码版本 `0.1.0.dev15`，对应提交 `2508543ccc1dc3015b07806e6e240dd88c8e871e`。Release 发布与 PyPI 二进制发行分开：原公开 **7.0.0** 未更新，不包含 GSLS，也不包含独立 SLS 的源码清理。

V15 黏声入口为 [GSLS](visco-gsls.md)：二维、固定空间变密度、原生 CPU/CUDA full/checkpoint、显式 Torch full、一阶 Vp/Q/source/provider 梯度。独立 SLS API/生产源码/构建入口已移除，单机制显式选择 `mode="sls_compat", n_mechanisms=1`。本手册不发布传播实现、CUDA 源码或完整源码归档。

参数和物理说明以该 V15 源码契约为准。默认 Hao1 的 Q 为名义 Q0；Vp 是参考相速度；训练材料且 PML 非零时须固定覆盖非松弛速度的 `max_vel`。不支持三维、rho 梯度、boundary 或传播高阶导数。完整限制见 GSLS 页，显式构建和重启步骤见[安装](installation.md)。

提供的 A30/RTX 4060 记录属于集成前的源码指纹；host guard 与集成后的身份不同，不能作为最终集成版 GPU 重跑的证明。本次手册更新没有运行 GPU、DataParallel、长程 FWI 或目标设备性能测试。旧独立 SLS API、实验页面及下载已从当前手册移除；原始材料保留在 Git 历史中，不改标为 GSLS 结果。

以下各节保留历史发行核验事实，不能转移为 V15 的新验收结论。

## V14 / 7.0.0 SLS（历史）

[StarWave 7.0.0](https://pypi.org/project/starwave/7.0.0/) 已于 2026-10-09 发布，对应授权 V14 / `0.1.0.dev14`。唯一发行文件为 `starwave-7.0.0-py3-none-manylinux_2_35_x86_64.whl`（17,515,547 字节），无 sdist。官方 PyPI 文件与已审 CI 构建逐字节一致；Python 3.10–3.12 安装/主机检查与公开文件干净环境回装通过。

当时新增双语 SLS API，保留原有九函数契约并增加三个入口。十二份签名、默认值、关键字边界和返回契约已与真实 wheel 对齐；核心、elastic、SLS CPU、SLS CUDA 四份 ELF 及 receipt 哈希均已核验。文档检查器读取归档，不加载或执行原生库。

7.0.0 公开 wheel SHA-256：

```text
0fb2d66555ad19d4178738db93ea5d71c3b7dbfb456cf4fd82898f5d44725964
```

适用范围为二维 single-SLS、固定可变密度、full 历史以及 Vp/Q/source 一阶梯度。当时记录的源码 GPU 结果包括：两张 A30 的固定 Q / 联合 Vp/Q 各 100 轮，以及独立 RTX 4060 环形小模型。这些原运行记录与公开 wheel 回装核验属于不同阶段。本次文档维护没有独立重跑 GPU。既有教程、图件、Scalar3D 分卷下载与 PDF 保留原版本和证据范围。

## V13 / 6.0.0 维护范围 · 2026-10-09

本次维护面向 6.0.0 / 源码 `0.1.0.dev13`，同步安装版本、旧库兼容警告、Scalar2D 内部存储和 PML 转置修复，以及 elastic 取消 L2 炮组轨迹调度的说明。九个公开函数的参数签名与既有 5.0.0 契约保持一致；不增加 API 参数或改写示例调用。

[StarWave 6.0.0](https://pypi.org/project/starwave/6.0.0/) 已于 2026-10-09 发布。官方 PyPI 版本记录仅含 `starwave-6.0.0-py3-none-manylinux_2_35_x86_64.whl`（16,674,439 字节），无 sdist；从官方文件地址独立下载后的字节数与 SHA-256 均匹配构建产物。已核对版本与平台元数据、两份原生库的 receipt 哈希，以及九个公开签名、默认值和已记录的类型/返回契约。文档检查器只读取归档，不导入或执行 StarWave。

发布流程通过 Python 3.10–3.12 的安装及主机/CPU 检查，并通过公开 PyPI 的干净环境回装检查。这些结果不代表 CUDA 数值验收；scalar3D 仍没有 CPU 传播路径。

6.0.0 公开 wheel SHA-256：

```text
297a4e359d86003513452294e6384f78a6ab7029fdabefa936e0733228882cf7
```

本次没有独立执行 GPU 数值、性能、多卡或长程 FWI 测试。存储布局说明不是峰值显存实测，也不构成通用速度保证。现有教程和 Scalar3D Example 的图件、下载与实测记录保持不变，其原始版本和适用范围继续有效。

## V12 / 5.0.0 Scalar3D API 核对 · 2026-10-08

本节历史安装版本为公开 [StarWave 5.0.0](https://pypi.org/project/starwave/5.0.0/)。同一个 `starwave.scalar` 签名按 `v.ndim` 选择二维/三维。三维支持逐轴不等间距、2/4/6/8 阶、Radius-M 六面 boundary 或 full、一次一阶速度与源梯度；二维源仍固定，三维 illumination 不支持。原有 VRZ、VTI 和 Deepwave elastic 契约保留。

本次独立读取公开 wheel，并核对 SHA-256、九个公开签名、参数默认值、类型和返回契约；还将 scalar、三维包装/保存布局、输入验证和时间采样模块逐字节与本次授权源码核对。双语文档与调用片段按 Python 3.10 语法检查，不在文档构建时加载或运行 StarWave。

5.0.0 公开 wheel SHA-256：

```text
bf159a6544cdc1ab12c99e22578ec40ed56e0d3011ae71be9fc8ecb04f9c2bff
```

发布记录确认 Python 3.10–3.12 安装检查及公开 wheel 独立下载/安装核验。CPU/主机检查不能代表 scalar3D CPU 传播，因为该路径只支持 CUDA。用户提供的源版本 GPU 报告属于另行标明的历史证据；本次文档维护未重跑 GPU，不宣称公开 wheel 的 A30、多 GPU、长程 FWI、峰值显存或性能已验收。下列 4.0.0、2.0.0 与 0.1.0.dev9 记录保留各自原始版本，不能改标为 5.0.0 新实验。

## V11 / 4.0.0 Elastic API 核对 · 2026-10-08

[StarWave 4.0.0](https://pypi.org/project/starwave/4.0.0/) 的发布 wheel 为本次新增接口的依据。62 项 elastic 参数、两个转换函数和 prepare_elastic 已逐项核对类型、顺序、默认值与调用边界；原有五个 API 签名在该 wheel 中保持不变。双语构建与本地链接/锚点检查包含新接口，Python 示例按 3.10 语法检查。没有在文档构建中导入或执行 StarWave。

公开 wheel SHA-256：

```text
9f64f2677c54af5b2bd1c48e509a24c7d40eb0257d9e86267e721d3f61eaa954
```

发布阶段已通过 Python 3.10–3.12 安装检查及公开 wheel 的二维/三维 CPU/主机检查；用户提供的 RTX 4060 源码测试报告是独立证据。本次未重跑 GPU，不宣称公开 wheel 的目标 GPU 数值、A30、多卡、长程 FWI 或性能已验收。下面保留原有教程和 2.0.0 文档基线的记录，不能把历史结果改标为 4.0.0 的新实验。

## 已完成与待完善

| 内容 | 当前状态 |
|---|---|
| 中英双语导航、安装与 WSL 指引 | 已完整翻译；另附 A30 / 0.1.0.dev9 的小规模运行记录，公开 wheel 仍待实测 |
| scalar / VRZ / VTI API 参考 | 单页 Usage；66 项参数（V16）的签名、默认值、类型、shape/单位与约束已核对 |
| Scalar3D / 5.0.0 | API、Radius-M 存储契约与独立小例子已补充；本次未运行 GPU |
| 原创三份教学 notebook | 服务器返回执行结果无错误；运行产物哈希及数组已核验 |
| scalar 正演、梯度与 25 次 FWI 更新 | A30 / 0.1.0.dev9 实测；数据拟合改善明显，模型恢复改善仅 0.86% |
| 原有 32 × 32 接线脚本 | 保留并通过语法检查；该独立配置没有附实测结果 |
| VRZ / VTI | 使用约定与调用片段已提供；完整独立示例待补 |
| DataParallel | 按炮拆分接线片段；多 GPU 对照待测 |
| INR | 概念与接线片段；独立网络程序及收敛实验待补 |
| 照明 API | 仅列导出名称；完整生命周期教程待补 |
| 伴随、性能、设备范围 | 方向导数与完整伴随未验证；无 GPU 计时、多卡或公开 2.0.0 wheel 验收结论 |

## 参考来源

- [StarWave 2.0.0 PyPI](https://pypi.org/project/starwave/2.0.0/)：公开发行要求与限制。
- [Deepwave API 文档](https://ausargeo.com/deepwave/usage)：参考 API 内容组织，站点保留 Read the Docs 主题，StarWave 说明均为原创。
- [PyFWI 文档](https://pyfwi.readthedocs.io/en/latest/)：仅参考手册导航组织。
- [PyTorch 历史版本安装](https://pytorch.org/get-started/previous-versions/) 与 [DataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)。
- [Microsoft WSL 安装](https://learn.microsoft.com/en-us/windows/wsl/install) 与 [NVIDIA WSL 用户指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。
- [Sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html)、[GitHub Pages 自定义工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) 和 [RTD 配置说明](https://docs.readthedocs.com/platform/stable/config-file/v2.html)。

以下为历史 2.0.0 文档基线：API 核对基于当时发布 wheel 的 Python 接口签名与授权使用说明；文档重新组织为入门内容，不包含软件实现。公开 wheel SHA-256：

```text
6622b863c76db1ba708622048377f3de4447609ff7295d1705b8145884ce06db
```

最后核对日期：2026-10-05。参考资料更新后应按实际发行版本重新审阅安装条件与 API。

(tutorial-evidence)=
## 教程结果的来源与复现范围

三份原创 notebook 的服务器回传代码与交付原件一致，执行输出没有错误。已对 39 个运行产物核对 SHA-256 与文件大小，并检查数值数组的形状、有限性、固定边带和最终迭代对应关系。文档编辑过程中没有重新执行 CUDA solver。

- {ref}`安装运行检查 <installation-smoke>`：48 × 40，记录 `[1,12,160]`，有限非零正演与一阶速度梯度。
- [简单梯度计算](modeling/gradient.md)：96 × 64，三炮，固定边带梯度为零；方向导数检查关闭。
- [简单 FWI 示例](inversion/fwi.md)：25 次更新，数据项下降 91.51%，活动区速度 RMSE 仅下降 0.86%。

记录环境为 NVIDIA A30、Python 3.10.18、PyTorch 2.5.1 / CUDA 11.8、StarWave **0.1.0.dev9**。这不证明公开 **2.0.0 wheel** 已被运行，也不根据原 notebook 中的参考开发提交推断实际安装提交。

公开 notebook 是未重新执行的隐私整理版：清空输出、执行计数及机器相关元数据，删除私有源码安装段与两个参考提交来源字段，其余数值、绘图和导出单元保持原样。中文和英文页面提供同一套原始中文教学 notebook；配套 `.py` 与公开 notebook 的代码一致。它们不是把回传 notebook 的完整运行输出直接上网。

可下载 {download}`精确配置与结果摘要 <../examples/tutorials/results_summary.json>`、{download}`原件与公开版哈希及修改范围 <../examples/tutorials/source_manifest.json>`。图件由已核验数组重绘；共享色标、数据源文件哈希和图件哈希见 [图件来源记录](_static/tutorials/figure_manifest.json)。原始系统信息、环境日志和私有路径不公开。
