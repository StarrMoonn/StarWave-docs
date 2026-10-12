# 文档状态与参考来源

## V19 / 0.1.0.dev19 · 2026-10-12

[V19 Release](https://github.com/StarrMoonn/StarWave/releases/tag/V19) 已发布；本手册对应冻结源码 `0.1.0.dev19`、提交 `46d32877e3336d531520a696d5f2e074e6f4ef74`。[PyPI **9.1.0**](https://pypi.org/project/starwave/9.1.0/) 已发布，对应 V19 源码契约并包含集成 VRR。现有 [9.0.0 wheel](https://pypi.org/project/starwave/9.0.0/) 仍对应 V18，不含集成 VRR。

公开 wheel：`starwave-9.1.0-py3-none-manylinux_2_35_x86_64.whl`，29,014,423 字节，上传时间 2026-10-12 06:05:56 UTC。SHA-256：`07e9bfe00425e8a28b67805c1e300a05f9f60a4d9b963fa7d4f7b1c0425d0df7`。公开 PyPI 元数据与已审产物一致。打包和 CPU 检查不代表 GPU、多卡、MPS、长程 FWI 或性能验收。

V19 将 {ref}`VRR <vrr>` 纳入主包，入口为 `starwave.vrr`，根目录安装与 `compile_all.py` 已覆盖 VRR，无需独立安装 `starwave_vrr`。VRR 显式 Torch 支持 CPU/CUDA float32/64、full/checkpoint 与 eager/compile；原生 CUDA 支持 float32 full/boundary。各方程的设备、梯度和边界限制仍独立。

V19 保留 V18 的 23 参数 scalar 契约：任意零宽面组合（正宽度仍须相等）、二维/三维可训练源、可微最终物理波场与诊断前向回调。终态可用于 loss，回调快照为 detached 观测，不能传递 loss 梯度。见 [Scalar Function](usage.md#scalar-function)。

V19 冻结已审阅的集成基线，不改变科学内核或公开签名。根目录安装/构建覆盖 core、elastic、GSLS、Born、VRR 五组原生库。正式示例按方程组织，采用 CUDA/TorchCompile 脚本与 notebook 配对，并包含示例启动及 INR memory 传递修复。源码发布不等于 9.1.0 wheel 身份核验、GPU/多卡、MPS、长程 FWI 收敛或性能验收；下列历史图件、实测教程与哈希保留原证据范围。


## 已发布二进制与历史证据

当前已发布二进制为 V19 / 9.1.0，包含集成 VRR；平台、Python/Torch 条件与命令见[安装](installation.md)。各版本发布回执、哈希、旧平台条件与核验记录已归档至[发布说明](release-notes.md)。V19 源码发布不替代或重标这些历史实测。

## 已完成与待完善

| 内容 | 当前证据与限制 |
|---|---|
| V19 源码 / Release | 冻结 0.1.0.dev19 契约与已发布 V19 Release；源码身份与公开二进制身份分别核验 |
| PyPI 9.1.0 | 已发布；产物身份和 CPU 检查通过，目标设备数值验收仍单独限定 |
| 双语 API 手册 | 七个传播入口；scalar 为 23 参数；集成 VRR、Born 与方程专属后端/梯度/边界契约 |
| 安装 / 导入 | 根目录注册、五组显式构建；导入绑定已有库，不编译或执行运行时身份审计 |
| 源码示例 | 按方程组织 CUDA/TorchCompile notebook/脚本配对；需要匹配的完整源码 |
| 原创教学 notebook | 历史 A30 / 0.1.0.dev9 实测与已核对的产物哈希，不重标为 V19 实验 |
| Scalar3D 示例 | 独立限定的历史源码数值与四卡实验，保留地表采集恢复不完整的结论 |
| 照明 | Usage 已说明公开 helper 与生命周期；设备和梯度限制分别限定 |
| DataParallel / INR | 接线与所选 memory 说明，不推断新增 V19 多卡或收敛验收 |
| GPU / MPS / 长程 FWI / 性能 | 文档、CPU、编译、发布、数值检查分开记录；本次手册更新不增加目标设备验收结论 |

## 参考来源

- [StarWave 9.1.0 PyPI](https://pypi.org/project/starwave/9.1.0/)：公开发行要求与限制。
- [Deepwave API 文档](https://ausargeo.com/deepwave/usage)：参考 API 内容组织，站点保留 Read the Docs 主题，StarWave 说明均为原创。
- [PyFWI 文档](https://pyfwi.readthedocs.io/en/latest/)：仅参考手册导航组织。
- [PyTorch 历史版本安装](https://pytorch.org/get-started/previous-versions/) 与 [DataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)。
- [Microsoft WSL 安装](https://learn.microsoft.com/en-us/windows/wsl/install) 与 [NVIDIA WSL 用户指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。
- [Sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html)、[GitHub Pages 自定义工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) 和 [RTD 配置说明](https://docs.readthedocs.com/platform/stable/config-file/v2.html)。

(tutorial-evidence)=
## 教程结果的来源与复现范围

三份原创 notebook 的服务器回传代码与交付原件一致，执行输出没有错误。已对 39 个运行产物核对 SHA-256 与文件大小，并检查数值数组的形状、有限性、固定边带和最终迭代对应关系。文档编辑过程中没有重新执行 CUDA solver。

- {ref}`安装运行检查 <installation-smoke>`：48 × 40，记录 `[1,12,160]`，有限非零正演与一阶速度梯度。
- [简单梯度计算](modeling/gradient.md)：96 × 64，三炮，固定边带梯度为零；方向导数检查关闭。
- [简单 FWI 示例](inversion/fwi.md)：25 次更新，数据项下降 91.51%，活动区速度 RMSE 仅下降 0.86%。

记录环境为 NVIDIA A30、Python 3.10.18、PyTorch 2.5.1 / CUDA 11.8、StarWave **0.1.0.dev9**。这不证明公开 **2.0.0 wheel** 已被运行，也不根据原 notebook 中的参考开发提交推断实际安装提交。

公开 notebook 是未重新执行的隐私整理版：清空输出、执行计数及机器相关元数据，删除私有源码安装段与两个参考提交来源字段，其余数值、绘图和导出单元保持原样。中文和英文页面提供同一套原始中文教学 notebook；配套 `.py` 与公开 notebook 的代码一致。它们不是把回传 notebook 的完整运行输出直接上网。

可下载 {download}`精确配置与结果摘要 <../examples/tutorials/results_summary.json>`、{download}`原件与公开版哈希及修改范围 <../examples/tutorials/source_manifest.json>`。图件由已核验数组重绘；共享色标、数据源文件哈希和图件哈希见 [图件来源记录](_static/tutorials/figure_manifest.json)。原始系统信息、环境日志和私有路径不公开。
