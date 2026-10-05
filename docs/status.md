# 文档状态与参考来源

## 已完成与待完善

| 内容 | 当前状态 |
|---|---|
| 中英双语导航、安装与 WSL 指引 | 已完整翻译；另附 A30 / 0.1.0.dev9 的小规模运行记录，公开 wheel 仍待实测 |
| scalar / VRZ / VTI API 参考 | 单页 Usage；54 项参数的签名、默认值、类型、shape/单位与约束已核对 |
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

API 核对基于发布 wheel 的 Python 接口签名与授权使用说明；文档重新组织为入门内容，不包含软件实现。公开 wheel SHA-256：

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
