# 文档状态与参考来源

## 已完成与待完善

| 内容 | 当前状态 |
|---|---|
| 中英双语导航、安装与 WSL 指引 | 已完整翻译；目标设备安装尚未实测 |
| scalar / VRZ / VTI API 参考 | 独立函数页；54 项参数的签名、默认值、类型、shape/单位与约束已核对 |
| 合成 scalar 正演与一次 FWI 更新脚本 | 已编写、语法检查；GPU 运行待验收 |
| VRZ / VTI | 使用约定与调用片段已提供；完整独立示例待补 |
| DataParallel | 按炮拆分接线片段；多 GPU 对照待测 |
| INR | 概念与接线片段；独立网络程序及收敛实验待补 |
| 照明 API | 仅列导出名称；完整生命周期教程待补 |
| 数值与性能证据 | 不提供未经实测的误差表、加速比或收敛结论 |

## 参考来源

- [StarWave 2.0.0 PyPI](https://pypi.org/project/starwave/2.0.0/)：公开发行要求与限制。
- [Deepwave API 文档](https://ausargeo.com/deepwave/usage)：参考 API 组织与 Sphinx/Alabaster 呈现，StarWave 说明均为原创。
- [PyFWI 文档](https://pyfwi.readthedocs.io/en/latest/)：仅参考手册导航组织。
- [PyTorch 历史版本安装](https://pytorch.org/get-started/previous-versions/) 与 [DataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)。
- [Microsoft WSL 安装](https://learn.microsoft.com/en-us/windows/wsl/install) 与 [NVIDIA WSL 用户指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。
- [Sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html)、[GitHub Pages 自定义工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) 和 [RTD 配置说明](https://docs.readthedocs.com/platform/stable/config-file/v2.html)。

API 核对基于发布 wheel 的 Python 接口签名与授权使用说明；文档重新组织为入门内容，不包含软件实现。公开 wheel SHA-256：

```text
6622b863c76db1ba708622048377f3de4447609ff7295d1705b8145884ce06db
```

最后核对日期：2026-10-05。参考资料更新后应按实际发行版本重新审阅安装条件与 API。
