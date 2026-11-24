# 快速入门：一个合成炮集

[V19 Release](https://github.com/StarrMoonn/StarWave/releases/tag/V19) 已发布；本手册对应冻结源码 `0.1.0.dev19`、提交 `46d32877e3336d531520a696d5f2e074e6f4ef74`。[PyPI **9.1.0**](https://pypi.org/project/starwave/9.1.0/) 已发布，对应 V19 源码契约并包含集成 VRR。现有 [9.0.0 wheel](https://pypi.org/project/starwave/9.0.0/) 仍对应 V18，不含集成 VRR。

下方二维 scalar 入门接线也兼容历史 7.1.0 基线。当前三维速度/源梯度见{ref}`三维 scalar 调用 <scalar-3d-example>`；[Born](usage.md#born-function)另有独立 CPU 示例。

目标是检查从模型、震源、采集几何到记录张量的接线。示例现场创建 32×32 的小速度模型、一炮一源和 16 个接收点；不下载数据，不保存结果，不执行编译。

先按[安装](installation.md)准备匹配的原生库。V19 导入即准备已有原生 dispatch，脚本中的显式 prepare 在该版本是可选设备检查，对 V16 wheel 仍有用；导入不会编译或传播。再下载 {download}`scalar_demo.py <../examples/scalar_demo.py>`。

```bash
python scalar_demo.py --mode forward --device 0
```

若使用本文档项目的源码目录：

```bash
python examples/scalar_demo.py --mode forward --device 0
```

## 核心代码

```{literalinclude} ../examples/scalar_demo.py
:language: python
:pyobject: acquisition
```

源信号为本例显式定义的 Ricker 波形；它不调用不存在的 StarWave wavelet 工具。网格轴序选为 `[x,z]`，几何坐标与之对应。

```{literalinclude} ../examples/scalar_demo.py
:language: python
:pyobject: propagate
```

## 如何判断执行状态

脚本期望得到 `(1, 16, 256)` 的记录，并检查有限、非全零。该形状是输入契约的预期值；这个独立 32×32 脚本没有提供实测结果；原创 notebook 的[安装运行检查](installation.md)与[梯度实验](modeling/gradient.md)另有实测记录。finite/nonzero smoke check 也不能证明解或梯度正确。

如遇加载错误先看[常见问题](faq.md)。如需反演，保持相同源单位、网格与采集，继续 [FWI 入门](inversion/fwi.md)。
