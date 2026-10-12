# 关于 StarWave

StarWave 是面向 PyTorch 的可微分波传播工具，用于连接地下介质模型、模拟观测与模型更新。受支持的原生 CPU/CUDA 路径与显式 PyTorch 后端负责传播，自动微分将损失函数的梯度传回模型，让正演与反演可以使用熟悉的张量和优化器接口。

可在线阅读或下载 [30 页原创演示文稿](presentation.md)，了解 FWI、自动微分与 StarWave 的整体工作流程。

## 能做什么

- **波传播模拟**：覆盖二维/三维标量声学、一维/二维/三维声学 Born、二维 VRZ、二维/三维声学 VTI 与弹性、二维 GSLS 黏声。独立 VRR 扩展提供二维矢量反射率。各方程有自己的模型、震源、记录和边界约定。
- **梯度与 FWI**：将模拟记录接入 PyTorch 损失函数，计算模型的一阶梯度，并在用户定义的优化循环中更新模型。
- **神经网络模型表示**：将用户网络生成的完整模型传给传播接口，探索隐式神经表示（INR）与波动物理的结合。

原生 scalar、VRZ、VTI 使用 CUDA float32；显式 Torch 路径支持 CPU/CUDA，限制依方程区分。二维/三维 scalar 支持一阶速度梯度；仅三维 scalar 支持震源波形的一阶梯度，二维源仍须固定。采集位置不可微，其他传播器的梯度范围见各自约定。完整的类型、形状、单位和限制见 [Usage](usage.md)。

本手册面向已发布 V17 源码（`0.1.0.dev17`），包含 [Born](usage.md#born-function)、方程专属平顶边界与导入时原生准备。**PyPI 8.0.0 尚未发布，目前的 7.1.0 wheel 仍为 V16。** Born 使用速度扰动 δv，返回背景/散射状态与记录，其背景算子不同于 `starwave.scalar`。匹配源码/构建要求见[安装](installation.md)。

## 来源与发展

StarWave 的 CUDA 传播内核基于 MIT 许可的 [SWEEP](https://github.com/DeepWave-KAUST/sweep) 代码进行适配，并保留上游署名与许可。在这一基础上，StarWave 围绕统一的 PyTorch 接口、模型梯度处理、输入检查和反演实验流程做了适配与改进。

StarWave 的调用风格参考了 [Deepwave](https://github.com/ar4/deepwave)；部分时间重采样工具参考并适配了 Deepwave；elastic 与 Born 源自 Deepwave 0.0.27，保留署名并分别记录适配契约；验证工作也使用了其参考代码。相关代码保留适用的 MIT 许可。[Deepwave 的 Usage 文档](https://ausargeo.com/deepwave/usage)也为本手册的组织方式提供了参考。StarWave 的方程范围、参数约定与返回值以本手册为准。

## 从哪里开始

1. [安装](installation.md)：准备 Python、PyTorch 与 CUDA 环境，完成运行检查。
2. [快速入门](quickstart.md)：了解从模型、采集设置到合成炮集的完整流程。
3. [梯度教程](modeling/gradient.md)与 [FWI 教程](inversion/fwi.md)：把观测差异连接到模型更新。
4. [INR](inversion/inr.md)：了解如何接入自己的神经网络模型。

教程实测来自 A30 / StarWave 0.1.0.dev9 的特定标量声学实验。历史 wheel 接口核对、已核验的 7.1.0 发布记录与当前 V17 源码文档分别限定范围，不能把教程结果转为新 wheel 或 V17 的 GPU 验收。INR 页面提供概念与接线示例，证据与限制见[文档状态](status.md)。