# 关于 StarWave

StarWave 是面向 PyTorch 的可微分波传播工具，用于连接地下介质模型、模拟观测与模型更新。CUDA 负责波传播计算，PyTorch 的自动微分将损失函数的梯度传回模型，让正演与反演可以使用熟悉的张量和优化器接口。

## 能做什么

- **波传播模拟**：支持二维标量声学、二维 VRZ，以及二维/三维声学 VTI。不同方程使用各自的介质参数、震源与接收量约定。
- **梯度与 FWI**：将模拟记录接入 PyTorch 损失函数，计算模型的一阶梯度，并在用户定义的优化循环中更新模型。
- **神经网络模型表示**：将用户网络生成的完整模型传给传播接口，探索隐式神经表示（INR）与波动物理的结合。

当前传播接口使用 CUDA float32 模型；震源波形与采集位置为固定输入。完整的类型、形状、单位和限制见 [Usage](usage.md)。

## 来源与发展

StarWave 的 CUDA 传播内核基于 MIT 许可的 [SWEEP](https://github.com/DeepWave-KAUST/sweep) 代码进行适配，并保留上游署名与许可。在这一基础上，StarWave 围绕统一的 PyTorch 接口、模型梯度处理、输入检查和反演实验流程做了适配与改进。

StarWave 的调用风格参考了 [Deepwave](https://github.com/ar4/deepwave)；部分时间重采样工具参考并适配了 Deepwave，验证工作也使用了其参考代码。相关代码保留适用的 MIT 许可。[Deepwave 的 Usage 文档](https://ausargeo.com/deepwave/usage)也为本手册的组织方式提供了参考。StarWave 的方程范围、参数约定与返回值以本手册为准。

## 从哪里开始

1. [安装](installation.md)：准备 Python、PyTorch 与 CUDA 环境，完成运行检查。
2. [快速入门](quickstart.md)：了解从模型、采集设置到合成炮集的完整流程。
3. [梯度教程](modeling/gradient.md)与 [FWI 教程](inversion/fwi.md)：把观测差异连接到模型更新。
4. [INR](inversion/inr.md)：了解如何接入自己的神经网络模型。

教程中的实测结果来自 A30 / StarWave 0.1.0.dev9 的特定标量声学实验；公开 2.0.0 wheel 的设备验证、完整伴随检查、性能和多卡验证仍待完成。INR 页面提供概念与接线示例。各项证据与适用范围见[文档状态](status.md)。
