# FWI 入门

全波形反演（FWI）用模型产生预测记录，再对预测与观测之间的目标函数求模型梯度。本节演示 scalar 的一次参数更新；它是接线示例，不是完整收敛实验。

## 运行一次更新

使用[快速入门](../quickstart.md)中的相同脚本：

```bash
python scalar_demo.py --mode fwi --device 0
```

脚本先用合成真模型产生固定观测，再用均匀初始速度预测。计算以观测能量归一化的均方误差，执行一次 `loss.backward()` 和 Adam 更新。学习率、归一化、固定边缘与更新后的速度区间都是本例显式选择，不是 StarWave 的默认策略。

## 循环要点

1. 将可训练模型放进 `torch.nn.Parameter`；源波形、源坐标和接收坐标保持固定。
2. 每次更新前清空梯度，每次迭代重新 forward。
3. 从返回的记录构造标量 loss，每次 forward 只 backward 一次。
4. 检查 loss 和梯度的有限性，再更新模型；记录用户选择的约束和优化参数。

```python
# model、observed、inputs 与 propagate 已由 scalar_demo.py 准备。
optimizer.zero_grad(set_to_none=True)
predicted = propagate(model, inputs)
loss = torch.nn.functional.mse_loss(predicted, observed)
loss.backward()
optimizer.step()
```

真实数据还需要核对源时序、分量、单位、预处理和采集几何；不能直接把 VTI 的 `vz` 观测与 scalar pressure-like 输出比较。VRZ 的两种介质参数化也会改变优化变量与梯度含义。

## 验证边界

先做小规模目标设备测试、内部模型扰动的有限差分或方向导数检查，再考虑长时序、多炮与多卡。边缘梯度受[已知扩展链限制](../modeling/conventions.md)约束；固定边缘不能被描述为已修复整个梯度问题。

本节尚缺目标 GPU 的误差表、步长选择依据和完整收敛曲线，不能由一次 loss 或 smoke check 推断科学验收通过。
