# INR：用隐式网络表示模型

INR（Implicit Neural Representation）让网络从固定空间坐标预测模型值，再通过传播误差训练网络参数。传播 API 仍接收完整 CUDA float32 模型；INR 不是 StarWave 内置传播方程或公开网络类。

## 单卡计算图

```text
固定坐标 → 用户网络 → 完整速度模型 → starwave.scalar → 记录 → loss
               ↑                     模型梯度经 autograd 返回
```

以下是接线片段，假设 `network`、`coords`、采集张量和 `observed` 已准备在同一 CUDA 设备上，`coords` 含 `nx*nz` 个坐标，网络每个坐标输出一个值：

```python
optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
optimizer.zero_grad(set_to_none=True)
raw = network(coords).reshape(nx, nz)
velocity = 1500.0 + 1000.0 * torch.sigmoid(raw)
prediction = starwave.scalar(
    velocity, 10.0, 0.001,
    source_amplitudes=amplitudes,
    source_locations=sources,
    receiver_locations=receivers,
    pml_freq=15.0,
)[0]
loss = torch.nn.functional.mse_loss(prediction, observed)
loss.backward()
optimizer.step()
```

1500–2500 m/s 的映射和学习率仅为示意。不要把 `velocity` 用 `detach()` 或新建 `nn.Parameter` 再包一次，否则会切断到网络的计算图。只训练网络参数；固定震源波形不能借由 INR 变成可求导的 StarWave 波形输入。

## 与多卡结合

若使用 DataParallel，将网络放入 Module，固定坐标注册为 buffer；每个副本在 forward 中计算完整模型，只切分采集炮维。网络参数梯度由 PyTorch 汇总。先验参数化、坐标归一化、学习率、网络结构和观测目标均属于实验设计。

本页是概念与接线入门。完整独立网络脚本、单/多卡数值对照、网络依赖版本、真实 FWI 收敛与性能结果仍待补齐；不附带私人 notebook、工具实现或数据集。
