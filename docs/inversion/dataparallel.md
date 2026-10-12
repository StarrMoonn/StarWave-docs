# 按炮拆分 DataParallel

多卡接线的关键是：**完整模型由 Module 持有，只沿炮维拆分采集张量**。将二维模型当作普通 forward 输入会被 DataParallel 沿第一维切开，破坏模型的空间域。

## 最小接线片段

以下片段需要已有的有效 CUDA 模型、至少两炮采集和观测，不是独立运行程序。`ScalarForward` 是本教程定义的类，不是 `starwave.Scalar` API。

```python
import torch
import starwave

class ScalarForward(torch.nn.Module):
    def __init__(self, velocity):
        super().__init__()
        self.velocity = torch.nn.Parameter(velocity)

    def forward(self, amplitudes, sources, receivers):
        return starwave.scalar(
            self.velocity, 10.0, 0.001,
            source_amplitudes=amplitudes,
            source_locations=sources,
            receiver_locations=receivers,
            pml_freq=15.0,
        )[0]

device_ids = [0, 1]  # 替换成当前进程可见的逻辑编号。
starwave.prepare_native(device_ids)  # V19 可选设备检查；V16 使用显式准备。
primary = torch.device(f"cuda:{device_ids[0]}")
module = ScalarForward(initial_velocity.to(primary))
parallel = torch.nn.DataParallel(module, device_ids=device_ids, dim=0)
optimizer = torch.optim.Adam(module.parameters(), lr=1.0)
optimizer.zero_grad(set_to_none=True)
prediction = parallel(
    amplitudes.to(primary), sources.to(primary), receivers.to(primary)
)
loss = torch.nn.functional.mse_loss(prediction, observed.to(primary))
loss.backward()
optimizer.step()
```

这里的三个输入分别为 `[B,S,T]`、`[B,S,D]`、`[B,R,D]`，同一调用使用相同 `B`。每个副本获得完整模型和部分炮；记录汇总到首卡后计算一次全局目标函数。固定的其它模型应使用 buffer 或显式固定参数持有。

V19 导入在工作线程调用前准备已有原生库与共享 dispatch，不初始化 CUDA，也不运行传播。正常首次并行传播不再要求手动 prepare 或串行预热；如保留显式设备检查，应在主线程调用。先构建再导入，更新后重启。缺少原生构建时须选择受支持的显式 Torch 路径；保留的无效构建会报错。V16 / 7.1.0 wheel 仍使用较早的显式准备流程。

改接 Born 时也要区分模型与炮维：`v`、`scatter` 由复制的 Module 持有，源及背景/散射接收几何一致地按炮拆分。Born 返回终态和两个记录张量，wrapper 应显式选择所需记录。见 [Born Function](../usage.md#born-function)；本页不构成 Born 多卡验收。

## 使用前检查

`CUDA_VISIBLE_DEVICES` 会重映射逻辑编号。先确认 `torch.cuda.device_count()`，并让模型、观测与首卡一致。多卡不能自动解决单炮显存不足，炮数太少也可能让部分卡空闲。

该片段已做语法与 API 调用形状审阅，尚未在多 GPU 上验收。应比较相同输入下单卡/多卡记录、全局 loss 和模型梯度；不应声称 DDP 已支持或多卡已获得线性加速。参考 [PyTorch DataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)。
