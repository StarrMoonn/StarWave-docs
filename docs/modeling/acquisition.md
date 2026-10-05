# 观测系统

观测系统把震源、波形和接收点连接到物理模型。本页用一个 32×32 网格，构造**两炮、每炮一个震源、每炮七个接收点**，再将这些输入交给 `starwave.scalar`。

## 炮、震源与接收点

一炮是一次独立的传播实验。一个批次可以包含多炮；同一炮内也可以有多个震源，它们各自对应一条波形，共同参与该炮传播。

| 张量 | 二维形状 | 本例 |
|---|---|---|
| `source_locations` | `[B,S,2]` | `[2,1,2]` |
| `source_amplitudes` | `[B,S,T]` | `[2,1,256]` |
| `receiver_locations` | `[B,R,2]` | `[2,7,2]` |
| 返回的记录张量 | `[B,R,T]` | `[2,7,256]` |

`B` 是炮数，`S` 是每炮震源数，`R` 是每炮接收点数，`T` 是时间采样数。单源时，scalar 也接受 `[B,2]` 的震源位置；本例保留显式源维。同一炮内重合源的增量相加，重复接收点也被允许。

## 坐标与物理网格

位置是物理模型的**整数网格下标**，建议使用 `torch.long`；不是米坐标，也不包含 PML 偏移。下标必须落在模型范围内。

本例主动选择 `v.shape == (nx,nz)`，第 0 轴为 x、第 1 轴为 z，因此位置 `[10,10]` 直接对应 `v[10,10]`。这是本例的轴序，scalar 不会自动判断物理方向。若原点为 `(0 m,0 m)`、网格间距为 10 m，该位置对应 `(100 m,100 m)`；非网格节点的位置应先明确离散化方案，不要靠整数转换悄悄截断。

```{figure} /_static/diagrams/acquisition.svg
:alt: 32×32 物理网格，第 0 轴 x 向右，第 1 轴 z 向下。第 0 炮震源在 (10,10)，第 1 炮震源在 (22,10)。两炮各有七个接收点，位于 z=6、x=4,8,12,16,20,24,28。
:class: sw-science-figure

两炮共用相同接收排列。图中并列标出两炮震源以便比较位置；每炮只激发自己的一个震源。所有数字均为网格下标，图中不含 PML。
```

通用的轴序、采样与边界规则见[模型、采集与时间约定](conventions.md)。

## 构造几何与波形

下面仅用 PyTorch 在 CPU 上构造输入；不会执行波传播。两炮共享接收排列和一个显式定义的 Ricker 波形，使用 `repeat` 创建各炮自己的张量条目。

```python
import math
import torch

nx, nz = 32, 32
B, S, R, T = 2, 1, 7, 256
grid_spacing, dt, frequency = 10.0, 0.001, 15.0

# Chosen axis order: [x, z]. Each shot has one source.
source_locations = torch.tensor(
    [[[10, 10]], [[22, 10]]], dtype=torch.long
)

# The same receiver line is explicitly repeated for both shots.
receiver_x = torch.arange(4, 29, 4, dtype=torch.long)
receiver_line = torch.stack(
    (receiver_x, torch.full_like(receiver_x, 6)), dim=-1
)
receiver_locations = receiver_line[None, :, :].repeat(B, 1, 1)

# Fixed scalar forcing waveform, with time in seconds.
time = torch.arange(T, dtype=torch.float32) * dt
a = (math.pi * frequency * (time - 0.08)) ** 2
wavelet = (1.0 - 2.0 * a) * torch.exp(-a)
source_amplitudes = wavelet[None, None, :].repeat(B, S, 1)

assert source_locations.shape == (B, S, 2)
assert receiver_locations.shape == (B, R, 2)
assert source_amplitudes.shape == (B, S, T)
for locations in (source_locations, receiver_locations):
    assert torch.all((0 <= locations[..., 0]) & (locations[..., 0] < nx))
    assert torch.all((0 <= locations[..., 1]) & (locations[..., 1] < nz))
```

对 scalar，`source_amplitudes` 是固定的归一化 forcing `f`，不是每步压力增量；不要额外乘以 `-v² dt²`。本例的波形幅值只是演示选择。源单位、时间处理及其它传播函数的差异见 [Usage](../usage.md)。

## 接入传播函数

先按[安装说明](../installation.md)与[快速入门](../quickstart.md)准备 CUDA 设备和原生库，再接着运行以下片段。它复用上方变量，并将模型、坐标和波形放到当前 CUDA 设备。

```python
import starwave

device = torch.device("cuda")
v = torch.full((nx, nz), 1800.0, dtype=torch.float32, device=device)

with torch.no_grad():
    records, = starwave.scalar(
        v, grid_spacing=grid_spacing, dt=dt,
        source_amplitudes=source_amplitudes.to(device),
        source_locations=source_locations.to(device),
        receiver_locations=receiver_locations.to(device),
        pml_freq=frequency,
    )

assert records.shape == (B, R, T)
```

`records[b,r,:]` 是第 `b` 炮第 `r` 个接收点的时间序列。本例预期形状为 `(2,7,256)`；这是接口契约的预期值，本页不提供该两炮例子的 GPU 实测结果。完整参数、默认值和运行限制以 [Usage](../usage.md) 为准。
