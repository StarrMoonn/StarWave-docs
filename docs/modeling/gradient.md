# 简单梯度计算

从解析速度模型出发，生成三炮合成观测，计算数据目标对速度的梯度。本例展示完整的 **模型 → 正演 → loss → backward** 链条；不读取外部地震数据，也不依赖其它 notebook 的输出。

```{admonition} 已执行的教学实验 · 2026-10-05
:class: sw-run-note
NVIDIA A30 · Python 3.10.18 · PyTorch 2.5.1 / CUDA 11.8 · StarWave **0.1.0.dev9**。以下图和指标来自服务器实际返回的数组，不是预期效果图。此结果不构成公开 2.0.0 wheel 的设备验收。
```

## 下载与运行

下载 {download}`02_forward_and_gradient.ipynb <../../examples/tutorials/02_forward_and_gradient.ipynb>`，或 {download}`配套 Python 脚本 <../../examples/tutorials/02_forward_and_gradient.py>`。先完成[安装与小规模运行检查](../installation.md)，再选择对应 CUDA kernel，从头运行全部单元。

每份 notebook 都包含模型、几何、Ricker 波形、正演封装、目标函数、绘图和结果导出。公开下载版清空了输出和执行元数据；教学说明保留原始中文，数值计算单元与参数未改变，隐私整理范围见{ref}`来源记录 <tutorial-evidence>`。

## 模型与采集设计

模型按 `[x,z]` 排列，大小 **96 × 64**，网格间距 **10 m**。背景速度由随深度线性增长的项和一个平滑层界面组成：

```{math}
v_0(z)=1600+0.55z+220\,\sigma((z-350)/28).
```

其中速度单位为 m/s，深度为 m，σ 是 sigmoid。真模型在背景上加入中心为 **(500, 260) m** 的正速度高斯透镜，幅度参数 180 m/s、水平/垂向标准差 110/65 m，并乘以在活动区边缘平滑归零的正弦平方窗。180 m/s 是未加窗参数，不是最终异常峰值。

最外侧 6 格、顶部 8 格固定，活动区共 **4,200 格**；真模型与初始模型在固定区完全相同。这里没有水层或自由表面。

```{figure} /_static/tutorials/gradient_models.png
:alt: 解析真模型与不含透镜的初始背景，共享速度色标，横轴为 x，纵轴为深度
:class: sw-science-image
:figclass: sw-science-figure

真模型与初始背景。各图采用相同速度范围；深度向下增加。图像由已核验的数值数组重绘。
```

| 设置 | 本次取值 |
|---|---|
| 炮点 | x = 160、470、790 m；z = 50 m，每炮一个源 |
| 接收点 | x = 100…840 m，间隔 20 m；z = 50 m，共 38 个 |
| 时间 | dt = 1.5 ms，480 个样点，最后一个样点 0.7185 s |
| 源 | 12 Hz Ricker，峰值时刻 0.10 s，固定 forcing；任意幅度单位 |
| 传播 | float32，accuracy=4，PML=20 格，boundary_buffer=5 |
| 内存与速度上界 | memory="boundary"，固定 max_vel=2600 m/s |
| 记录形状 | `[3,38,480]` = `[shot,receiver,time]` |

```{figure} /_static/tutorials/acquisition_source.png
:alt: 活动区、三炮及接收点几何与固定 Ricker 源波形，含米和秒坐标
:class: sw-science-image
:figclass: sw-science-figure

采集点位于浅部固定区。源波形是固定 forcing；不要再手动乘以传播器内部的 −v² dt² 注入系数。记录幅度不是校准后的 Pa。
```

## 目标函数与自动微分

观测由同一个 solver、相同网格上的真模型生成，随后固定。以全部观测的 RMS **S = 1.9230548143** 同时缩放预测和观测，不能分别归一化二者或逐道归一化。

```{math}
r=(d_{\mathrm{pred}}-d_{\mathrm{obs}})/S,
\qquad J_{\mathrm{data}}=\operatorname{mean}\ell_{0.25}(r).
```

```{math}
\ell_\beta(r)=
\begin{cases}
 r^2/(2\beta),& |r|<\beta,\\
 |r|-\beta/2,& |r|\ge\beta.
\end{cases}
```

这是 PyTorch `SmoothL1Loss(beta=0.25)`，等于 `HuberLoss(delta=0.25) / 0.25`。参数是速度本身，固定区通过 `torch.where` 排除。每炮重新建图，炮损失除以炮数后各反传一次，最终梯度对应等尺寸三炮数据的全局平均：

```python
# 完整定义见可下载 notebook；observed 和 scale 已固定。
v_parameter = torch.nn.Parameter(v_initial.clone())
for s in range(NSHOTS):
    model = torch.where(active, v_parameter, v_initial)
    predicted = propagate(
        model, source_amplitudes[s:s+1], source_locations[s:s+1],
        receiver_locations[s:s+1], CONFIG,
    )
    term = data_loss(predicted, observed[s:s+1], scale, CONFIG) / NSHOTS
    term.backward()
gradient = v_parameter.grad
```

本例只求梯度，不执行优化器更新；梯度的符号描述增大局部速度对目标函数的一阶影响，不等于最终速度修正。

## 实测记录与梯度

```{figure} /_static/tutorials/gradient_gathers.png
:alt: 第二炮观测、初始预测和预测减观测残差，共享完整的零对称振幅色标
:class: sw-science-image
:figclass: sw-science-figure

第二炮（x = 470 m）的观测、初始预测和残差。三图共享完整、零对称振幅范围，不独立归一化或截幅；初始数据差异较小，因此残差在这个尺度上较淡。
```

```{figure} /_static/tutorials/gradient_velocity.png
:alt: 目标函数对速度的带符号梯度，全范围零对称色标，固定边带梯度为零
:class: sw-science-image
:figclass: sw-science-figure

带符号速度梯度，采用完整的零对称色标。局部较弱结构没有被另行放大，固定区梯度为零。
```

| 指标 | 实测值 |
|---|---:|
| 归一化 SmoothL1 数据目标 | 2.47999068 × 10⁻⁴ |
| 相对数据 L2，‖预测−观测‖₂ / ‖观测‖₂ | 0.0111355064 |
| 速度梯度 L2 范数 | 2.01557941 × 10⁻⁶ |
| 活动区最大绝对梯度 | 1.27708390 × 10⁻⁷ |
| 固定区最大绝对梯度 | 0 |

## 结论与边界

本次完成了有限、非零的正演和一阶速度梯度，且固定区梯度为零。**方向导数检查关闭，未提供有限差分或完整伴随正确性的证据。** 同 solver 合成数据排除了噪声、源不确定性与建模误差，因此不能代表真实资料效果。

外缘扩展链与 PML 的已知限制仍适用，见[模型约定](conventions.md)。固定边带是一项实验约束，不表示修复了这些限制。下一步在相同问题上运行[简单 FWI 示例](../inversion/fwi.md)，同时观察数据拟合和模型恢复。
