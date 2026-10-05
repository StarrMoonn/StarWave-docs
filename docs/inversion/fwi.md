# 简单 FWI 示例

用三炮合成数据完成 **25 次 Adam 更新**，同时检查数据拟合与模型恢复。本页沿用[简单梯度计算](../modeling/gradient.md)的解析模型和采集，展示一次实际运行的全过程。

```{important}
数据目标下降 **91.51%**，但活动区速度 RMSE 只改善 **0.86%**（30.905 → 30.639 m/s）。中心透镜仍未充分恢复。完成优化循环不等于 FWI 收敛或科学验收通过。
```

```{admonition} 本次结果的环境范围
:class: sw-run-note
2026-10-05 · NVIDIA A30 · Python 3.10.18 · PyTorch 2.5.1 / CUDA 11.8 · StarWave **0.1.0.dev9**。本页不是公开 2.0.0 wheel、其它 GPU 或多 GPU 的验证报告。
```

## 下载与运行

下载 {download}`03_simple_fwi.ipynb <../../examples/tutorials/03_simple_fwi.ipynb>`，或 {download}`配套 Python 脚本 <../../examples/tutorials/03_simple_fwi.py>`。从已完成[安装检查](../installation.md)的 CUDA kernel 开始，重新从头运行全部单元；本例独立生成所有输入，无需先运行其它 notebook。

公开下载版保留原始中文教学说明，清空历史输出与执行元数据，仅作隐私与说明整理。完整参数与来源记录可下载：{download}`results_summary.json <../../examples/tutorials/results_summary.json>` · {download}`source_manifest.json <../../examples/tutorials/source_manifest.json>`。

## 实验配置

| 项目 | 本次设置 |
|---|---|
| 模型与活动区 | 96 × 64，10 m 网格；4,200 格可训练，外侧 6 格及顶部 8 格固定 |
| 观测 | 同 solver 真模型合成；3 炮 × 38 道 × 480 样点，无外加噪声 |
| 采集与源 | 源 x=160/470/790 m，接收 x=100…840 m，均在 z=50 m；12 Hz Ricker |
| 时间与传播 | dt=1.5 ms；accuracy=4，PML=20，boundary_buffer=5，memory="boundary" |
| 速度约束 | 1450 < v < 2450 m/s；固定 max_vel=2600 m/s |
| 优化器 | Adam，学习率 0.025，25 次更新；每次累积全部三炮 |
| 正则化 | 改变量的网格一阶差分平方，权重 10⁻⁴，改变量尺度 100 m/s |
| 梯度裁剪 | 潜变量 θ 的全局范数，阈值 1.0 |
| 完整评估 | 更新 0、5、10、15、20、25 后 |

初始模型是平滑分层背景，真模型另有加窗高斯透镜。模型公式、源单位与几何见[梯度示例](../modeling/gradient.md)。真模型只用于生成观测和计算诊断误差，未参与优化更新或选择“最佳”模型。

## 循环要点

活动区的无约束潜变量 θ 通过 sigmoid 映射为物理速度；固定区直接取初始背景：

```{math}
v(\theta)=1450+1000\,\sigma(\theta)
\quad\text{(active cells)}.
```

初始化 `theta = logit((v_initial − 1450) / 1000)`。数据项与[梯度示例](../modeling/gradient.md)相同：观测 RMS 只计算一次，预测和观测使用共同尺度，取 beta=0.25 的 SmoothL1 均值。

```{math}
u=(v-v_0)/(100\ \mathrm{m/s}),
\qquad R=\operatorname{mean}[(\Delta_x u)^2]+
          \operatorname{mean}[(\Delta_z u)^2].
```

```{math}
J(\theta)=J_{\mathrm{data}}(v(\theta))+10^{-4}R.
```

这里 Δ 是相邻网格点差，不除以网格间距；R 是教学示例显式选择的改变量平滑项，不是传播器的默认处理。

```python
# 完整类、函数和输入由 notebook 定义。
optimizer.zero_grad(set_to_none=True)
for s in range(NSHOTS):
    model = velocity()  # 每炮重新建立模型与传播图
    predicted = propagate(
        model, source_amplitudes[s:s+1], source_locations[s:s+1],
        receiver_locations[s:s+1], CONFIG,
    )
    term = data_loss(predicted, observed[s:s+1], scale, CONFIG) / NSHOTS
    term.backward()
penalty = CONFIG["smoothness_weight"] * smoothness(velocity(), v_initial, CONFIG)
penalty.backward()
torch.nn.utils.clip_grad_norm_(velocity.parameters(), 1.0, error_if_nonfinite=True)
optimizer.step()  # 累积全部三炮后，才执行一次更新
```

每个 native forward 只对应一次 backward；每炮都重建图，正则项另建图反传。完整 notebook 在更新前后检查有限性、边带不变性和速度范围；出现异常立即停止，不用 NaN 替换或删点掩盖失败。

## 数据拟合与模型误差

```{figure} /_static/tutorials/fwi_history.png
:alt: 六个更新后评估点的数据目标、总目标与活动区速度 RMSE；RMSE 先升高再回落
:class: sw-science-image
:figclass: sw-science-figure

六个完整评估状态，横轴为已完成的优化更新次数。曲线连接实测点，不代表中间状态的测量；没有用各自初值把曲线缩放为同一个起点。模型 RMSE 在第 10 次更新时增至约 32.636 m/s，随后回落。
```

| 指标 | 初始（0 次） | 最终（25 次） | 变化 |
|---|---:|---:|---:|
| 归一化 SmoothL1 数据项 | 2.47999065 × 10⁻⁴ | 2.10452727 × 10⁻⁵ | 下降 91.51% |
| 数据项 + 加权正则项 | 2.47999065 × 10⁻⁴ | 2.14042969 × 10⁻⁵ | 分开记录数据项与总目标 |
| 相对数据 L2 | 0.0111355064 | 0.0032438610 | 下降 70.87% |
| 活动区速度 RMSE（m/s） | 30.9053688 | 30.6389942 | 仅下降 0.2663745（0.86%） |
| 固定边带最大速度变化（m/s） | 0 | 0 | 保持固定 |

相对数据 L2 为 ‖预测−观测‖₂/‖观测‖₂。模型 RMSE 只在 4,200 个活动格点上相对合成真模型计算；真实资料通常没有这个真值指标。表中最终记录、最终模型和第 25 次评估对应同一状态，不是按真值误差筛选的迭代。

## 恢复模型与残差

```{figure} /_static/tutorials/fwi_models.png
:alt: 真模型、初始背景与第 25 次更新后模型，共用一个物理速度范围
:class: sw-science-image
:figclass: sw-science-figure

真模型、初始与最终模型使用相同速度范围。主背景相近不代表透镜已恢复；请结合下面的模型误差图阅读。
```

```{figure} /_static/tutorials/fwi_model_errors.png
:alt: 初始减真值与最终减真值模型误差，共享完整零对称色标
:class: sw-science-image
:figclass: sw-science-figure

模型误差 = 当前速度 − 真值。两图共用完整、零对称色标，未截幅或单图归一化；中心透镜仍有明显误差。
```

```{figure} /_static/tutorials/fwi_residuals.png
:alt: 第一炮的初始与最终数据残差，使用三炮共用的残差专用对称色标
:class: sw-science-image
:figclass: sw-science-figure

第一炮（源 x = 160 m）。数据残差 = 预测 − 观测。下列全部三炮、初始/最终共享完整残差范围；这是残差专用色标，与梯度页的完整炮集振幅色标不同。不能用两页颜色深浅直接比较振幅。
```

```{figure} /_static/tutorials/fwi_residuals_shot_1.png
:alt: 第二炮初始与最终残差，统一残差色标
:class: sw-science-image
:figclass: sw-science-figure

三炮残差共用上述同一色标；此图为第二炮（源 x = 470 m）。
```

```{figure} /_static/tutorials/fwi_residuals_shot_2.png
:alt: 第三炮初始与最终残差，统一残差色标
:class: sw-science-image
:figclass: sw-science-figure

第三炮（源 x = 790 m），仍使用同一完整残差色标。
```

## 验证边界

这次运行证明了该环境下三炮合成问题的优化流程可以完成，数据拟合有明显改善；它只提供很弱的模型恢复改善，**不能宣称获得高质量反演或已收敛**。

- 方向导数/有限差分检查未执行，完整伴随正确性未验证。
- 同 solver、同网格、无外加噪声，未覆盖真实数据、源误差或建模误差。
- 单频带、有限采集与 25 次更新限制解释范围；本实验没有分离各因素的影响。
- 固定外缘不修复[已知扩展链/PML 限制](../modeling/conventions.md)，没有测试多 GPU、AMP 或高阶导数。
- 不提供 GPU 耗时或加速比；保存的模型快照未含 Adam 动量，不是可精确续跑的 checkpoint。

## 保留的一次更新检查

原有小规模接线脚本仍可使用，旧链接继续有效。它是另一套 32 × 32 微型配置，不是上述三炮实验的结果来源：

```bash
python scalar_demo.py --mode fwi --device 0
```

下载入口在[快速入门](../quickstart.md)。进一步实验应先补充内部扰动的步长扫描与方向导数检查，再分别改变频率、采集或优化参数；这些扩展尚未在本页执行。
