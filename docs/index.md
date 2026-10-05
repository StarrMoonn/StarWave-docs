# StarWave 使用手册

```{raw} html
<div class="sw-home-hero">
  <div class="sw-home-intro">
    <p class="sw-home-kicker">可微分波传播 · PyTorch · CUDA</p>
    <h2>让波动物理融入<br><span>可微分计算。</span></h2>
    <p class="sw-home-lead">从一次波场模拟，到一次模型更新。StarWave 将波传播与 PyTorch 的自动微分连接起来，让正演、全波形反演与神经网络模型表示，在同一条计算链路中协同工作。</p>
    <nav class="sw-home-actions" aria-label="开始使用 StarWave">
      <a class="sw-button sw-button-primary" href="installation.html">安装 StarWave <span aria-hidden="true">↗</span></a>
      <a class="sw-button" href="quickstart.html">快速入门 <span aria-hidden="true">→</span></a>
      <a class="sw-home-api-link" href="usage.html">Usage / API <span aria-hidden="true">→</span></a>
    </nav>
  </div>
  <div class="sw-home-identity">
    <img class="sw-home-logo" src="_static/brand/starwave-main.svg" alt="StarWave" width="950" height="645" fetchpriority="high">
    <p>SEE A DEEPER EARTH</p>
  </div>
</div>
```

(home-capabilities)=
## 从物理模型，到可学习的模型

```{raw} html
<div class="sw-home-capabilities">
  <div>
    <span class="sw-home-number" aria-hidden="true">01 / PROPAGATE</span>
    <h3>用波场，连接模型与观测</h3>
    <p>CUDA 波传播覆盖二维标量声学、二维 VRZ 与二维/三维声学 VTI，为不同介质参数化提供清晰的建模入口。</p>
    <a href="usage.html#propagators">探索传播接口 <span aria-hidden="true">→</span></a>
  </div>
  <div>
    <span class="sw-home-number" aria-hidden="true">02 / DIFFERENTIATE</span>
    <h3>让梯度，参与每一次更新</h3>
    <p>将模拟记录接入 PyTorch 损失函数，经 autograd 回传模型梯度。沿用熟悉的优化器，构建自己的 FWI 实验。</p>
    <a href="modeling/gradient.html">查看梯度实验 <span aria-hidden="true">→</span></a>
  </div>
  <div>
    <span class="sw-home-number" aria-hidden="true">03 / CONNECT</span>
    <h3>让模型表示，有更多可能</h3>
    <p>模型可以来自网格参数，也可以来自用户定义的隐式神经表示（INR）。通过可微分链路，探索物理与学习的结合。</p>
    <a href="inversion/inr.html">了解 INR 实验接线 <span aria-hidden="true">→</span></a>
  </div>
</div>
```

(home-workflow)=
## 熟悉的 PyTorch，面向波动问题

正演产生记录，损失衡量差异，梯度连接模型。以下片段展示核心接线；完整环境与采集设置见[快速入门](quickstart.md)。

```{raw} html
<div class="sw-home-code">
<p class="sw-home-code-label">模型 → 波传播 → 记录 → 损失 → 梯度</p>
```

```python
import torch
import starwave

predicted = starwave.scalar(
    velocity, grid_spacing=10.0, dt=0.001,
    source_amplitudes=amplitudes,
    source_locations=sources,
    receiver_locations=receivers,
    pml_freq=15.0,
)[0]
loss = torch.nn.functional.mse_loss(predicted, observed)
loss.backward()
```

```{raw} html
</div>
<p class="sw-home-code-note">片段假设原生库已准备就绪，模型、固定采集和观测张量符合接口约定并位于同一 CUDA 设备；velocity 为开启梯度的 float32 模型。INR 页面提供用户网络的概念与接线示例，尚未提供完整收敛验证。</p>
```

(home-start)=
## 从一个小实验开始

```{raw} html
<div class="sw-home-paths">
  <a href="installation.html#installation-smoke"><span>01</span><strong>安装与运行检查</strong><span aria-hidden="true">→</span></a>
  <a href="modeling/gradient.html"><span>02</span><strong>计算模型梯度</strong><span aria-hidden="true">→</span></a>
  <a href="inversion/fwi.html"><span>03</span><strong>完成一次简单 FWI 实验</strong><span aria-hidden="true">→</span></a>
</div>
<p class="sw-home-footnote">本手册对应公开 2.0.0 接口；教程展示的 A30 实测结果来自 0.1.0.dev9。运行条件与验证范围见<a href="status.html">文档状态</a>。Windows 用户请从 <a href="wsl.html">WSL 2</a> 开始。</p>
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: 开始使用

installation
wsl
quickstart
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: 正演建模

modeling/conventions
modeling/scalar
modeling/gradient
modeling/vrz
modeling/vti
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: 反演与神经网络

inversion/fwi
inversion/dataparallel
inversion/inr
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Usage

usage
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: 参考与维护

faq
release-notes
status
```
