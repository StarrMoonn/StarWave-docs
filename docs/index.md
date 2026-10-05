# StarWave 使用手册

StarWave 为 PyTorch 提供 CUDA 波传播接口，覆盖二维标量声学、二维 VRZ 与二维/三维声学 VTI。本手册面向使用公开 **2.0.0 wheel** 的用户，从环境安装、小规模正演开始，再介绍梯度与反演接线。

第一次使用：阅读[安装](installation.md) → [快速入门](quickstart.md) → [模型与采集约定](modeling/conventions.md)。Windows 用户从 [WSL 2](wsl.md) 开始；查看实际实验：{ref}`安装运行检查 <installation-smoke>` → [简单梯度计算](modeling/gradient.md) → [简单 FWI 示例](inversion/fwi.md)。

```{raw} html
<div class="sw-api-overview" aria-label="传播接口比较">
<table role="table">
<colgroup><col class="sw-col-interface"><col class="sw-col-model"><col class="sw-col-record"><col class="sw-col-detail"></colgroup>
<thead role="rowgroup"><tr role="row"><th scope="col" role="columnheader">接口</th><th scope="col" role="columnheader">模型与维数</th><th scope="col" role="columnheader">默认记录</th><th scope="col" role="columnheader">主要区别</th></tr></thead>
<tbody role="rowgroup">
<tr role="row"><th scope="row" role="rowheader"><a href="usage.html#starwave.scalar"><code>starwave.scalar</code></a></th><td role="cell" data-label="模型与维数"><span class="sw-field-value"><code>v</code>，2D</span></td><td role="cell" data-label="默认记录"><span class="sw-field-value">pressure-like</span></td><td role="cell" data-label="主要区别"><span class="sw-field-value">等间距；标量声学</span></td></tr>
<tr role="row"><th scope="row" role="rowheader"><a href="usage.html#starwave.vrz"><code>starwave.vrz</code></a></th><td role="cell" data-label="模型与维数"><span class="sw-field-value"><code>v</code> 与 <code>impedance</code> 或 <code>density</code>，2D</span></td><td role="cell" data-label="默认记录"><span class="sw-field-value">pressure-like</span></td><td role="cell" data-label="主要区别"><span class="sw-field-value">必须恰好选择一种介质参数</span></td></tr>
<tr role="row"><th scope="row" role="rowheader"><a href="usage.html#starwave.vti"><code>starwave.vti</code></a></th><td role="cell" data-label="模型与维数"><span class="sw-field-value"><code>vp</code>、<code>epsilon</code>、<code>delta</code>、<code>rho</code>，2D/3D</span></td><td role="cell" data-label="默认记录"><span class="sw-field-value"><code>vz</code></span></td><td role="cell" data-label="主要区别"><span class="sw-field-value">声学 VTI；支持各轴不同网格间距</span></td></tr>
</tbody>
</table>
</div>
```

```{toctree}
:maxdepth: 1
:caption: 开始使用

installation
wsl
quickstart
```

```{toctree}
:maxdepth: 1
:caption: 正演建模

modeling/conventions
modeling/scalar
modeling/gradient
modeling/vrz
modeling/vti
```

```{toctree}
:maxdepth: 1
:caption: 反演与神经网络

inversion/fwi
inversion/dataparallel
inversion/inr
```

```{toctree}
:maxdepth: 1
:caption: Usage

usage
```

```{toctree}
:maxdepth: 1
:caption: 参考与维护

faq
release-notes
status
```

导航结构参考 [PyFWI 使用手册](https://pyfwi.readthedocs.io/en/latest/) 的入门、正演、反演分类；本手册内容针对 StarWave 编写，不表示两者接口或数值实现相同。
