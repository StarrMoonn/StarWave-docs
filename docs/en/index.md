# StarWave User Manual

StarWave provides CUDA wave-propagation interfaces for PyTorch, covering 2D scalar acoustics, 2D VRZ, and 2D/3D acoustic VTI. This manual is for users of the public **2.0.0 wheel**. It begins with environment setup and small forward-modeling runs, then introduces gradients and inversion workflows.

New users: read [Installation](installation.md) → [Quickstart](quickstart.md) → [Model and acquisition conventions](modeling/conventions.md). Windows users should start with [WSL 2](wsl.md); explore the measured experiments: {ref}`installation smoke test <installation-smoke>` → [Simple Gradient Computation](modeling/gradient.md) → [Simple FWI Example](inversion/fwi.md).

```{raw} html
<div class="sw-api-overview" aria-label="Propagation interface comparison">
<table role="table">
<colgroup><col class="sw-col-interface"><col class="sw-col-model"><col class="sw-col-record"><col class="sw-col-detail"></colgroup>
<thead role="rowgroup"><tr role="row"><th scope="col" role="columnheader">Interface</th><th scope="col" role="columnheader">Models and dimensions</th><th scope="col" role="columnheader">Default recording</th><th scope="col" role="columnheader">Main distinctions</th></tr></thead>
<tbody role="rowgroup">
<tr role="row"><th scope="row" role="rowheader"><a href="usage.html#starwave.scalar"><code>starwave.scalar</code></a></th><td role="cell" data-label="Models and dimensions"><span class="sw-field-value"><code>v</code>, 2D</span></td><td role="cell" data-label="Default recording"><span class="sw-field-value">pressure-like</span></td><td role="cell" data-label="Main distinctions"><span class="sw-field-value">Equal grid spacing; scalar acoustics</span></td></tr>
<tr role="row"><th scope="row" role="rowheader"><a href="usage.html#starwave.vrz"><code>starwave.vrz</code></a></th><td role="cell" data-label="Models and dimensions"><span class="sw-field-value"><code>v</code> and either <code>impedance</code> or <code>density</code>, 2D</span></td><td role="cell" data-label="Default recording"><span class="sw-field-value">pressure-like</span></td><td role="cell" data-label="Main distinctions"><span class="sw-field-value">Exactly one of the two medium parameters is required</span></td></tr>
<tr role="row"><th scope="row" role="rowheader"><a href="usage.html#starwave.vti"><code>starwave.vti</code></a></th><td role="cell" data-label="Models and dimensions"><span class="sw-field-value"><code>vp</code>, <code>epsilon</code>, <code>delta</code>, <code>rho</code>, 2D/3D</span></td><td role="cell" data-label="Default recording"><span class="sw-field-value"><code>vz</code></span></td><td role="cell" data-label="Main distinctions"><span class="sw-field-value">Acoustic VTI; supports different grid spacings along each axis</span></td></tr>
</tbody>
</table>
</div>
```

```{toctree}
:maxdepth: 1
:caption: Getting started

installation
wsl
quickstart
```

```{toctree}
:maxdepth: 1
:caption: Forward modeling

modeling/conventions
modeling/scalar
modeling/gradient
modeling/vrz
modeling/vti
```

```{toctree}
:maxdepth: 1
:caption: Inversion and neural networks

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
:caption: Reference and maintenance

faq
release-notes
status
```

The navigation draws on the getting-started, forward-modeling, and inversion categories in the [PyFWI user manual](https://pyfwi.readthedocs.io/en/latest/). The content here was written for StarWave; this does not imply that the two packages share the same interfaces or numerical implementations.
