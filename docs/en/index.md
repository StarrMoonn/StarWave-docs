# StarWave User Manual

StarWave provides CUDA wave-propagation interfaces for PyTorch, covering 2D scalar acoustics, 2D VRZ, and 2D/3D acoustic VTI. This manual is for users of the public **2.0.0 wheel**. It begins with environment setup and small forward-modeling runs, then introduces gradients and inversion workflows.

```{important}
This is a bilingual draft manual; use the language switch in the sidebar, or open the navigation menu on mobile. The example interfaces have been checked, but numerical accuracy, performance, and inversion convergence have not been validated on real GPUs. Compiling for a GPU architecture does not establish that the software has passed tests on that device. Read the numerical limitations for each equation first.
```

New users: read [Installation](installation.md) → [Quickstart](quickstart.md) → [Model and acquisition conventions](modeling/conventions.md). Windows users should start with [WSL 2](wsl.md); users preparing for inversion should continue to [FWI](inversion/fwi.md).

| Interface | Models and dimensions | Default recording | Main distinctions |
|---|---|---|---|
| `starwave.scalar` | `v`, 2D | pressure-like | Equal grid spacing; scalar acoustics |
| `starwave.vrz` | `v` and either `impedance` or `density`, 2D | pressure-like | Exactly one of the two medium parameters is required |
| `starwave.vti` | `vp, epsilon, delta, rho`, 2D/3D | `vz` | Acoustic VTI; supports different grid spacings along each axis |

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
wsl
quickstart
```

```{toctree}
:maxdepth: 2
:caption: Forward modeling

modeling/conventions
modeling/scalar
modeling/vrz
modeling/vti
```

```{toctree}
:maxdepth: 2
:caption: Inversion and neural networks

inversion/fwi
inversion/dataparallel
inversion/inr
```

```{toctree}
:maxdepth: 3
:caption: Usage / API

api/index
```

```{toctree}
:maxdepth: 2
:caption: Reference and maintenance

faq
release-notes
status
```

The navigation draws on the getting-started, forward-modeling, and inversion categories in the [PyFWI user manual](https://pyfwi.readthedocs.io/en/latest/). The content here was written for StarWave; this does not imply that the two packages share the same interfaces or numerical implementations.
