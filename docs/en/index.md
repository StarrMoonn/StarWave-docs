# StarWave User Manual

StarWave provides CUDA wave-propagation interfaces for PyTorch, covering 2D scalar acoustics, 2D VRZ, and 2D/3D acoustic VTI. This manual is for users of the public **2.0.0 wheel**. It begins with environment setup and small forward-modeling runs, then introduces gradients and inversion workflows.

```{important}
The API reference targets the public 2.0.0 wheel. Three original tutorials now include measured NVIDIA A30 runs using StarWave 0.1.0.dev9: a small smoke check, gradients, and 25 FWI updates. These do not validate the public wheel, complete adjoint correctness, convergence, or performance. Read each page’s version and evidence scope.
```

New users: read [Installation](installation.md) → [Quickstart](quickstart.md) → [Model and acquisition conventions](modeling/conventions.md). Windows users should start with [WSL 2](wsl.md); explore the measured experiments: {ref}`installation smoke test <installation-smoke>` → [Simple Gradient Computation](modeling/gradient.md) → [Simple FWI Example](inversion/fwi.md).

| Interface | Models and dimensions | Default recording | Main distinctions |
|---|---|---|---|
| `starwave.scalar` | `v`, 2D | pressure-like | Equal grid spacing; scalar acoustics |
| `starwave.vrz` | `v` and either `impedance` or `density`, 2D | pressure-like | Exactly one of the two medium parameters is required |
| `starwave.vti` | `vp, epsilon, delta, rho`, 2D/3D | `vz` | Acoustic VTI; supports different grid spacings along each axis |

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
