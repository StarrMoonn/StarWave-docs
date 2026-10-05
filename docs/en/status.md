# Documentation Status and References

## Completed and pending work

| Content | Current status |
|---|---|
| Chinese and English navigation, installation, and WSL guidance | Fully translated; installation on target devices has not yet been tested |
| scalar / VRZ / VTI API reference | Dedicated function pages; signatures, defaults, types, shapes/units, and constraints checked for 54 parameters |
| Synthetic scalar forward-modeling and single-update FWI script | Written and syntax-checked; GPU execution awaits validation |
| VRZ / VTI | Usage conventions and call snippets provided; complete standalone examples still pending |
| DataParallel | Shot-wise splitting snippet; multi-GPU comparisons still pending |
| INR | Concepts and integration snippet; standalone network program and convergence experiments still pending |
| Illumination API | Exported names listed only; complete lifecycle tutorial still pending |
| Numerical and performance evidence | No unmeasured error tables, speedups, or convergence claims are provided |

## References

- [StarWave 2.0.0 on PyPI](https://pypi.org/project/starwave/2.0.0/): public release requirements and limitations.
- [Deepwave API documentation](https://ausargeo.com/deepwave/usage): consulted for API organization and Sphinx/Alabaster presentation; all StarWave descriptions are original.
- [PyFWI documentation](https://pyfwi.readthedocs.io/en/latest/): consulted only for manual navigation structure.
- [PyTorch previous-version installation instructions](https://pytorch.org/get-started/previous-versions/) and [DataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html).
- [Microsoft WSL installation](https://learn.microsoft.com/en-us/windows/wsl/install) and the [NVIDIA WSL user guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
- [Sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html), [GitHub Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages), and the [RTD configuration reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html).

API checks are based on the released wheel's Python interface signatures and authorized usage documentation. The documentation has been reorganized as introductory material and does not include the software implementation. Public wheel SHA-256:

```text
6622b863c76db1ba708622048377f3de4447609ff7295d1705b8145884ce06db
```

Last checked: 2026-10-05. When reference material changes, review the installation requirements and API again against the actual release version.
