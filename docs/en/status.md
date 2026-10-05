# Documentation Status and References

## Completed and pending work

| Content | Current status |
|---|---|
| Chinese and English navigation, installation, and WSL guidance | Fully translated; a separate A30 / 0.1.0.dev9 smoke run is documented, while public-wheel device tests remain pending |
| scalar / VRZ / VTI API reference | Single Usage page; signatures, defaults, types, shapes/units, and constraints checked for 54 parameters |
| Three original teaching notebooks | Returned executed notebooks contain no errors; artifact hashes and arrays checked |
| Scalar forward, gradients, and 25 FWI updates | Measured A30 / 0.1.0.dev9 run; data fit improves substantially, model recovery only 0.86% |
| Original 32 × 32 wiring script | Retained and syntax-checked; this separate configuration has no attached measured results |
| VRZ / VTI | Usage conventions and call snippets provided; complete standalone examples still pending |
| DataParallel | Shot-wise splitting snippet; multi-GPU comparisons still pending |
| INR | Concepts and integration snippet; standalone network program and convergence experiments still pending |
| Illumination API | Exported names listed only; complete lifecycle tutorial still pending |
| Adjoint, performance, and device scope | Directional/full-adjoint checks pending; no GPU timing, multi-GPU, or public 2.0.0 wheel acceptance claim |

## References

- [StarWave 2.0.0 on PyPI](https://pypi.org/project/starwave/2.0.0/): public release requirements and limitations.
- [Deepwave API documentation](https://ausargeo.com/deepwave/usage): consulted for API content organization; the site retains the Read the Docs theme; all StarWave descriptions are original.
- [PyFWI documentation](https://pyfwi.readthedocs.io/en/latest/): consulted only for manual navigation structure.
- [PyTorch previous-version installation instructions](https://pytorch.org/get-started/previous-versions/) and [DataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html).
- [Microsoft WSL installation](https://learn.microsoft.com/en-us/windows/wsl/install) and the [NVIDIA WSL user guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
- [Sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html), [GitHub Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages), and the [RTD configuration reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html).

API checks are based on the released wheel's Python interface signatures and authorized usage documentation. The documentation has been reorganized as introductory material and does not include the software implementation. Public wheel SHA-256:

```text
6622b863c76db1ba708622048377f3de4447609ff7295d1705b8145884ce06db
```

Last checked: 2026-10-05. When reference material changes, review the installation requirements and API again against the actual release version.

(tutorial-evidence)=
## Tutorial provenance and reproduction scope

All returned server notebook code sources match the three original deliveries, with no execution-error outputs. SHA-256 and byte counts match for 39 run artifacts. Numerical-array shapes, finiteness, fixed bands, and final-iterate consistency were checked. The CUDA solver was not rerun during documentation editing.

- {ref}`Installation smoke test <installation-smoke>`: 48 × 40, records `[1,12,160]`, finite nonzero forward records and first-order velocity gradient.
- [Simple Gradient Computation](modeling/gradient.md): 96 × 64, three shots, zero fixed-band gradient; directional check disabled.
- [Simple FWI Example](inversion/fwi.md): 25 updates, data term 91.51% lower, active velocity RMSE only 0.86% lower.

The recorded environment is NVIDIA A30, Python 3.10.18, PyTorch 2.5.1 / CUDA 11.8, and StarWave **0.1.0.dev9**. This does not establish execution of the public **2.0.0 wheel**; a reference development commit in the original notebook does not identify the installed commit.

Public notebooks are unexecuted privacy-edited editions: outputs, execution counts, and machine-related metadata were cleared; private-source installation notes and two reference-commit provenance fields were removed. Other numerical, plotting, and export cells remain unchanged. Both language pages provide the same original Chinese teaching notebooks. Paired `.py` files match public notebook code. The complete returned notebook outputs have not been published.

Download the {download}`exact settings and result summary <../examples/tutorials/results_summary.json>` and {download}`original/public hashes and edit scope <../examples/tutorials/source_manifest.json>`. Figures were redrawn from verified arrays; shared scales, source-array hashes, and image hashes appear in the [figure provenance record](_static/tutorials/figure_manifest.json). Raw system details, environment logs, and private paths are excluded.
