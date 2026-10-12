# Documentation Status and References

## V19 / 0.1.0.dev19 · 2026-10-12

[V19 Release](https://github.com/StarrMoonn/StarWave/releases/tag/V19) is published. This manual follows frozen source `0.1.0.dev19`, commit `46d32877e3336d531520a696d5f2e074e6f4ef74`. [PyPI **9.1.0**](https://pypi.org/project/starwave/9.1.0/) is published with the V19 source contract and integrated VRR. The existing [9.0.0 wheel](https://pypi.org/project/starwave/9.0.0/) remains V18 and excludes integrated VRR.

Published wheel: `starwave-9.1.0-py3-none-manylinux_2_35_x86_64.whl`, 29,014,423 bytes, uploaded 2026-10-12 06:05:56 UTC. SHA-256: `07e9bfe00425e8a28b67805c1e300a05f9f60a4d9b963fa7d4f7b1c0425d0df7`. Public PyPI metadata matches the audited artifact. Packaging and CPU checks do not establish GPU, multi-GPU, MPS, long-FWI or performance acceptance.

V19 integrates {ref}`VRR <vrr>` as `starwave.vrr`. Root installation and `compile_all.py` include VRR; no separate `starwave_vrr` installation is needed. VRR supports explicit Torch CPU/CUDA float32/64 full/checkpoint with eager/compile, and native CUDA float32 full/boundary. Device, gradient and boundary restrictions remain equation-specific.

V19 preserves the V18 23-parameter scalar contract: arbitrary zero-face subsets (positive widths must remain equal), trainable 2D/3D sources, differentiable terminal physical fields and diagnostic forward callbacks. Terminal returns can carry a loss; callback snapshots are detached observations and cannot carry loss gradients. See [Scalar Function](usage.md#scalar-function).

V19 freezes the reviewed integrated baseline without changing scientific kernels or public signatures. Root installation/build includes five native groups: core, elastic, GSLS, Born and VRR. Formal examples use equation directories and paired CUDA/TorchCompile scripts/notebooks; example startup and INR memory forwarding fixes are included. Source publication does not establish 9.1.0 wheel identity, GPU/multi-GPU acceptance, MPS acceptance, long-FWI convergence or performance. Historical figures, measured tutorials and hashes below retain their original evidence scope.


## Published binary and historical evidence

The current published binary is V19 / 9.1.0 with integrated VRR. Its Linux/Python/Torch requirements and installation commands are in [Installation](installation.md#published-v19-binary-channel). Version-specific publication receipts, hashes, old platform requirements and verification records are archived in [Release Notes](release-notes.md). The V19 source release does not replace or relabel those historical measurements.

## Completed and pending work

| Area | Current evidence and remaining limits |
|---|---|
| V19 source / release | Frozen 0.1.0.dev19 contract and published V19 Release; source identity is distinct from public binary identity |
| PyPI 9.1.0 | Published; artifact identity and CPU checks passed; target-device numerical acceptance remains separate |
| Bilingual API manual | Seven propagation entries; scalar 23 parameters; integrated VRR, Born, and equation-specific backend/gradient/boundary contracts |
| Installation / import | Root registration and five-group explicit builds; import binds available libraries without compiling or runtime identity audits |
| Source examples | Equation-organized CUDA/TorchCompile notebook/script pairs; matching complete source is required |
| Original teaching notebooks | Historical A30 / 0.1.0.dev9 results and checked asset hashes; not rerun as V19 experiments |
| Scalar3D examples | Separately scoped historical source-build numerical and four-GPU experiments; incomplete surface-only recovery remains disclosed |
| Illumination | Public helpers and lifecycle documented in Usage; device and gradient restrictions remain scoped |
| DataParallel / INR | Wiring and selected-memory guidance; no new V19 multi-GPU or convergence acceptance is inferred |
| GPU / MPS / long FWI / performance | Documentation, CPU, compilation, release and numerical checks remain separate; this manual update adds no target-hardware acceptance |

## References

- [StarWave 9.1.0 on PyPI](https://pypi.org/project/starwave/9.1.0/): public release requirements and limitations.
- [Deepwave API documentation](https://ausargeo.com/deepwave/usage): consulted for API content organization; the site retains the Read the Docs theme; all StarWave descriptions are original.
- [PyFWI documentation](https://pyfwi.readthedocs.io/en/latest/): consulted only for manual navigation structure.
- [PyTorch previous-version installation instructions](https://pytorch.org/get-started/previous-versions/) and [DataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html).
- [Microsoft WSL installation](https://learn.microsoft.com/en-us/windows/wsl/install) and the [NVIDIA WSL user guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
- [Sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html), [GitHub Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages), and the [RTD configuration reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html).

(tutorial-evidence)=
## Tutorial provenance and reproduction scope

All returned server notebook code sources match the three original deliveries, with no execution-error outputs. SHA-256 and byte counts match for 39 run artifacts. Numerical-array shapes, finiteness, fixed bands, and final-iterate consistency were checked. The CUDA solver was not rerun during documentation editing.

- {ref}`Installation smoke test <installation-smoke>`: 48 × 40, records `[1,12,160]`, finite nonzero forward records and first-order velocity gradient.
- [Simple Gradient Computation](modeling/gradient.md): 96 × 64, three shots, zero fixed-band gradient; directional check disabled.
- [Simple FWI Example](inversion/fwi.md): 25 updates, data term 91.51% lower, active velocity RMSE only 0.86% lower.

The recorded environment is NVIDIA A30, Python 3.10.18, PyTorch 2.5.1 / CUDA 11.8, and StarWave **0.1.0.dev9**. This does not establish execution of the public **2.0.0 wheel**; a reference development commit in the original notebook does not identify the installed commit.

Public notebooks are unexecuted privacy-edited editions: outputs, execution counts, and machine-related metadata were cleared; private-source installation notes and two reference-commit provenance fields were removed. Other numerical, plotting, and export cells remain unchanged. Both language pages provide the same original Chinese teaching notebooks. Paired `.py` files match public notebook code. The complete returned notebook outputs have not been published.

Download the {download}`exact settings and result summary <../examples/tutorials/results_summary.json>` and {download}`original/public hashes and edit scope <../examples/tutorials/source_manifest.json>`. Figures were redrawn from verified arrays; shared scales, source-array hashes, and image hashes appear in the [figure provenance record](_static/tutorials/figure_manifest.json). Raw system details, environment logs, and private paths are excluded.
