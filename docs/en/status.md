# Documentation Status and References

## V15 / GSLS · 2026-10-10

The current manual targets the published [V15 source Release](https://github.com/StarrMoonn/StarWave/releases/tag/V15), source version `0.1.0.dev15`, at commit `2508543ccc1dc3015b07806e6e240dd88c8e871e`. A source Release and a PyPI binary distribution are separate: public **7.0.0** is unchanged and contains neither GSLS nor the standalone-SLS source cleanup.

[GSLS](visco-gsls.md) is the current viscoacoustic entry point: 2D, fixed spatially varying density, native CPU/CUDA full/checkpoint, explicit Torch full, and first-order Vp/Q/source/provider gradients. Independent SLS API/production sources/build entry points are removed; a single mechanism explicitly uses `mode="sls_compat", n_mechanisms=1`. This manual does not publish propagation implementations, CUDA source or complete source archives.

Parameters and physical descriptions follow that V15 source contract. Default Hao1 Q is nominal Q0; Vp is reference phase velocity. Trainable material with nonzero PML requires a fixed `max_vel` covering unrelaxed speed. 3D, density gradients, boundary mode and higher-order propagation derivatives are unsupported. See GSLS for the complete limits and [Installation](installation.md) for explicit build/restart steps.

Supplied A30/RTX 4060 records identify pre-integration source, not the final host-guard/integration fingerprint. They do not establish a final integrated GPU rerun. This manual update ran no GPU, DataParallel, long FWI or target-device performance test. The old independent SLS API, experiment pages and downloads are removed from the current manual. Original materials remain in Git history and are not relabelled as GSLS results.

The sections below preserve historical release-verification facts; they are not new V15 acceptance results.

## V14 / 7.0.0 SLS (historical)

[StarWave 7.0.0](https://pypi.org/project/starwave/7.0.0/) was published on 2026-10-09, corresponding to authorized V14 / `0.1.0.dev14`. Its sole distribution is `starwave-7.0.0-py3-none-manylinux_2_35_x86_64.whl` (17,515,547 bytes), with no sdist. The official PyPI file is byte-identical to the reviewed CI build. Python 3.10–3.12 installation/host checks and a clean reinstall from the public artifact passed.

The historical bilingual SLS API retained the nine existing function contracts and added three entry points. All twelve signatures, defaults, keyword boundaries and return contracts match the actual wheel. Four ELF libraries (core, elastic, SLS CPU and SLS CUDA) and their receipt hashes were verified. The documentation checker reads the archive without loading or executing native libraries.

Public 7.0.0 wheel SHA-256:

```text
0fb2d66555ad19d4178738db93ea5d71c3b7dbfb456cf4fd82898f5d44725964
```

Scope is 2D single-SLS, fixed variable density, full histories and first-order Vp/Q/source gradients. The source-version GPU results recorded at the time included: fixed-Q and joint Vp/Q runs of 100 epochs each on two A30 GPUs, plus a separate RTX 4060 ring model. These original runs are distinct from reinstalling the public wheel. This documentation update did not independently rerun a GPU. Existing tutorials, figures, Scalar3D split downloads and PDFs retain their original versions and evidence scopes.

## V13 / 6.0.0 maintenance scope · 2026-10-09

This maintenance targets 6.0.0 / source `0.1.0.dev13`: installation versions, old-library compatibility warnings, Scalar2D internal storage and PML-transpose corrections, and removal of elastic L2 shot-group trajectory scheduling. The nine public function signatures retain the existing 5.0.0 contract; no API parameters or example calls are added or changed.

[StarWave 6.0.0](https://pypi.org/project/starwave/6.0.0/) was published on 2026-10-09. The official PyPI version record contains only `starwave-6.0.0-py3-none-manylinux_2_35_x86_64.whl` (16,674,439 bytes), with no sdist; an independent download from the official file URL matches the built artifact in byte count and SHA-256. Version/platform metadata, both native-library receipt hashes, and nine public signatures, defaults, and recorded type/return contracts were checked. The documentation checker only reads the archive and never imports or executes StarWave.

The release workflow passed Python 3.10–3.12 installation and host/CPU checks, followed by a clean-environment installation from public PyPI. These results are not CUDA numerical acceptance; scalar3D still has no CPU propagation path.

Public 6.0.0 wheel SHA-256:

```text
297a4e359d86003513452294e6384f78a6ab7029fdabefa936e0733228882cf7
```

This update independently ran no GPU numerical, performance, multi-GPU, or long-FWI tests. Storage-layout descriptions are not peak-memory measurements or general speed guarantees. Existing tutorials and Scalar3D Example figures, downloads, and measurements are unchanged and retain their original versions and scopes.

## V12 / 5.0.0 Scalar3D API review · 2026-10-08

The historical installation target in this section is public [StarWave 5.0.0](https://pypi.org/project/starwave/5.0.0/). The unchanged `starwave.scalar` signature selects 2D/3D from `v.ndim`. 3D supports unequal per-axis spacing, orders 2/4/6/8, Radius-M six-face boundary or full memory, and one first-order velocity/source backward. 2D sources remain fixed; 3D illumination is unsupported. Existing VRZ, VTI, and Deepwave elastic contracts are preserved.

This update independently read the public wheel and checked its SHA-256, nine public signatures, parameter defaults/types, and return contracts. Scalar, the 3D wrapper/storage layout, validation, and temporal-sampling modules were also byte-matched to the authorized source. Bilingual documentation and call fragments are checked with Python 3.10 syntax; the documentation build does not load or execute StarWave.

Public 5.0.0 wheel SHA-256:

```text
bf159a6544cdc1ab12c99e22578ec40ed56e0d3011ae71be9fc8ecb04f9c2bff
```

Publication records confirm Python 3.10–3.12 installation checks and independent public-wheel download/install verification. CPU/host checks do not imply scalar3D CPU propagation: that path is CUDA-only. User-provided source-version GPU reports are separately identified historical evidence. This documentation update did not rerun a GPU and makes no public-wheel A30, multi-GPU, long-FWI, peak-memory, or performance acceptance claim. The 4.0.0, 2.0.0, and 0.1.0.dev9 records below retain their original versions rather than being relabeled as new 5.0.0 experiments.

## V11 / 4.0.0 Elastic API review · 2026-10-08

The published [StarWave 4.0.0](https://pypi.org/project/starwave/4.0.0/) wheel is the authority for the added interface. All 62 elastic parameters, both conversion helpers and prepare_elastic were checked for types, order, defaults and calling boundaries; the existing five API signatures are unchanged in this wheel. Bilingual builds and local link/anchor checks include the new APIs, and Python examples are checked as Python 3.10 syntax. Documentation builds neither import nor execute StarWave.

Public wheel SHA-256:

```text
9f64f2677c54af5b2bd1c48e509a24c7d40eb0257d9e86267e721d3f61eaa954
```

Publication checks passed Python 3.10–3.12 installation and 2D/3D CPU/host checks of the public wheel. User-supplied RTX 4060 source-test reports are separate evidence. This update did not rerun a GPU and does not claim public-wheel target-GPU numerical, A30, multi-GPU, long-FWI or performance acceptance. The original tutorials and 2.0.0 documentation baseline remain below; their historical results are not relabeled as new 4.0.0 experiments.

## Completed and pending work

| Content | Current status |
|---|---|
| Chinese and English navigation, installation, and WSL guidance | Fully translated; a separate A30 / 0.1.0.dev9 smoke run is documented, while public-wheel device tests remain pending |
| scalar / VRZ / VTI API reference | Single Usage page; signatures, defaults, types, shapes/units, and constraints checked for 54 parameters |
| Scalar3D / 5.0.0 | API, Radius-M storage contract, and a standalone small example added; no GPU run in this update |
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

The following is the historical 2.0.0 documentation baseline: API checks were based on that released wheel's Python interface signatures and authorized usage documentation. The documentation has been reorganized as introductory material and does not include the software implementation. Public wheel SHA-256:

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
