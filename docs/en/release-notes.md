# Release Notes

## V15 / GSLS source Release · 2026-10-10

The [V15 source Release](https://github.com/StarrMoonn/StarWave/releases/tag/V15) is published with source version `0.1.0.dev15`, commit `2508543ccc1dc3015b07806e6e240dd88c8e871e`. PyPI is not updated: public 7.0.0 remains the historical V14 binary and cannot run the current GSLS API.

- [GSLS](visco-gsls.md) becomes the only viscoacoustic interface. Independent `visco_sls` and its preparation/status helpers are removed. Single-mechanism calculations use `mode="sls_compat", n_mechanisms=1`.
- 2D fixed variable density, native CPU/CUDA full/checkpoint, explicit Torch full, and first-order Vp/Q/source/provider gradients. Default Hao1, explicit band_fit and sls_compat have distinct Q meanings and are not interchangeable.
- Existing scalar/VRZ/VTI/elastic scientific implementations, signatures and defaults remain unchanged. GSLS ABI 2 / semantics 1 / capabilities 255 requires matching rebuilds and a restart. `compile_all.py` covers core, elastic and GSLS; GSLS CPU is built explicitly and separately.
- Navigation and current API documentation introduce GSLS directly. The independent SLS API, experiment pages and downloads are removed from the current manual. Original materials remain in Git history and are not relabelled as GSLS.

This documentation update does not claim final integrated GPU, multi-GPU, long-FWI or performance acceptance. See [Installation](installation.md) and [Status](status.md) for channels, historical verification scope and sources.

## 7.0.0 / V14 (historical) · 2026-10-09

[StarWave 7.0.0](https://pypi.org/project/starwave/7.0.0/) corresponds to authorized source `0.1.0.dev14` and first publicly supplies the newly integrated SLS viscoacoustic interface. The Linux binary wheel contains four native libraries: core, elastic, SLS CPU and SLS CUDA. Local compilation is unnecessary and no sdist is provided.

- Adds `starwave.visco_sls`, `starwave.prepare_visco_sls` and `starwave.visco_sls_native_status`; signatures and defaults of the nine existing public functions are unchanged.
- 2D single-SLS, fixed variable density and full volume histories, with independent first-order Vp/Q/source gradients. Native CPU, CUDA and Torch-reference backends are selected explicitly, without automatic fallback.
- Vp is phase velocity at `f_ref`; Q is the complex-modulus quality factor there. Sources use Pa/m² with half-weight startup; recordings are pre-update `p[n]`, without resampling.
- Trainable Vp/Q with nonzero PML require a fixed explicit unrelaxed-speed envelope `max_vel`. Density gradients, 3D and boundary mode are unsupported.

Existing scalar, VRZ, VTI and elastic scientific calls and Scalar3D Examples are preserved. Restart Python/the notebook kernel after upgrading; SLS uses separate native preparation. See [Installation](installation.md) and [Status](status.md) for installation and final artifact identity.

## 6.0.0 / V13 · 2026-10-09

The public [StarWave 6.0.0 wheel](https://pypi.org/project/starwave/6.0.0/) corresponds to authorized source `0.1.0.dev13` and focuses on internal storage and CUDA execution maintenance. See [Status](status.md) for artifact identity and verification scope:

- Scalar2D uses directional compact PML states, pressure boundary strips of width `M=accuracy//2`, and full history excluding PML/finite-difference halos but retaining `boundary_buffer` for ordinary objective backpropagation (`illumination=None`). CUDA updates use 32 × 8 thread blocks and direct loads.
- Scalar2D transpose updates correct extra support in the plain region and neighboring union-PML contributions. This does not supply the complete transpose of model replicate-padding.
- Elastic2D/3D full/boundary remove L2-cache-based shot-group trajectory scheduling while retaining the existing kernel mapping. 3D forward still loops over shots inside the kernel; this is not full shot parallelism in the CUDA grid.

Public API signatures, defaults, source scaling, CFL, and existing calls are unchanged. Scalar3D, VRZ, and VTI scientific implementations are outside this optimization scope. Source users must rebuild matching native libraries; all users should restart Python / the notebook kernel after upgrading. See [Installation and upgrading](installation.md).

This documentation update did not independently rerun a GPU and makes no general speedup, peak-memory, or faster-than-SWEEP claim. Existing tutorials and Scalar3D Examples retain their original versions and evidence scopes; they are not new 6.0.0 acceptance results.

## 5.0.0 / V12 · 2026-10-08

The public [StarWave 5.0.0 wheel](https://pypi.org/project/starwave/5.0.0/) extends `starwave.scalar` without changing its signature: `v.ndim` selects 2D/3D, and 3D supports unequal spacing, orders 2/4/6/8, Radius-M six-face pressure reconstruction, and one first-order velocity/source backward. `full` retains `Lap(u)` volume history at each internal step. Coordinates directly index the model in its own axis order. The result remains the one-element `(receiver_amplitudes,)` tuple with `[B,R,T]` recordings.

2D scalar behavior and fixed sources are preserved, as are VRZ, VTI, Deepwave elastic, and their examples. 3D requires CUDA FP32; illumination remains 2D scalar only. PML/model-extension gradient limits still apply, and boundary mode does not guarantee production-scale 3D memory fit. The public package version is 5.0.0; the private-source version is not a PyPI installation target.

This maintenance synchronizes bilingual Usage, installation, model conventions, Scalar3D reconstruction, and runnable call fragments, with API checks against the 5.0.0 wheel. It changes no propagation implementation and ran no GPU numerical, long-FWI, multi-GPU, or performance tests. See [Status](status.md) for publication and historical tutorial evidence.

## 4.0.0 · 2026-10-07

The public [StarWave 4.0.0 wheel](https://pypi.org/project/starwave/4.0.0/) corresponds to V11 and adds `starwave.elastic(lamb, mu, buoyancy, ...)` with a Deepwave 0.0.27-derived backend, 2D/3D, orders 2/4/6/8 and complete state/receiver returns. Convert materials explicitly through `starwave.common`; elastic uses the separate `prepare_elastic` helper. Scalar remains 2D, and the scalar/VRZ/VTI public signatures and defaults are preserved.

## Elastic API documentation review · 2026-10-08

Added Elastic Function within the existing Usage page, covering 62 parameters, 16/31 return order, axes and a minimal example, plus both material conversions and prepare_elastic. The review targets the published 4.0.0 wheel and distinguishes full/boundary modes, sampling intervals, survey_pad and existing validation scopes. Installation pins and bilingual API checks are updated; existing scalar/VRZ/VTI prose, historical tutorial values, attribution, typography and navigation are preserved.

This is documentation-only maintenance, with no propagation runtime or release artifact changes and no GPU numerical/performance execution. Sources, wheel hash and acceptance limits are recorded in [Status](status.md).

## 2.0.0 · 2026-10-05

Public release of prebuilt Linux x86_64 wheels. This version improves the fused paths for VTI 2D/3D boundary backward. The public Python API and defaults remain unchanged. The move to version 2.0 does not add new scalar/VRZ capabilities. Changes in floating-point operation order mean that bitwise equality with the previous version should not be required.

Restart the interpreter or notebook kernel after upgrading. This version number does not indicate new validation of numerical behavior on target GPUs, multi-GPU execution, FWI, or performance. The known limitations involving model-extension gradients, VRZ stability under strong contrasts, and VTI `epsilon < delta` still apply.

Use [PyPI 2.0.0](https://pypi.org/project/starwave/2.0.0/) as the public release source for installation requirements and the full release scope.

## First draft of this user manual

Adds installation and WSL guidance, a synthetic scalar example, usage guidance for all three equation types, introductions to FWI/DataParallel/INR, an API index, and an FAQ. The first draft used Sphinx with the Read the Docs theme and could be deployed to GitHub Pages or another static server. The current bilingual edition retains that original template.

The first draft of this manual includes no CUDA implementation, Python propagation implementation, private experimental data, or internal records. See [Documentation status](status.md) for tutorials and validation work still to be completed.

## Expanded API reference

Scalar, VRZ, and VTI each have a dedicated reference page using the Sphinx Python domain to present complete signatures, individual parameter types/defaults/shapes/units, return structures, autograd scope, and examples. Descriptions of 54 parameters have been added, and the function-entry anchors from the original API index have been preserved. This is a documentation improvement; it does not change the StarWave 2.0.0 runtime interface or validation status.

## Chinese and English edition

The complete manual is available in Chinese and English, including navigation, search, notices, and all 18 content pages. Each language has an independent build and search index. Language switching preserves the corresponding page and section, and legacy Chinese links and API anchors remain accessible. The site retains the original PyFWI / Read the Docs template. API content follows Deepwave’s Sphinx organization, adds typed signatures, and retains StarWave’s own return contracts and numerical limitations.
