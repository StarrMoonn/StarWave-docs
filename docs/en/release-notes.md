# Release Notes

## V19 / 0.1.0.dev19 · 2026-10-12

[V19 Release](https://github.com/StarrMoonn/StarWave/releases/tag/V19) is published. This manual follows frozen source `0.1.0.dev19`, commit `46d32877e3336d531520a696d5f2e074e6f4ef74`. [PyPI **9.1.0**](https://pypi.org/project/starwave/9.1.0/) is published with the V19 source contract and integrated VRR. The existing [9.0.0 wheel](https://pypi.org/project/starwave/9.0.0/) remains V18 and excludes integrated VRR.

Published wheel: `starwave-9.1.0-py3-none-manylinux_2_35_x86_64.whl`, 29,014,423 bytes, uploaded 2026-10-12 06:05:56 UTC. SHA-256: `07e9bfe00425e8a28b67805c1e300a05f9f60a4d9b963fa7d4f7b1c0425d0df7`. Public PyPI metadata matches the audited artifact. Packaging and CPU checks do not establish GPU, multi-GPU, MPS, long-FWI or performance acceptance.

V19 integrates {ref}`VRR <vrr>` as `starwave.vrr`. Root installation and `compile_all.py` include VRR; no separate `starwave_vrr` installation is needed. VRR supports explicit Torch CPU/CUDA float32/64 full/checkpoint with eager/compile, and native CUDA float32 full/boundary. Device, gradient and boundary restrictions remain equation-specific.

V19 preserves the V18 23-parameter scalar contract: arbitrary zero-face subsets (positive widths must remain equal), trainable 2D/3D sources, differentiable terminal physical fields and diagnostic forward callbacks. Terminal returns can carry a loss; callback snapshots are detached observations and cannot carry loss gradients. See [Scalar Function](usage.md#scalar-function).

V19 freezes the reviewed integrated baseline without changing scientific kernels or public signatures. Root installation/build includes five native groups: core, elastic, GSLS, Born and VRR. Formal examples use equation directories and paired CUDA/TorchCompile scripts/notebooks; example startup and INR memory forwarding fixes are included. Source publication does not establish 9.1.0 wheel identity, GPU/multi-GPU acceptance, MPS acceptance, long-FWI convergence or performance. Historical figures, measured tutorials and hashes below retain their original evidence scope.


## V18 / PyPI 9.0.0 · 2026-10-12

V18 scalar (reviewed at dev `e040aa9b8225ab387f7e4853bfe6162f5351f5d5`) adds arbitrary zero-face subsets (positive widths still equal), trainable 2D/3D sources, optional differentiable terminal physical fields and diagnostic forward callbacks. These source features are absent from frozen V17/its 8.0.0 release baseline and from 7.1.0. See [Scalar Function](usage.md#scalar-function) for the 23-parameter contract; the older 20-parameter release snapshot remains separately pinned.

[Release V18](https://github.com/StarrMoonn/StarWave/releases/tag/V18) is published at `fba1f496b890f2dbaf3fbb3011d39f6d723cfea1`, preserving the reviewed scalar API from dev `e040aa9`. [PyPI 9.0.0](https://pypi.org/project/starwave/9.0.0/) is now published with these features for supported Linux x86_64 hosts; see [installation](installation.md) for requirements and commands. V17 and the unpublished 8.0.0 baseline remain unchanged; 7.1.0 is the historical V16 wheel.

The published file is `starwave-9.0.0-py3-none-manylinux_2_35_x86_64.whl` (27,975,674 bytes), uploaded on 2026-10-12; SHA-256 `d290ea0a7fdc2a74229d989525aa2bff1cdc9d675fb715bd6f43a6e17b1ddd0b`. There is no sdist or macOS wheel. Requirements: Linux x86_64 / glibc ≥ 2.35, Python ≥ 3.10 and < 3.13, PyTorch ≥ 2.5 and < 2.6, NumPy ≥ 1.23. The public download matches this hash. The [publication workflow](https://github.com/StarrMoonn/StarWave/actions/runs/38199707015) passed all eight jobs, including clean public-PyPI download/install/smoke checks on Python 3.10, 3.11 and 3.12. These are packaging and host/CPU checks, not public-wheel GPU, multi-GPU, long-FWI or performance acceptance. Mac MPS remains source-only within its documented Scalar2D limits.

## V17 source, Born and flat-top boundaries

[Release V17](https://github.com/StarrMoonn/StarWave/releases/tag/V17) is published from frozen source `0.1.0.dev17`, commit `27c2e14b184b94d8031a5b3a83c25b1ed3d4ac67`. PyPI **8.0.0 has not been published**. The historical **7.1.0 wheel remains the V16 binary** recorded below; it does not contain Born or the V17 flat-top/import changes.

- The main package adds `starwave.scalar_born` and `starwave.ScalarBorn`: 1D/2D/3D constant-density acoustic Born, velocity perturbation δv, background/scattered records and final states. Native CPU full, native CUDA full/boundary, and explicit Torch CPU/CUDA full/checkpoint have distinct contracts. See [Born Function](usage.md#born-function).
- Flat-top behavior is equation-specific: scalar/VRZ use a last-axis top-zero fixed halo; VTI enforces zero normal traction; elastic uses a first-axis improved-vacuum surface; GSLS uses its 2D pressure-release closure. Born retains its own nonnegative per-face zero-ghost convention. These are not interchangeable free-surface implementations; see [model conventions](modeling/conventions.md).
- After explicit compilation, import prepares available native libraries and shared dispatch. It does not compile, initialize CUDA, select devices, allocate wavefields or propagate. Manual preparation is optional for explicit checks. Missing builds permit supported explicit Torch paths; a retained invalid build fails rather than being silently ignored.
- The frozen V17 archive's `compile_all.py` builds core/elastic/GSLS; add `python compile_born.py --arch 80` for Born. Reviewed later dev `3f2b24d` integrates Born into `compile_all.py`, building/verifying all four libraries. `python compile_born.py --cpu-only` remains available. The independent `starwave_vrr` extension stays separately installed/built. See [installation](installation.md).

The manual publishes public contracts and synthetic examples only. It contains no private scientific implementation or source archive. Release publication, source/build identity, documentation checks, CPU checks and target-GPU/multi-GPU/FWI acceptance remain separate evidence. Existing measured tutorials, figures, PDFs and hashes retain their original versions and scope. The records below are historical snapshots, including the historical V16 wheel.

## V16 / 7.1.0 · 2026-10-11

<span id="v16-pytorch-backend-publication-verification-pending"></span>

[StarWave 7.1.0](https://pypi.org/project/starwave/7.1.0/) was published on 2026-10-11 from frozen V16 source `0.1.0.dev16`, commit `5a224dbb1e86154ff034bbedaf67c0e4cf7a1b6a`. It includes GSLS and the additive [PyTorch backend](pytorch-backend.md). The sole distribution is `starwave-7.1.0-py3-none-manylinux_2_35_x86_64.whl` (17,628,561 bytes), with no sdist. It targets Linux x86_64 / glibc ≥ 2.35, Python ≥ 3.10 and < 3.13, and PyTorch ≥ 2.5 and < 2.6. Historical [7.0.0](https://pypi.org/project/starwave/7.0.0/) remains the V14 binary without GSLS or the unified Torch backend.

The independently downloaded public wheel matches the verified artifact SHA-256:

```text
36a0be76af997e2d31de8313cab00ff59d79f7537055c829df2bfdee0ff61e88
```

Its receipt pins source-manifest SHA-256 `f682ff0a836c0c259cd03a2e760460d68426cc5688e4852f8c62a151804694dc`. The [publication workflow](https://github.com/StarrMoonn/StarWave/actions/runs/38108561775) passed artifact verification, Python 3.10/3.11/3.12 pre-publication host/CPU checks, publishing, and installation from public PyPI on all three Python versions. These checks do not establish new-binary GPU numerical, multi-GPU, long-FWI or performance acceptance.

There is no macOS wheel. Mac MPS use requires authorized complete V16 source and remains limited to the Scalar2D configuration below. The later dev free-surface work and independently packaged {ref}`VRR extension <vrr>` are not included in V16 / PyPI 7.1.0. The earlier VRR documentation review covered the source integration at `904c9106bdb883325df8586b7c3412f7814a9737`; this is a documentation update, not a new package release. User-supplied device reports remain distinct from documentation/CPU checks and public-wheel validation.

- scalar/VRZ/VTI/elastic add backend, execution, checkpoint_interval and compile_steps; GSLS adds execution/compile_steps and Torch checkpoint. Existing positional arguments, native defaults and scientific CUDA/C++ kernels are preserved.
- Torch offers CPU/CUDA full/checkpoint and eager/compile. It has no boundary, AMP, elastic callbacks/offload or sampled material gradients. MPS is supported only for Scalar2D accuracy=4 float32 with backend="torch", full/checkpoint and eager/compile. Other MPS combinations, including Scalar3D, VRZ, VTI, elastic and GSLS, are rejected. No final Mac hardware, 500-epoch FWI or performance acceptance is claimed. See [Usage](usage.md), {ref}`GSLS <visco-gsls>` and the backend page for complete parameters and limits.
- Torch elastic differentiates every internal step; native material gradients sample at CFL ratio × sampling_interval. Exact cross-backend comparisons require ratio=1; this distinction is not a compilation error.
- Documentation gates check bilingual complete signatures, defaults, parameter coverage and snippet syntax without importing propagation libraries. CPU/documentation checks do not replace final GPU, DataParallel, long FWI or performance acceptance. Existing examples, figures and PDFs retain their original version/evidence scope.

V15 and older records below describe their historical snapshots only.


## V15 / GSLS source Release · 2026-10-10

The [V15 source Release](https://github.com/StarrMoonn/StarWave/releases/tag/V15) is published with source version `0.1.0.dev15`, commit `2508543ccc1dc3015b07806e6e240dd88c8e871e`. PyPI is not updated: public 7.0.0 remains the historical V14 binary and cannot run the current GSLS API.

- {ref}`GSLS <visco-gsls>` becomes the only viscoacoustic interface. Independent `visco_sls` and its preparation/status helpers are removed. Single-mechanism calculations use `mode="sls_compat", n_mechanisms=1`.
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


## Historical V16 installation and migration

These archived commands reproduce the V16 / 7.1.0 channel. They are not the V19 installation instructions; use [Installation](installation.md) for current source or published binary choices.

(historical-7-0-0-wheel-requirements)=
### Historical 7.1.0 wheel requirements

| Component | StarWave 7.1.0 wheel requirement |
|---|---|
| Operating system | Linux x86_64, glibc ≥ 2.35 |
| C++ runtime | libstdc++ providing GLIBCXX_3.4.30 / CXXABI_1.3.13 symbols or newer |
| Python | ≥ 3.10 and < 3.13 (3.10–3.12) |
| PyTorch / NumPy | PyTorch ≥ 2.5 and < 2.6; NumPy ≥ 1.23 |
| Backend / device | Native scalar 2D/3D, VRZ/VTI and elastic boundary require NVIDIA CUDA; elastic full and GSLS native_cpu support CPU. Explicit Torch full/checkpoint supports CPU/CUDA across all five propagators |
| Compilation targets | SASS 70/75/80/86/89/90 and compute80 PTX |

Installing the prebuilt wheel does not require nvcc or a local CUDA Toolkit. A conservative target for the Linux driver is 520.61.05 or newer, and the driver must also support the actual GPU. WSL uses the Windows host driver; see the [WSL guidance](wsl.md). The A30 and RTX 4060 Laptop are covered by the compilation targets; device-level numerical and performance tests of the public 7.1.0 wheel are still pending. A separate development-version A30 run is documented below.

The version information above comes from the [PyPI 7.1.0 release description](https://pypi.org/project/starwave/7.1.0/); see [Documentation Status](status.md) for artifact identity and verification scope. The public release contains only a Linux wheel, with no sdist. This wheel cannot be used directly on native Windows, macOS, or ARM hosts.

(install-the-historical-7-0-0-wheel)=
### Install the historical 7.1.0 wheel

Run the following commands in a Linux or WSL 2 Linux terminal that meets the requirements. Choose Python 3.10–3.12; this example uses 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --only-binary=starwave starwave==7.1.0
python -m pip check
```

The PyTorch installation command follows the [PyTorch previous-versions page](https://pytorch.org/get-started/previous-versions/). `cu118` is the starting configuration for validating this version; it does not mean that other PyTorch versions have been validated. Restart Python or the notebook kernel after upgrading StarWave.

(upgrade-to-v16-source)=
### Historical migration to V16 / 7.1.0

V16 has one viscoacoustic entry point, `starwave.visco_gsls`; independent `visco_sls` and its preparation/status helpers are removed. Single-mechanism calculations explicitly use `mode="sls_compat", n_mechanisms=1`; changing only the function name would select default Hao1 instead. Source version `0.1.0.dev16` identifies the frozen source underlying the public `7.1.0` wheel; the distribution version numbers differ.

Source users must pair this version's Python package with rebuilt native libraries, following the instructions supplied with the source. When upgrading from a version earlier than 6.0.0, Scalar2D PML, boundary, and full-history layouts have changed; matching core ABI layouts do not make old libraries reusable, and compatibility guards must not be bypassed. The public wheel needs no local compilation. After upgrading or replacing loaded libraries, restart Python / the notebook kernel, then repeat native preparation and small forward/gradient checks.

Internal operation ordering and the PML-transpose correction can change low-order bits or previously incorrect gradients; bitwise equality with older versions is not expected. Memory optimizations do not remove model-extension gradient limits or guarantee speed or memory benefits for every configuration.



## Historical artifact verification records

The following records preserve the original version-specific documentation and artifact checks. None is a V19 runtime acceptance claim.

### V18 / PyPI 9.0.0 · 2026-10-12

V18 scalar (reviewed at dev `e040aa9b8225ab387f7e4853bfe6162f5351f5d5`) adds arbitrary zero-face subsets (positive widths still equal), trainable 2D/3D sources, optional differentiable terminal physical fields and diagnostic forward callbacks. These source features are absent from frozen V17/its 8.0.0 release baseline and from 7.1.0. See [Scalar Function](usage.md#scalar-function) for the 23-parameter contract; the older 20-parameter release snapshot remains separately pinned.

[Release V18](https://github.com/StarrMoonn/StarWave/releases/tag/V18) is published at `fba1f496b890f2dbaf3fbb3011d39f6d723cfea1`, preserving the reviewed scalar API from dev `e040aa9`. [PyPI 9.0.0](https://pypi.org/project/starwave/9.0.0/) is now published with these features for supported Linux x86_64 hosts; see [installation](installation.md) for requirements and commands. V17 and the unpublished 8.0.0 baseline remain unchanged; 7.1.0 is the historical V16 wheel.

The published file is `starwave-9.0.0-py3-none-manylinux_2_35_x86_64.whl` (27,975,674 bytes), uploaded on 2026-10-12; SHA-256 `d290ea0a7fdc2a74229d989525aa2bff1cdc9d675fb715bd6f43a6e17b1ddd0b`. There is no sdist or macOS wheel. Requirements: Linux x86_64 / glibc ≥ 2.35, Python ≥ 3.10 and < 3.13, PyTorch ≥ 2.5 and < 2.6, NumPy ≥ 1.23. The public download matches this hash. The [publication workflow](https://github.com/StarrMoonn/StarWave/actions/runs/38199707015) passed all eight jobs, including clean public-PyPI download/install/smoke checks on Python 3.10, 3.11 and 3.12. These are packaging and host/CPU checks, not public-wheel GPU, multi-GPU, long-FWI or performance acceptance. Mac MPS remains source-only within its documented Scalar2D limits.

The documentation CPU checkpoint example passed on Torch 2.5.1+cpu: terminal shape (1,12,10), empty records (1,0,7), callbacks at user steps 3/6/7 and internal steps 6/12/14, finite model/source gradients, no callbacks during backward replay. This is a small CPU wiring check. The user supplied an RTX 4060/A30 feature-branch report; this documentation update did not rerun GPU propagation and does not transfer its results to a merged build or released wheel.

### V17 source, Born and flat-top boundaries

[Release V17](https://github.com/StarrMoonn/StarWave/releases/tag/V17) is published from frozen source `0.1.0.dev17`, commit `27c2e14b184b94d8031a5b3a83c25b1ed3d4ac67`. PyPI **8.0.0 has not been published**. The historical **7.1.0 wheel remains the V16 binary** recorded below; it does not contain Born or the V17 flat-top/import changes.

- The main package adds `starwave.scalar_born` and `starwave.ScalarBorn`: 1D/2D/3D constant-density acoustic Born, velocity perturbation δv, background/scattered records and final states. Native CPU full, native CUDA full/boundary, and explicit Torch CPU/CUDA full/checkpoint have distinct contracts. See [Born Function](usage.md#born-function).
- Flat-top behavior is equation-specific: scalar/VRZ use a last-axis top-zero fixed halo; VTI enforces zero normal traction; elastic uses a first-axis improved-vacuum surface; GSLS uses its 2D pressure-release closure. Born retains its own nonnegative per-face zero-ghost convention. These are not interchangeable free-surface implementations; see [model conventions](modeling/conventions.md).
- After explicit compilation, import prepares available native libraries and shared dispatch. It does not compile, initialize CUDA, select devices, allocate wavefields or propagate. Manual preparation is optional for explicit checks. Missing builds permit supported explicit Torch paths; a retained invalid build fails rather than being silently ignored.
- The frozen V17 archive's `compile_all.py` builds core/elastic/GSLS; add `python compile_born.py --arch 80` for Born. Reviewed later dev `3f2b24d` integrates Born into `compile_all.py`, building/verifying all four libraries. `python compile_born.py --cpu-only` remains available. The independent `starwave_vrr` extension stays separately installed/built. See [installation](installation.md).

The manual publishes public contracts and synthetic examples only. It contains no private scientific implementation or source archive. Release publication, source/build identity, documentation checks, CPU checks and target-GPU/multi-GPU/FWI acceptance remain separate evidence. Existing measured tutorials, figures, PDFs and hashes retain their original versions and scope. The records below are historical snapshots, including the historical V16 wheel.

### V16 / 7.1.0 · 2026-10-11



[StarWave 7.1.0](https://pypi.org/project/starwave/7.1.0/) was published on 2026-10-11 from frozen V16 source `0.1.0.dev16`, commit `5a224dbb1e86154ff034bbedaf67c0e4cf7a1b6a`. It includes GSLS and the additive [PyTorch backend](pytorch-backend.md). The sole distribution is `starwave-7.1.0-py3-none-manylinux_2_35_x86_64.whl` (17,628,561 bytes), with no sdist. It targets Linux x86_64 / glibc ≥ 2.35, Python ≥ 3.10 and < 3.13, and PyTorch ≥ 2.5 and < 2.6. Historical [7.0.0](https://pypi.org/project/starwave/7.0.0/) remains the V14 binary without GSLS or the unified Torch backend.

The independently downloaded public wheel matches the verified artifact SHA-256:

```text
36a0be76af997e2d31de8313cab00ff59d79f7537055c829df2bfdee0ff61e88
```

Its receipt pins source-manifest SHA-256 `f682ff0a836c0c259cd03a2e760460d68426cc5688e4852f8c62a151804694dc`. The [publication workflow](https://github.com/StarrMoonn/StarWave/actions/runs/38108561775) passed artifact verification, Python 3.10/3.11/3.12 pre-publication host/CPU checks, publishing, and installation from public PyPI on all three Python versions. These checks do not establish new-binary GPU numerical, multi-GPU, long-FWI or performance acceptance.

There is no macOS wheel. Mac MPS use requires authorized complete V16 source and remains limited to the Scalar2D configuration below. The later dev free-surface work and independently packaged {ref}`VRR extension <vrr>` are not included in V16 / PyPI 7.1.0. The earlier VRR documentation review covered the source integration at `904c9106bdb883325df8586b7c3412f7814a9737`; this is a documentation update, not a new package release. User-supplied device reports remain distinct from documentation/CPU checks and public-wheel validation.

- scalar/VRZ/VTI/elastic add backend, execution, checkpoint_interval and compile_steps; GSLS adds execution/compile_steps and Torch checkpoint. Existing positional arguments, native defaults and scientific CUDA/C++ kernels are preserved.
- Torch offers CPU/CUDA full/checkpoint and eager/compile. It has no boundary, AMP, elastic callbacks/offload or sampled material gradients. MPS is supported only for Scalar2D accuracy=4 float32 with backend="torch", full/checkpoint and eager/compile. Other MPS combinations, including Scalar3D, VRZ, VTI, elastic and GSLS, are rejected. No final Mac hardware, 500-epoch FWI or performance acceptance is claimed. See [Usage](usage.md), {ref}`GSLS <visco-gsls>` and the backend page for complete parameters and limits.
- Torch elastic differentiates every internal step; native material gradients sample at CFL ratio × sampling_interval. Exact cross-backend comparisons require ratio=1; this distinction is not a compilation error.
- Documentation gates check bilingual complete signatures, defaults, parameter coverage and snippet syntax without importing propagation libraries. CPU/documentation checks do not replace final GPU, DataParallel, long FWI or performance acceptance. Existing examples, figures and PDFs retain their original version/evidence scope.

V15 and older records below describe their historical snapshots only.


### V15 / GSLS · 2026-10-10

The V15 manual targeted the published [V15 source Release](https://github.com/StarrMoonn/StarWave/releases/tag/V15), source version `0.1.0.dev15`, at commit `2508543ccc1dc3015b07806e6e240dd88c8e871e`. A source Release and a PyPI binary distribution are separate: public **7.0.0** is unchanged and contains neither GSLS nor the standalone-SLS source cleanup.

{ref}`GSLS <visco-gsls>` is the current viscoacoustic entry point: 2D, fixed spatially varying density, native CPU/CUDA full/checkpoint, explicit Torch full, and first-order Vp/Q/source/provider gradients. Independent SLS API/production sources/build entry points are removed; a single mechanism explicitly uses `mode="sls_compat", n_mechanisms=1`. This manual does not publish propagation implementations, CUDA source or complete source archives.

Parameters and physical descriptions follow that V15 source contract. Default Hao1 Q is nominal Q0; Vp is reference phase velocity. Trainable material with nonzero PML requires a fixed `max_vel` covering unrelaxed speed. 3D, density gradients, boundary mode and higher-order propagation derivatives are unsupported. See GSLS for the complete limits and [Installation](installation.md) for explicit build/restart steps.

Supplied A30/RTX 4060 records identify pre-integration source, not the final host-guard/integration fingerprint. They do not establish a final integrated GPU rerun. This manual update ran no GPU, DataParallel, long FWI or target-device performance test. The old independent SLS API, experiment pages and downloads are removed from the current manual. Original materials remain in Git history and are not relabelled as GSLS results.

The sections below preserve historical release-verification facts; they are not new V15 acceptance results.

### V14 / 7.0.0 SLS (historical)

[StarWave 7.0.0](https://pypi.org/project/starwave/7.0.0/) was published on 2026-10-09, corresponding to authorized V14 / `0.1.0.dev14`. Its sole distribution is `starwave-7.0.0-py3-none-manylinux_2_35_x86_64.whl` (17,515,547 bytes), with no sdist. The official PyPI file is byte-identical to the reviewed CI build. Python 3.10–3.12 installation/host checks and a clean reinstall from the public artifact passed.

The historical bilingual SLS API retained the nine existing function contracts and added three entry points. All twelve signatures, defaults, keyword boundaries and return contracts match the actual wheel. Four ELF libraries (core, elastic, SLS CPU and SLS CUDA) and their receipt hashes were verified. The documentation checker reads the archive without loading or executing native libraries.

Public 7.0.0 wheel SHA-256:

```text
0fb2d66555ad19d4178738db93ea5d71c3b7dbfb456cf4fd82898f5d44725964
```

Scope is 2D single-SLS, fixed variable density, full histories and first-order Vp/Q/source gradients. The source-version GPU results recorded at the time included: fixed-Q and joint Vp/Q runs of 100 epochs each on two A30 GPUs, plus a separate RTX 4060 ring model. These original runs are distinct from reinstalling the public wheel. This documentation update did not independently rerun a GPU. Existing tutorials, figures, Scalar3D split downloads and PDFs retain their original versions and evidence scopes.

### V13 / 6.0.0 maintenance scope · 2026-10-09

This maintenance targets 6.0.0 / source `0.1.0.dev13`: installation versions, old-library compatibility warnings, Scalar2D internal storage and PML-transpose corrections, and removal of elastic L2 shot-group trajectory scheduling. The nine public function signatures retain the existing 5.0.0 contract; no API parameters or example calls are added or changed.

[StarWave 6.0.0](https://pypi.org/project/starwave/6.0.0/) was published on 2026-10-09. The official PyPI version record contains only `starwave-6.0.0-py3-none-manylinux_2_35_x86_64.whl` (16,674,439 bytes), with no sdist; an independent download from the official file URL matches the built artifact in byte count and SHA-256. Version/platform metadata, both native-library receipt hashes, and nine public signatures, defaults, and recorded type/return contracts were checked. The documentation checker only reads the archive and never imports or executes StarWave.

The release workflow passed Python 3.10–3.12 installation and host/CPU checks, followed by a clean-environment installation from public PyPI. These results are not CUDA numerical acceptance; scalar3D still has no CPU propagation path.

Public 6.0.0 wheel SHA-256:

```text
297a4e359d86003513452294e6384f78a6ab7029fdabefa936e0733228882cf7
```

This update independently ran no GPU numerical, performance, multi-GPU, or long-FWI tests. Storage-layout descriptions are not peak-memory measurements or general speed guarantees. Existing tutorials and Scalar3D Example figures, downloads, and measurements are unchanged and retain their original versions and scopes.

### V12 / 5.0.0 Scalar3D API review · 2026-10-08

The historical installation target in this section is public [StarWave 5.0.0](https://pypi.org/project/starwave/5.0.0/). The unchanged `starwave.scalar` signature selects 2D/3D from `v.ndim`. 3D supports unequal per-axis spacing, orders 2/4/6/8, Radius-M six-face boundary or full memory, and one first-order velocity/source backward. 2D sources remain fixed; 3D illumination is unsupported. Existing VRZ, VTI, and Deepwave elastic contracts are preserved.

This update independently read the public wheel and checked its SHA-256, nine public signatures, parameter defaults/types, and return contracts. Scalar, the 3D wrapper/storage layout, validation, and temporal-sampling modules were also byte-matched to the authorized source. Bilingual documentation and call fragments are checked with Python 3.10 syntax; the documentation build does not load or execute StarWave.

Public 5.0.0 wheel SHA-256:

```text
bf159a6544cdc1ab12c99e22578ec40ed56e0d3011ae71be9fc8ecb04f9c2bff
```

Publication records confirm Python 3.10–3.12 installation checks and independent public-wheel download/install verification. CPU/host checks do not imply scalar3D CPU propagation: that path is CUDA-only. User-provided source-version GPU reports are separately identified historical evidence. This documentation update did not rerun a GPU and makes no public-wheel A30, multi-GPU, long-FWI, peak-memory, or performance acceptance claim. The 4.0.0, 2.0.0, and 0.1.0.dev9 records below retain their original versions rather than being relabeled as new 5.0.0 experiments.

### V11 / 4.0.0 Elastic API review · 2026-10-08

The published [StarWave 4.0.0](https://pypi.org/project/starwave/4.0.0/) wheel is the authority for the added interface. All 62 elastic parameters, both conversion helpers and prepare_elastic were checked for types, order, defaults and calling boundaries; the existing five API signatures are unchanged in this wheel. Bilingual builds and local link/anchor checks include the new APIs, and Python examples are checked as Python 3.10 syntax. Documentation builds neither import nor execute StarWave.

Public wheel SHA-256:

```text
9f64f2677c54af5b2bd1c48e509a24c7d40eb0257d9e86267e721d3f61eaa954
```

Publication checks passed Python 3.10–3.12 installation and 2D/3D CPU/host checks of the public wheel. User-supplied RTX 4060 source-test reports are separate evidence. This update did not rerun a GPU and does not claim public-wheel target-GPU numerical, A30, multi-GPU, long-FWI or performance acceptance. The original tutorials and 2.0.0 documentation baseline remain below; their historical results are not relabeled as new 4.0.0 experiments.



### Historical 2.0.0 documentation baseline

The following is the historical 2.0.0 documentation baseline: API checks were based on that released wheel's Python interface signatures and authorized usage documentation. The documentation has been reorganized as introductory material and does not include the software implementation. Public wheel SHA-256:

```text
6622b863c76db1ba708622048377f3de4447609ff7295d1705b8145884ce06db
```

Last checked: 2026-10-05. When reference material changes, review the installation requirements and API again against the actual release version.
