# Installation

This manual targets [V17 source](https://github.com/StarrMoonn/StarWave/releases/tag/V17), `0.1.0.dev17` at `27c2e14b184b94d8031a5b3a83c25b1ed3d4ac67`, including Born and the merged flat-top/import behavior. PyPI **8.0.0 is not published**. The available [7.1.0 wheel](https://pypi.org/project/starwave/7.1.0/) is the V16 baseline with GSLS and the [PyTorch backend](pytorch-backend.md), without the V17 additions. See [Status](status.md) for the exact artifacts and evidence.

The separately installed {ref}`VRR source extension <vrr>` uses `starwave_vrr.vrr`, separate from the main-package entry points. It is absent from 7.1.0. Its explicit Torch mode supports CPU/CUDA float32/64 full/checkpoint and eager/compile; native CUDA float32 full/boundary requires its own build. Follow that page for extension installation and limits.

Use the prebuilt 7.1.0 wheel on supported Linux x86_64 hosts. The source repository is private; source users need complete authorized V17 source. There is no macOS wheel or PyPI sdist: Mac MPS requires the source installation below and only the documented Scalar2D configuration is eligible.


Unreleased dev `e040aa9b8225ab387f7e4853bfe6162f5351f5d5` scalar adds arbitrary zero-face subsets (positive widths still equal), trainable 2D/3D sources, optional differentiable terminal physical fields and diagnostic forward callbacks. These are source-only development features, absent from frozen V17/its 8.0.0 release baseline and from 7.1.0. See [Scalar Function](usage.md#scalar-function) for the 23-parameter contract; the older 20-parameter release snapshot remains separately pinned.

The planned release targets are V18 source and PyPI 9.0.0; publication and wheel validation are still pending. Until verified artifacts are available, use the reviewed source. These features are not retroactively part of V17 or the 8.0.0 baseline.

This scalar development integration retains import-time native binding. The core loader now loads/binds once without runtime source/library hash, fingerprint or version probing; loading/binding errors still fail. A successful import is not a source/build identity audit or numerical check. Rebuild matching source and restart Python/Jupyter after changes; prepare_native remains an optional explicit device check.

## Source and wheel Torch installation and execution

Linux wheel users can select the Torch backend directly after installing 7.1.0 below. For source use, including Mac MPS, run from the complete V17 source root, or optionally register it:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

Explicit `backend="torch"` supports CPU/CUDA full/checkpoint without propagation native libraries, nvcc or compile_all.py. Execution mode and model device are separate selections. Compile uses the current PyTorch Inductor toolchain and raises on failure without eager fallback. MPS is supported only for Scalar2D accuracy=4 float32 with backend="torch", full/checkpoint and eager/compile. Other MPS combinations, including Scalar3D, VRZ, VTI, elastic and GSLS, are rejected. No final Mac hardware, 500-epoch FWI or performance acceptance is claimed. AMP remains rejected. Start with the {ref}`standalone CPU example <torch-example>`.

## V17 and later dev native builds

Native source use requires complete matching source/library builds; the 7.1.0 wheel already contains its own matching V16 libraries. V17 changes native boundary/build contracts. Rebuild and restart after source updates; never edit an old manifest or bypass compatibility checks. Supported pure-Torch source use can omit native compilation, but import still loads retained native builds when present, with backend-specific compatibility checks.


Use an installed CUDA Toolkit/C++ toolchain matching PyTorch. From the complete source root:

```bash
python compile_all.py --arch 80
```

For reviewed later dev `3f2b24d`, this builds/verifies four libraries: core (scalar/VRZ/VTI), elastic, GSLS and Born. In the immutable V17 archive, the same command builds only core/elastic/GSLS; build Born separately:

```bash
# Required after compile_all.py for frozen V17; also usable separately on later dev.
python compile_born.py --arch 80
# Alternative for native CPU full only:
python compile_born.py --cpu-only
```

Use `80` for A30, `89` for RTX 4060, or your actual GPU architecture. `combined_build.py` covers core/elastic only. Born is a main-package API with its own native library; it is absent from 7.1.0. Pure Torch Born CPU/CUDA full/checkpoint needs no native Born build. VRR remains separately installed/built. To build GSLS alone, select CUDA or CPU explicitly:

```bash
# GSLS CUDA
python compile_visco_gsls.py --arch 80
# GSLS native CPU
python compile_visco_gsls.py --backend cpu
```

GSLS source builds output to `native/build/gsls/`, or the `gsls/` subdirectory of the common `STARWAVE_BUILD_DIR` override. Build and runtime must use matching source/library inputs; do not mix old standalone libraries. Native compatibility is checked against each backend's matching build; the historical V16 wheel retains its previously recorded build identity. Restart Python/the notebook kernel after updates. Compilation or loading does not establish GPU numerical acceptance.

A complete source checkout runs without pip registration. Optional registration is separate from compilation and does not implicitly build:

```bash
python -m pip install --no-deps -e .
```

The following requirements and commands install the verified public 7.1.0 wheel.

(historical-7-0-0-wheel-requirements)=
(runtime-requirements)=
## 7.1.0 wheel requirements

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
## Install the public 7.1.0 wheel

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
## Upgrade to V16 / 7.1.0

V16 has one viscoacoustic entry point, `starwave.visco_gsls`; independent `visco_sls` and its preparation/status helpers are removed. Single-mechanism calculations explicitly use `mode="sls_compat", n_mechanisms=1`; changing only the function name would select default Hao1 instead. Source version `0.1.0.dev16` identifies the frozen source underlying the public `7.1.0` wheel; the distribution version numbers differ.

Source users must pair this version's Python package with rebuilt native libraries, following the instructions supplied with the source. When upgrading from a version earlier than 6.0.0, Scalar2D PML, boundary, and full-history layouts have changed; matching core ABI layouts do not make old libraries reusable, and compatibility guards must not be bypassed. The public wheel needs no local compilation. After upgrading or replacing loaded libraries, restart Python / the notebook kernel, then repeat native preparation and small forward/gradient checks.

Internal operation ordering and the PML-transpose correction can change low-order bits or previously incorrect gradients; bitwise equality with older versions is not expected. Memory optimizations do not remove model-extension gradient limits or guarantee speed or memory benefits for every configuration.

## Check the core library (installed wheel or registered source)

```python
import torch
import starwave
from importlib.metadata import version

print(version("starwave"))
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available())
status = starwave.native_status()
print({key: status[key] for key in ("library_exists", "library_loaded")})
# Optional explicit device check in V17; required preparation for the V16 native example:
starwave.prepare_native([0])
```

V17 `import starwave` prepares available core/elastic/Born/GSLS libraries and shared dispatch before returning; an installed/importable VRR extension is prepared separately. It does not compile, initialize CUDA, select devices, allocate wavefields or propagate. Set build paths before import, and rebuild/restart after updates. Missing unbuilt libraries permit supported explicit Torch use; retained invalid builds fail. Manual prepare calls remain optional device checks, not mandatory serial warm-up. The older V16 / 7.1.0 wheel still leaves the core library unloaded until preparation. Neither import nor preparation replaces numerical forward/gradient checks; continue with [Quickstart](quickstart.md).

The documentation site's build environment is separate from the computational runtime environment. Reading or building the documentation does not require a GPU.


Elastic retains an optional explicit V17 preparation entry point (required by older native workflows): call `starwave.prepare_elastic([0])` on the GPU main thread, or `starwave.prepare_elastic()` with `memory="full"` for CPU. See {ref}`Elastic Function <elastic>` for parameters and vp/vs/rho conversion. Native Scalar3D has no CPU propagation path; V16 Torch Scalar3D supports CPU/CUDA. GPU, DataParallel and long-FWI acceptance scopes are listed in [Status](status.md).

## GSLS preparation and backend selection

GSLS retains an optional explicit V17 preparation entry point (required by older native workflows): call `starwave.prepare_visco_gsls(backend="cuda", device="cuda:0")` on the CUDA main thread, or `starwave.prepare_visco_gsls(backend="native_cpu")` for CPU. `starwave.visco_gsls_native_status(backend="cuda")` only reports status. Preparation defaults to CPU while propagation defaults to CUDA; select explicitly, without automatic fallback. `backend="torch"` supports full/checkpoint and eager/compile without a native build. See the {ref}`GSLS API <visco-gsls>` for the complete contract and CPU example.

(installation-smoke)=
## Notebook 01: Executed smoke test

Download {download}`01_installation_environment_smoke.ipynb <../examples/tutorials/01_installation_environment_smoke.ipynb>` or its {download}`paired Python script <../examples/tutorials/01_installation_environment_smoke.py>`. The notebook also needs NumPy, Matplotlib ≥ 3.6, and Jupyter. Install them in a separate terminal for the selected environment, rather than upgrading a running kernel:

```bash
python -m pip install numpy "matplotlib>=3.6" jupyterlab ipykernel
python -m jupyter lab
```

Select the corresponding CUDA kernel and run every cell from the beginning. Each download is a standalone, output-free public edition retaining the original Chinese teaching notes. Numerical experiments are unchanged; see the {ref}`provenance record <tutorial-evidence>` for removed environment provenance fields and private installation notes.

```{admonition} Measured environment · 2026-10-05
:class: sw-run-note
NVIDIA A30 · Python 3.10.18 · PyTorch 2.5.1 / CUDA 11.8 · installed StarWave **0.1.0.dev9**. The server results below check a small workflow in this development-version environment. They do not establish device validation of the public **2.0.0, 4.0.0, 5.0.0, 6.0.0, 7.0.0, or 7.1.0 wheel**.
```

## Small experiment design

Use a uniform 1800 m/s model with 48 × 40 cells and 10 m spacing. One source is at (240, 50) m; 12 receivers span x=80…410 m every 30 m at z=50 m. The 15 Hz Ricker peaks at 0.055 s; dt=1 ms with 160 samples, ending at 0.159 s.

Settings are float32, accuracy=4, PML=12 cells, boundary_buffer=5, memory="boundary", and fixed max_vel=2200 m/s. The installed planner returned a 1 ms internal step and substep ratio 1. All four sides use PML, with no free surface. The trainable region is `active[4:-4,8:-4]`; shallow and outer cells remain fixed.

```{figure} /_static/tutorials/installation_model.png
:alt: Uniform 1800 m/s model, one source, twelve receivers, and fixed boundary band with metre coordinates
:class: sw-science-image
:figclass: sw-science-figure

Model and geometry of the smallest experiment. The propagator applies its internal injection coefficient to fixed source forcing; do not multiply by −v² dt² again.
```

Run `scalar` once, then set `objective = record.square().mean()` and call `objective.backward()`. This energy-like scalar checks the backward connection. It is not an FWI misfit or a physical-solution error benchmark.

## Recorded data and checks

```{figure} /_static/tutorials/installation_gather.png
:alt: Executed one-shot twelve-receiver synthetic gather with time in seconds, receiver x in metres, and arbitrary amplitude units
:class: sw-science-image
:figclass: sw-science-figure

Server-exported gather on its full symmetric amplitude range, without clipping or per-trace normalization. Amplitudes are not calibrated Pa.
```

| Check | Measured result |
|---|---:|
| Record shape `[shot,receiver,time]` | `[1,12,160]` |
| Maximum absolute amplitude / RMS | 25.1871452 / 4.5169382 |
| Mean-squared-record objective | 20.4027290 |
| Velocity-gradient L2 norm | 0.0019382348 |
| Active-region maximum absolute gradient | 0.0010059283 |
| Fixed-band maximum absolute gradient | 0 |

Records and gradients are finite, with a nonzero active-region gradient. **This is a passed small smoke check, not numerical gradient-correctness, FWI convergence, or multi-GPU validation.** Native preparation, compiled architecture coverage, and this numerical experiment are distinct evidence levels.

Continue with [Simple Gradient Computation](modeling/gradient.md) and [Simple FWI Example](inversion/fwi.md), or download the shorter command-line wiring script from [Quickstart](quickstart.md). The {download}`result summary <../examples/tutorials/results_summary.json>` contains complete experiment settings and exact values.
