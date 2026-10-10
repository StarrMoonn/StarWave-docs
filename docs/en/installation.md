# Installation

The current API targets the additive [PyTorch backend](pytorch-backend.md) in V16 / `0.1.0.dev16`; source Release and target PyPI `7.1.0` wheel publication verification are pending. Historical PyPI **7.0.0** is a V14 binary without GSLS or the unified Torch backend and cannot run the new examples.

The source repository is private; use complete authorized V16 source. No unverified new-version pip command is advertised before publication verification. Historical 7.0.0 requirements and commands remain below as a clearly separate legacy channel.

## V16 pure Torch installation and execution

Run directly from the complete V16 source root, or optionally register it:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

Explicit `backend="torch"` supports CPU/CUDA full/checkpoint without propagation native libraries, nvcc or compile_all.py. Execution mode and model device are separate selections. Compile uses the current PyTorch Inductor toolchain and raises on failure without eager fallback. MPS is supported only for Scalar2D accuracy=4 float32 with backend="torch", full/checkpoint and eager/compile. Other MPS combinations, including Scalar3D, VRZ, VTI, elastic and GSLS, are rejected. No final Mac hardware, 500-epoch FWI or performance acceptance is claimed. AMP remains rejected. Start with the {ref}`standalone CPU example <torch-example>`.

## V16 native build

Native paths require matching complete source/library builds. The new Python entries change elastic/GSLS source-closure identity even though scientific CUDA/C++ kernels are unchanged. Rebuild and restart; do not edit an old build manifest. Only pure Torch calls can omit the native build below.


Use an installed CUDA Toolkit/C++ toolchain matching PyTorch. From the complete source root:

```bash
python compile_all.py --arch 80
```

This builds/verifies three CUDA libraries: core (scalar/VRZ/VTI), elastic and GSLS. Use `80` for A30, `89` for RTX 4060, or your actual GPU architecture. `combined_build.py` covers core/elastic only. To build GSLS alone, select CUDA or CPU explicitly:

```bash
# GSLS CUDA
python compile_visco_gsls.py --arch 80
# GSLS native CPU
python compile_visco_gsls.py --backend cpu
```

GSLS source builds output to `native/build/gsls/`, or the `gsls/` subdirectory of the common `STARWAVE_BUILD_DIR` override. Build and runtime must use matching source/library inputs; do not mix old standalone libraries. GSLS ABI 2 / semantics 1 / capabilities 255 and core ABI 2 are checked separately. Restart Python/the notebook kernel after updates. Compilation or loading does not establish GPU numerical acceptance.

A complete source checkout runs without pip registration. Optional registration is separate from compilation and does not implicitly build:

```bash
python -m pip install --no-deps -e .
```

The following wheel requirements and commands describe only the historical 7.0.0 channel.

(runtime-requirements)=
## Historical 7.0.0 wheel requirements

| Component | StarWave 7.0.0 wheel requirement |
|---|---|
| Operating system | Linux x86_64, glibc ≥ 2.35 |
| C++ runtime | libstdc++ providing GLIBCXX_3.4.30 / CXXABI_1.3.13 symbols or newer |
| Python | 3.10–3.12 |
| PyTorch / NumPy | PyTorch 2.5.x; NumPy ≥ 1.23 |
| GPU | NVIDIA CUDA for scalar 2D/3D, VRZ/VTI, and elastic boundary; elastic full and SLS native_cpu support CPU; SLS torch is an explicit reference backend |
| Compilation targets | SASS 70/75/80/86/89/90 and compute80 PTX |

Installing the prebuilt wheel does not require nvcc or a local CUDA Toolkit. A conservative target for the Linux driver is 520.61.05 or newer, and the driver must also support the actual GPU. WSL uses the Windows host driver; see the [WSL guidance](wsl.md). The A30 and RTX 4060 Laptop are covered by the compilation targets; device-level numerical and performance tests of the public 7.0.0 wheel are still pending. A separate development-version A30 run is documented below.

The version information above comes from the [PyPI 7.0.0 release description](https://pypi.org/project/starwave/7.0.0/); see [Documentation Status](status.md) for artifact identity and verification scope. The public release contains only a Linux wheel, with no sdist. This wheel cannot be used directly on native Windows, macOS, or ARM hosts.

## Install the historical 7.0.0 wheel

Run the following commands in a Linux or WSL 2 Linux terminal that meets the requirements. Choose Python 3.10–3.12; this example uses 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --only-binary=starwave starwave==7.0.0
python -m pip check
```

The PyTorch installation command follows the [PyTorch previous-versions page](https://pytorch.org/get-started/previous-versions/). `cu118` is the starting configuration for validating this version; it does not mean that other PyTorch versions have been validated. Restart Python or the notebook kernel after upgrading StarWave.

## Upgrade to V16 source

V16 has one viscoacoustic entry point, `starwave.visco_gsls`; independent `visco_sls` and its preparation/status helpers are removed. Single-mechanism calculations explicitly use `mode="sls_compat", n_mechanisms=1`; changing only the function name would select default Hao1 instead. Source version `0.1.0.dev16` and public wheel `7.0.0` identify different distributions.

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
# In the main thread, use logical device IDs visible to the current process:
starwave.prepare_native([0])
```

In a fresh Python process, before the first native preparation, the wheel should report that the library file exists but has not yet been loaded. A successful `prepare_native` call means initialization and compatibility checks passed; it does not replace numerical validation of forward modeling and gradients. Next, run the small checks in the [Quickstart](quickstart.md).

The documentation site's build environment is separate from the computational runtime environment. Reading or building the documentation does not require a GPU.


Elastic uses a separate preparation entry point: call `starwave.prepare_elastic([0])` on the GPU main thread, or `starwave.prepare_elastic()` with `memory="full"` for CPU. See {ref}`Elastic Function <elastic>` for parameters and vp/vs/rho conversion. Scalar3D has no CPU propagation path. GPU, DataParallel and long-FWI acceptance scopes are listed in [Status](status.md).

## V16 GSLS preparation and backend selection

V16 GSLS uses separate preparation: call `starwave.prepare_visco_gsls(backend="cuda", device="cuda:0")` on the CUDA main thread, or `starwave.prepare_visco_gsls(backend="native_cpu")` for CPU. `starwave.visco_gsls_native_status(backend="cuda")` only reports status. Preparation defaults to CPU while propagation defaults to CUDA; select explicitly, without automatic fallback. `backend="torch"` supports full/checkpoint and eager/compile without a native build. See the [GSLS API](visco-gsls.md) for the complete contract and CPU example.

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
NVIDIA A30 · Python 3.10.18 · PyTorch 2.5.1 / CUDA 11.8 · installed StarWave **0.1.0.dev9**. The server results below check a small workflow in this development-version environment. They do not establish device validation of the public **2.0.0, 4.0.0, 5.0.0, 6.0.0, or 7.0.0 wheel**.
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
