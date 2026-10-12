# Installation

[V19 Release](https://github.com/StarrMoonn/StarWave/releases/tag/V19) is published. This manual follows frozen source `0.1.0.dev19`, commit `46d32877e3336d531520a696d5f2e074e6f4ef74`. [PyPI **9.1.0**](https://pypi.org/project/starwave/9.1.0/) is published with the V19 source contract and integrated VRR. The existing [9.0.0 wheel](https://pypi.org/project/starwave/9.0.0/) remains V18 and excludes integrated VRR.

V19 integrates {ref}`VRR <vrr>` as `starwave.vrr`. Root installation and `compile_all.py` include VRR; no separate `starwave_vrr` installation is needed. VRR supports explicit Torch CPU/CUDA float32/64 full/checkpoint with eager/compile, and native CUDA float32 full/boundary. Device, gradient and boundary restrictions remain equation-specific.

V19 preserves the V18 23-parameter scalar contract: arbitrary zero-face subsets (positive widths must remain equal), trainable 2D/3D sources, differentiable terminal physical fields and diagnostic forward callbacks. Terminal returns can carry a loss; callback snapshots are detached observations and cannot carry loss gradients. See [Scalar Function](usage.md#scalar-function).

## Install V19 source

Use the complete authorized V19 source. Install its requirements in the intended PyTorch environment, then run from the source root:

```bash
python -m pip install --no-deps -e .
```

Root registration includes VRR and does not implicitly compile native libraries. A complete checkout can also run directly without registration. The source repository remains private; this manual does not distribute its implementation or model data.

Explicit `backend="torch"` supports CPU/CUDA full/checkpoint without propagation native libraries or nvcc; choose `execution="eager"` or `"compile"` separately from the model device. Compile uses the installed PyTorch Inductor toolchain and raises on failure without eager fallback. MPS is limited to Scalar2D accuracy=4 float32 with Torch full/checkpoint and eager/compile. Born, Scalar3D, VRZ, VRR, VTI, elastic, GSLS and other MPS combinations are rejected. AMP remains unsupported. See the {ref}`standalone CPU example <torch-example>`.

## V19 native builds

Use a compatible CUDA Toolkit/C++ toolchain and complete matching source. Build all five native groups explicitly:

```bash
python compile_all.py --arch 80
```

This builds core (scalar/VRZ/VTI), elastic, GSLS, Born and VRR. Use SM `80` for A30, `89` for RTX 4060, or the actual GPU architecture. The older `combined_build.py` covers core/elastic only. Optional individual builds are:

```bash
# Born native CPU full:
python compile_born.py --cpu-only
# GSLS CUDA or native CPU:
python compile_visco_gsls.py --arch 80
python compile_visco_gsls.py --backend cpu
# VRR CUDA only:
python -m starwave._vrr.build --arch 80
```

VRR needs no separate package installation. Its native build uses nvcc/C++17 without Torch C++ headers or libtorch linkage; GPU execution still requires CUDA-enabled Torch and compatible driver/runtime support. Pure Torch Born and VRR need no native build.

`import starwave` binds available native libraries and shared dispatch before returning, including integrated VRR. It does not compile, initialize CUDA, select devices, allocate wavefields or propagate. Missing unbuilt libraries permit supported explicit Torch use; loading/binding errors still fail. Import is not a runtime source/library hash, fingerprint, ABI/capability-version or artifact-identity audit. Rebuild matching source and restart Python/Jupyter after changes; manual `prepare_*` calls remain optional device checks. Do not mix old libraries with new source. Successful compilation/import is not GPU numerical validation.

GSLS outputs to `native/build/gsls/`, or the `gsls/` subdirectory of `STARWAVE_BUILD_DIR`. Configure build paths before import and use the same paths for build and execution. Existing tutorials, figures and measured results retain their historical versions.

(published-v18-binary-channel)=
(runtime-requirements)=
## Published V19 binary channel

The following installs published V19 / 9.1.0, including integrated VRR, Born and the 23-parameter scalar API. Mac MPS requires the complete source installation above; this Linux wheel cannot be installed on macOS. Source acceptance of compile mode does not guarantee compiler availability on every PyTorch version or device.

**Install the published V19 / 9.1.0 wheel**

Use Linux x86_64 / glibc ≥ 2.35, Python ≥ 3.10 and < 3.13, PyTorch ≥ 2.5 and < 2.6, and NumPy ≥ 1.23. There is no macOS wheel or sdist. A prebuilt installation needs no nvcc or local CUDA Toolkit; native CUDA execution still requires a supported NVIDIA driver/device. The following uses the publication-tested PyTorch 2.5.1 CUDA 11.8 configuration; CPU-only use can select a matching PyTorch 2.5.x CPU build. This does not expand backend/device eligibility.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --only-binary=starwave starwave==9.1.0
python -m pip check
```

Restart Python/Jupyter after upgrading. The public wheel's Python 3.10–3.12 install/smoke checks passed; target-GPU numerical acceptance remains separate. Older wheel requirements and migration commands are archived in [Release Notes](release-notes.md#historical-v16-installation-and-migration).

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
# Optional explicit V19 device check:
starwave.prepare_native([0])
```

V19 `import starwave` prepares available core/elastic/Born/GSLS libraries and shared dispatch before returning; integrated VRR prepares its own available library. It does not compile, initialize CUDA, select devices, allocate wavefields or propagate. Set build paths before import, and rebuild/restart after updates. Missing unbuilt libraries permit supported explicit Torch use; retained invalid builds fail. Manual prepare calls remain optional device checks, not mandatory serial warm-up. Neither import nor preparation replaces numerical forward/gradient checks; continue with [Quickstart](quickstart.md).

The documentation site's build environment is separate from the computational runtime environment. Reading or building the documentation does not require a GPU.


Elastic retains an optional explicit V19 preparation entry point (required by older native workflows): call `starwave.prepare_elastic([0])` on the GPU main thread, or `starwave.prepare_elastic()` with `memory="full"` for CPU. See {ref}`Elastic Function <elastic>` for parameters and vp/vs/rho conversion. Native Scalar3D has no CPU propagation path; Torch Scalar3D supports CPU/CUDA. GPU, DataParallel and long-FWI acceptance scopes are listed in [Status](status.md).

## GSLS preparation and backend selection

GSLS retains an optional explicit V19 preparation entry point (required by older native workflows): call `starwave.prepare_visco_gsls(backend="cuda", device="cuda:0")` on the CUDA main thread, or `starwave.prepare_visco_gsls(backend="native_cpu")` for CPU. `starwave.visco_gsls_native_status(backend="cuda")` only reports status. Preparation defaults to CPU while propagation defaults to CUDA; select explicitly, without automatic fallback. `backend="torch"` supports full/checkpoint and eager/compile without a native build. See the {ref}`GSLS API <visco-gsls>` for the complete contract and CPU example.

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
