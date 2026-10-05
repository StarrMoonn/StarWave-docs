# Installation

## Runtime requirements

| Component | StarWave 2.0.0 wheel requirement |
|---|---|
| Operating system | Linux x86_64, glibc ≥ 2.35 |
| C++ runtime | libstdc++ providing GLIBCXX_3.4.30 / CXXABI_1.3.13 symbols or newer |
| Python | 3.10–3.12 |
| PyTorch / NumPy | PyTorch 2.5.x; NumPy ≥ 1.23 |
| GPU | NVIDIA CUDA GPU; propagation has no CPU fallback |
| Compilation targets | SASS 70/75/80/86/89/90 and compute80 PTX |

Installing the prebuilt wheel does not require nvcc or a local CUDA Toolkit. A conservative target for the Linux driver is 520.61.05 or newer, and the driver must also support the actual GPU. WSL uses the Windows host driver; see the [WSL guidance](wsl.md). The A30 and RTX 4060 Laptop are covered by the compilation targets; device-level numerical and performance tests of the public 2.0.0 wheel are still pending. A separate development-version A30 run is documented below.

The version information above comes from the [PyPI 2.0.0 release description](https://pypi.org/project/starwave/2.0.0/). The public release contains only Linux wheels, with no sdist. This wheel cannot be used directly on native Windows, macOS, or ARM hosts.

## Create a runtime environment

Run the following commands in a Linux or WSL 2 Linux terminal that meets the requirements. Choose Python 3.10–3.12; this example uses 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --only-binary=starwave starwave==2.0.0
python -m pip check
```

The PyTorch installation command follows the [PyTorch previous-versions page](https://pytorch.org/get-started/previous-versions/). `cu118` is the starting configuration for validating this version; it does not mean that other PyTorch versions have been validated. Restart Python or the notebook kernel after upgrading StarWave.

## Check imports and the native library

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
NVIDIA A30 · Python 3.10.18 · PyTorch 2.5.1 / CUDA 11.8 · installed StarWave **0.1.0.dev9**. The server results below check a small workflow in this development-version environment. They do not establish device validation of the public **2.0.0 wheel** described above.
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
