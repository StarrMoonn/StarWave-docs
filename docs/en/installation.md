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

Installing the prebuilt wheel does not require nvcc or a local CUDA Toolkit. A conservative target for the Linux driver is 520.61.05 or newer, and the driver must also support the actual GPU. WSL uses the Windows host driver; see the [WSL guidance](wsl.md). The A30 and RTX 4060 Laptop are covered by the compilation targets, but numerical and performance tests on actual devices are still pending.

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
