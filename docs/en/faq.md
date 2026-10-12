# Frequently Asked Questions

[V19 Release](https://github.com/StarrMoonn/StarWave/releases/tag/V19) is published. This manual follows frozen source `0.1.0.dev19`, commit `46d32877e3336d531520a696d5f2e074e6f4ef74`. [PyPI **9.1.0**](https://pypi.org/project/starwave/9.1.0/) is published with the V19 source contract and integrated VRR. The existing [9.0.0 wheel](https://pypi.org/project/starwave/9.0.0/) remains V18 and excludes integrated VRR.

The [PyTorch backend](pytorch-backend.md) offers explicit CPU/CUDA full/checkpoint without propagation native builds. MPS remains Scalar2D accuracy=4 float32 only and requires source installation. AMP is unsupported. The published 9.0.0 Linux wheel provides the V18 contract; VRR requires V19 source until 9.1.0 is published and verified. Larger checkpoint intervals do not guarantee lower peak memory.

## Why can't pip find a compatible StarWave version?

Check that you are using Linux x86_64, Python 3.10–3.12, and glibc ≥2.35, and update pip. Native Windows Python, macOS, ARM, and older glibc versions do not match this wheel. Do not rename the wheel file to bypass its platform tags.

## Do I need to compile CUDA code?

The published 9.0.0 wheel does not require compilation or an nvcc installation. Native scalar/VRZ/VTI and elastic boundary require a compatible GPU and driver; elastic full and GSLS native_cpu also run on CPU. Explicit Torch full/checkpoint paths need no propagation native libraries. If integrity or compatibility checks fail, reinstall the corresponding wheel in the same environment and restart Python. Do not expect the documentation project to contain private build tools.

## Why does prepare_native fail even though import succeeds?

V19 import prepares existing native libraries and shared dispatch without compiling, initializing CUDA or propagating. Explicit prepare helpers remain optional device checks; import success does not prove a usable CUDA device. Missing unbuilt libraries permit supported explicit Torch use, but a retained invalid library/build raises rather than being silently ignored. Set build paths before import, rebuild matching source, and restart after changes. The older 7.1.0/V16 wheel uses explicit native preparation. For its CUDA path, check PyTorch 2.5.x CUDA availability, logical device IDs, the C++ runtime, driver and package integrity. WSL uses the Windows host driver; do not assume a loading error means missing architecture coverage.

## Where are Born and its build command?

V19 includes `starwave.scalar_born` / `starwave.ScalarBorn` and `starwave.vrr` in the root package. `python compile_all.py --arch 80` builds core, elastic, GSLS, Born and VRR. `python compile_born.py --cpu-only` is the native CPU full option; `python -m starwave._vrr.build --arch 80` rebuilds only VRR. Explicit Torch Born/VRR CPU/CUDA full/checkpoint needs no native build. See [installation](installation.md).

## Does setting pml_width to zero give every equation the same free surface?

No. V19 scalar accepts arbitrary zero-face subsets, including all-zero PML, with equal positive widths elsewhere; zero faces use fixed-zero halos, not odd reflection. VRZ retains its last-axis top-only zero-face restriction. VTI uses zero normal traction, elastic a first-axis improved-vacuum surface, GSLS a 2D pressure-release closure, and VRR its documented top-surface restrictions. Born independently accepts nonnegative per-face widths with fixed-zero ghosts. Follow [model conventions](modeling/conventions.md) and the individual [Usage](usage.md) contracts.

V19 preserves the V18 23-parameter scalar contract: arbitrary zero-face subsets (positive widths must remain equal), trainable 2D/3D sources, differentiable terminal physical fields and diagnostic forward callbacks. Terminal returns can carry a loss; callback snapshots are detached observations and cannot carry loss gradients. See [Scalar Function](usage.md#scalar-function).



## Can I run it on an A30 or RTX 4060 Laptop?

The corresponding sm_80 / sm_89 architectures are covered by compilation. This alone does not establish numerical correctness, stability, or acceptable performance on actual devices. On your actual system, work through loading, small-model recordings, gradients, long time series, and multi-GPU tests in sequence.

## Why is the first waveform sample not exactly zero, or why does the amplitude differ from another package?

Internal time resampling can introduce FFT endpoint ringing. Also check source units, the sampling clock, the physical component being recorded, and spatial staggering. VTI sources and scalar sources are not directly interchangeable. Similar Python call syntax does not guarantee numerical equivalence with other software.

## Why do I see a warning about six points per wavelength?

This is a spatial-resolution diagnostic. `pml_freq` is a frequency proxy, not an automatic measurement of the actual waveform spectrum. Reducing `dt` does not improve spatial grid resolution. Reassess the grid based on the target frequency band and model velocities.

## What should I do if I run out of GPU memory?

First reduce the number of shots in the current batch, the model size, or the time-series length, and assess the resource differences between full and boundary modes. Multi-GPU execution splits work by shot, so a single shot that is too large can still fail. The library does not automatically reduce experimental parameters. Passing with a smaller configuration does not validate the original configuration.

## Why does backward fail even when the loss is finite?

A finite loss does not guarantee that its gradient with respect to the records is finite. Check objective-function normalization, nonlinear operations, and precision. Each forward pass supports only one first-order backward pass. Also check parameter validity, propagation stability, and model gradients separately. Silently replacing NaN/Inf values cannot establish that the computation is correct.

## Do I need a GPU to build the documentation?

No. Sphinx only renders manually written documentation and example text; it does not import StarWave or run CUDA.
