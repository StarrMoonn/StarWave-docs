# Frequently Asked Questions

V16 adds the [PyTorch backend](pytorch-backend.md): explicit CPU/CUDA full/checkpoint without propagation native libraries; MPS is supported only for Scalar2D accuracy=4 float32 with backend="torch", full/checkpoint and eager/compile. Other MPS combinations, including Scalar3D, VRZ, VTI, elastic and GSLS, are rejected. No final Mac hardware, 500-epoch FWI or performance acceptance is claimed. AMP remains unsupported. Source Release and PyPI `7.1.0` wheel publication verification are pending. Wheel-platform and CUDA-loading answers below describe historical 7.0.0 / native paths. For Torch memory pressure, evaluate checkpoint intervals; larger intervals do not guarantee lower peaks.

## Why can't pip find a compatible StarWave version?

Check that you are using Linux x86_64, Python 3.10–3.12, and glibc ≥2.35, and update pip. Native Windows Python, macOS, ARM, and older glibc versions do not match this wheel. Do not rename the wheel file to bypass its platform tags.

## Do I need to compile CUDA code?

The historical public 7.0.0 wheel does not require compilation or an nvcc installation. Scalar/VRZ/VTI and elastic boundary require a compatible GPU and driver; elastic full also runs on CPU. If integrity or compatibility checks fail, reinstall the corresponding wheel in the same environment and restart Python. Do not expect the documentation project to contain private build tools.

## Why does prepare_native fail even though import succeeds?

Importing the package does not load the native library. Check CUDA availability in PyTorch 2.5.x, logical device IDs, the Linux C++ runtime, the driver, and package integrity. WSL should use the Windows host's NVIDIA driver. A loading error should not be immediately attributed to missing GPU architecture coverage.

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
