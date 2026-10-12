# Frequently Asked Questions

The current source manual covers published V17 and its Born/flat-top/import contracts. **PyPI 8.0.0 is not published; 7.1.0 remains V16.** Features described for V17 require the matching authorized source. See [installation](installation.md) and [Born Function](usage.md#born-function).

V16 adds the [PyTorch backend](pytorch-backend.md): explicit CPU/CUDA full/checkpoint without propagation native libraries; MPS is supported only for Scalar2D accuracy=4 float32 with backend="torch", full/checkpoint and eager/compile. Other MPS combinations, including Scalar3D, VRZ, VTI, elastic and GSLS, are rejected. No final Mac hardware, 500-epoch FWI or performance acceptance is claimed. AMP remains unsupported. [PyPI 7.1.0](https://pypi.org/project/starwave/7.1.0/) is published for Linux x86_64 only. Mac MPS requires authorized complete source. Wheel-platform and CUDA-loading answers below describe the current Linux wheel and native paths. For Torch memory pressure, evaluate checkpoint intervals; larger intervals do not guarantee lower peaks.

## Why can't pip find a compatible StarWave version?

Check that you are using Linux x86_64, Python 3.10–3.12, and glibc ≥2.35, and update pip. Native Windows Python, macOS, ARM, and older glibc versions do not match this wheel. Do not rename the wheel file to bypass its platform tags.

## Do I need to compile CUDA code?

The public 7.1.0 wheel does not require compilation or an nvcc installation. Native scalar/VRZ/VTI and elastic boundary require a compatible GPU and driver; elastic full and GSLS native_cpu also run on CPU. Explicit Torch full/checkpoint paths need no propagation native libraries. If integrity or compatibility checks fail, reinstall the corresponding wheel in the same environment and restart Python. Do not expect the documentation project to contain private build tools.

## Why does prepare_native fail even though import succeeds?

V17 import prepares existing native libraries and shared dispatch without compiling, initializing CUDA or propagating. Explicit prepare helpers remain optional device checks; import success does not prove a usable CUDA device. Missing unbuilt libraries permit supported explicit Torch use, but a retained invalid library/build raises rather than being silently ignored. Set build paths before import, rebuild matching source, and restart after changes. The older 7.1.0/V16 wheel uses explicit native preparation. For its CUDA path, check PyTorch 2.5.x CUDA availability, logical device IDs, the C++ runtime, driver and package integrity. WSL uses the Windows host driver; do not assume a loading error means missing architecture coverage.

## Where are Born and its build command?

Born is part of V17 source (`starwave.scalar_born` / `starwave.ScalarBorn`), with its own native library. Frozen V17 requires `python compile_born.py --arch 80` after its three-library `compile_all.py`; later reviewed dev `3f2b24d` includes Born in the four-library unified build. `python compile_born.py --cpu-only` remains the standalone CPU option. Explicit Torch Born CPU/CUDA full/checkpoint requires no native Born build. The separate VRR extension retains its own package and build. None of these source additions is supplied by the 7.1.0 wheel.

## Does setting pml_width to zero give every equation the same free surface?

No. V17 accepts different face layouts and physical closures per equation. Scalar/VRZ allow only the last-axis top face zero, with equal positive widths elsewhere; their exterior halo is fixed zero, not an odd-reflection surface, and scalar illumination is rejected. VTI uses a zero-normal-traction top, elastic a first-axis improved-vacuum surface, and GSLS a 2D pressure-release closure. Born independently accepts nonnegative per-face widths with fixed-zero ghost cells. Follow [model conventions](modeling/conventions.md) and the individual [Usage](usage.md) contracts; do not transfer Born's broader width domain to scalar.

V18 scalar (reviewed at dev `e040aa9b8225ab387f7e4853bfe6162f5351f5d5`) adds arbitrary zero-face subsets (positive widths still equal), trainable 2D/3D sources, optional differentiable terminal physical fields and diagnostic forward callbacks. These source features are absent from frozen V17/its 8.0.0 release baseline and from 7.1.0. See [Scalar Function](usage.md#scalar-function) for the 23-parameter contract; the older 20-parameter release snapshot remains separately pinned. VRZ and VTI still reject arbitrary zero faces. Use the dev scalar terminal return for a differentiable full-field loss; its callback snapshots are for observation only.

[Release V18](https://github.com/StarrMoonn/StarWave/releases/tag/V18) is published at `fba1f496b890f2dbaf3fbb3011d39f6d723cfea1`, preserving the reviewed scalar API from dev `e040aa9`. [PyPI 9.0.0](https://pypi.org/project/starwave/9.0.0/) is now published with these features for supported Linux x86_64 hosts; see [installation](installation.md) for requirements and commands. V17 and the unpublished 8.0.0 baseline remain unchanged; 7.1.0 is the historical V16 wheel.

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
