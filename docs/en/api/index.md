# API reference

This reference targets the public StarWave **2.0.0** wheel. Each propagator includes its actual signature, every parameter, return values, gradient scope, notes, and a call example. Parameter types describe accepted runtime values; signatures retain the actual keyword-only boundaries and defaults.

The page organization follows the Sphinx Python API style of the [official Deepwave Usage documentation](https://ausargeo.com/deepwave/usage). All descriptions are newly written from StarWave’s actual contracts. The libraries’ parameter sets, source units, return structures, and differentiability scopes are not interchangeable.

```{toctree}
:maxdepth: 2

scalar
vrz
vti
```

## Propagators at a glance

<span id="starwave.scalar"></span>

- {py:func}`starwave.scalar`: 2D scalar acoustics; velocity model `v`; returns a one-element tuple of recordings.

<span id="starwave.vrz"></span>

- {py:func}`starwave.vrz`: 2D variable-density acoustics; `v` plus exactly one `impedance` / `density` parameterization.

<span id="starwave.vti"></span>

- {py:func}`starwave.vti`: 2D/3D acoustic VTI; `vp, epsilon, delta, rho`; returns recordings in the selected component order.

Notation: `B` is the number of shots, `S` the sources per shot, `R` the receivers per shot, `T` the number of user time samples, and `D` the number of spatial dimensions. Sources and receivers use integer grid indices in the physical model, not coordinates in meters, and do not include PML offsets.

## Native runtime

```{py:function} starwave.native_status() -> dict

Queries current native-library status without compiling or propagating.

:returns: Status dictionary. Common entries include `library_exists`, `library_loaded`, `torch_version`, `torch_cuda_version`, and `cuda_available`. Neither existence nor successful loading demonstrates numerical acceptance.
:rtype: `dict`
```

```{py:function} starwave.prepare_native(device_ids: list[int] | tuple[int, ...]) -> dict

Validates and preloads an existing native library on the main thread for subsequent propagation or DataParallel use. Does not compile.

:param device_ids: Required. Nonempty, duplicate-free collection of nonnegative visible logical CUDA device IDs; Booleans are rejected. Use the device-visibility mapping of the current process.
:type device_ids: `list[int] | tuple[int, ...]`
:returns: Dictionary containing native status, `selected_device_ids`, and preparation messages. Successful initialization is not a GPU numerical test.
:rtype: `dict`
```

## Native runtime notes

Use `native_status()` to diagnose installation and loading. Call `prepare_native()` on the main thread before propagation or DataParallel begins. Neither function compiles the library or replaces GPU numerical acceptance. Logical device `0` in the example must be visible to the current process.

## Native runtime examples

After installing the public wheel and a matching PyTorch version, query the status. The subsequent preparation call requires an available CUDA device.

```python
import starwave

status = starwave.native_status()
print(status)
prepared = starwave.prepare_native([0])
```

## Other exported names and scope

`ScalarIllumination`, `IlluminationFields`, and `precondition_gradient` are verified exported names related to scalar illumination. This edition does not yet provide a complete lifecycle tutorial for them; the propagator pages explain the usage boundaries of the `illumination` parameter.

StarWave 2.0.0 has no public `starwave.Scalar` wrapper class. The tutorials define their own Module wrapper. Do not add another library’s class names, state parameters, `nt`, or storage options directly to these calls.

{ref}`genindex`
