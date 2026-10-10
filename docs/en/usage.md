# Usage

GSLS: [visco_gsls](visco-gsls.md).

```{container} sw-page-toc

**On this page**

- [Scalar Function](#scalar)
- [VRZ Function](#vrz)
- [VTI Function](#vti)
- [Elastic Function](#elastic)
- [Material conversions](#elastic-conversions)
- [starwave.native_status](#native-status)
- [starwave.prepare_native](#prepare-native)
- [starwave.prepare_elastic](#prepare-elastic)
- [Other exported names and scope](#other-exports)
```

This reference targets the published [V15 source Release](https://github.com/StarrMoonn/StarWave/releases/tag/V15) (`0.1.0.dev15`). The scalar/VRZ/VTI/elastic contracts below retain those of the 7.0.0 wheel. [GSLS](visco-gsls.md) is the current viscoacoustic entry point; the public PyPI 7.0.0 wheel does not contain GSLS. `starwave.scalar` now selects 2D or 3D from the model dimension; the existing scalar2D, VRZ, VTI, and elastic contracts are preserved. Each propagator includes its actual signature, every parameter, return values, gradient scope, notes, and a call example. Parameter types describe accepted runtime values; signatures retain the actual keyword-only boundaries and defaults.

See [Release Notes](release-notes.md) for 6.0.0 internal storage and PML-transpose maintenance, and [Scalar](modeling/scalar.md) for 2D storage changes. These introduce no new public parameters; source upgrades require rebuilt matching native libraries.

The page organization follows the Sphinx Python API style of the [official Deepwave Usage documentation](https://ausargeo.com/deepwave/usage). All descriptions are newly written from StarWave’s actual contracts. The libraries’ parameter sets, source units, return structures, and differentiability scopes are not interchangeable.

(propagators)=
## Propagators at a glance

- {py:func}`starwave.scalar`: `2D/3D` scalar acoustics; velocity model `v`; returns a one-element tuple of recordings.

- {py:func}`starwave.vrz`: `2D` variable-density acoustics; `v` plus exactly one `impedance` / `density` parameterization.

- {py:func}`starwave.vti`: `2D/3D` acoustic VTI; `vp, epsilon, delta, rho`; returns recordings in the selected component order.

- {py:func}`starwave.elastic`: `2D/3D` isotropic elasticity; `lamb, mu, buoyancy`; complete final states plus p/velocity records.

- {py:func}`starwave.visco_gsls`: 2D generalized-standard-linear-solid viscoacoustics, fixed variable density and native full/checkpoint; see the dedicated [GSLS API](visco-gsls.md).

Notation: `B` is the number of shots, `S` the sources per shot, `R` the receivers per shot, `T` the number of user time samples, and `D` the number of spatial dimensions. Sources and receivers use integer grid indices in the physical model, not coordinates in meters, and do not include PML offsets.

(scalar)=
## Scalar Function

```{py:function} starwave.scalar(v: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, accuracy: int=8, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float | int=0.0, time_pad_frac: float | int=0.0, time_taper: bool=False, illumination: ScalarIllumination | None=None) -> tuple[torch.Tensor]

Propagates scalar acoustic waves on a 2D or 3D grid selected by `v.ndim`, with a batch of independent shots. Both dimensions support one first-order velocity gradient; 3D also supports source-waveform gradients. Coordinates and numerical settings are not differentiable. Each call starts with fresh propagation state and does not return final wavefields.

:param v: **Required; positional or keyword.** Two-dimensional `[N0,N1]` or three-dimensional `[N0,N1,N2]` model, with at least 2 cells per dimension; CUDA float32, typically in m/s. `v.ndim` selects the dimension. Coordinates `[i,j]` / `[i,j,k]` directly index `v[i,j]` / `v[i,j,k]`; the 3D examples use `[nx,ny,nz]`, with no automatic physical-axis permutation. All values must be finite. Signed and zero values are accepted without guaranteeing physical validity. Model-axis transformations preserve autograd; `requires_grad=True` requests a velocity gradient.
:type v: `torch.Tensor`
:param grid_spacing: **Required; positional or keyword.** Positive, finite spacing, typically in m. In 2D, accepts one value or two equal values. In 3D, accepts one value or three values in model-axis order (for example `[dx,dy,dz]`); unequal spacings are supported. Booleans, zero/negative values, and mismatched sequence lengths are rejected.
:type grid_spacing: `float | int | list[float] | tuple[float, ...]`
:param dt: **Required; positional or keyword.** Positive, finite sampling interval of the user input/output, typically in s. Internally, the CFL condition may require a smaller time step; the returned sampling interval remains this value. Tensor and Boolean inputs are not accepted, and this parameter is not differentiated.
:type dt: `float | int`
:param source_amplitudes: **Required; keyword-only.** Real tensor `[B,S,T]` with nonempty dimensions. Values must be finite and representable as float32; conversion to the model device and float32 preserves the supported gradient chain. In 2D the source must be fixed (`requires_grad=True` is rejected); 3D supports first-order source gradients, alone or together with velocity gradients. This is normalized forcing `f`, not a per-step pressure increment; its unit is Pa/m² if the wavefield is in Pa and length in m. `[T]` and `[B,T]` are not accepted: explicitly expand shot/source dimensions for a shared waveform.
:type source_amplitudes: `torch.Tensor`
:param source_locations: **Required; keyword-only.** Fixed integer tensor `[B,S,D]`, where `D=v.ndim`; `[B,D]` is also accepted for a single source per shot. `torch.long` is recommended. Coordinates directly index the physical grid in input-model axis order; 3D `[i,j,k]` means `v[i,j,k]`. No PML offsets or out-of-bounds values are allowed. Each source has a waveform, and coincident sources within one shot add their increments.
:type source_locations: `torch.Tensor`
:param receiver_locations: **Required; keyword-only.** Fixed integer tensor `[B,R,D]`, where `D=v.ndim` and `R>0`; `torch.long` is recommended. The shot count matches the sources. Coordinates index the physical grid in input-model axis order. Duplicate positions are allowed. Receivers cannot be omitted, and an empty tensor cannot request propagation without recordings.
:type receiver_locations: `torch.Tensor`
:param accuracy: **Default `8`; keyword-only.** Spatial finite-difference order, one of `2,4,6,8`, not an error tolerance. Affects stencil extent, computational cost, and the minimum boundary buffer. Booleans and other orders are rejected.
:type accuracy: `int`
:param pml_freq: **Default `25.0`; keyword-only.** Positive, finite frequency used to configure the PML, typically in Hz; also used for spatial-sampling diagnostics. It does not generate a waveform or automatically measure the source frequency. Choose it for the actual experiment.
:type pml_freq: `float | int`
:param pml_width: **Default `20`; keyword-only.** Positive integer thickness in grid cells, applied to every face. Also accepts four equal integers in 2D or six equal integers in 3D, paired as the beginning/end of each model axis. Zero widths, asymmetric faces, and requesting a free surface with zero width are unsupported.
:type pml_width: `int | list[int] | tuple[int, ...]`
:param boundary_buffer: **Default `5`; keyword-only.** Nonnegative buffer in spatial grid cells, not a number of time steps or a checkpoint interval. Boundary mode requires at least `accuracy // 2 + 1`, which is 5 at eighth order; full mode allows 0. Total padding on each side is `pml_width + boundary_buffer + accuracy // 2`.
:type boundary_buffer: `int`
:param memory: **Default `'boundary'`; keyword-only.** Backpropagation-history strategy. `"boundary"` stores pressure strips and two terminal pressure fields for reconstruction; the six 3D faces each have width `M=accuracy//2` (Radius-M). In 3D, `"full"` stores unscaled `Lap(u)` history over the whole padded volume at every internal step and is intended for small comparisons. Source-only 3D gradients also retain the selected history; a forward with neither velocity nor source gradients retains none. Full mode can exhaust GPU memory. There is no automatic fallback, CPU/disk offload, or `"checkpoint"` option.
:type memory: `str`
:param max_vel: **Default `None`; keyword-only.** CFL/PML velocity envelope, typically in m/s. With `None`, each call replans from the current model’s `max(abs(v))`. An explicit value must be positive, finite, and cover that maximum; it is not a clipping limit. An all-zero model requires an explicit positive value. Extrema, time substeps, and PML configuration are not differentiated.
:type max_vel: `float | int | None`
:param freq_taper_frac: **Default `0.0`; keyword-only.** Finite fraction in `[0,1]` controlling the cosine taper of high-frequency FFT bins during temporal resampling; the bin count is obtained by truncating the proportional count to an integer. During resampling, the last positive rFFT bin is suppressed even when this fraction is 0, including for odd lengths; preservation of every frequency is not guaranteed. This setting does not change the signal when the internal resampling ratio is 1.
:type freq_taper_frac: `float | int`
:param time_pad_frac: **Default `0.0`; keyword-only.** Finite fraction in `[0,1]`. Appends `int(time_pad_frac*T)` user samples of zeros before resampling and removes them afterwards. Does not increase the public recording length or the actual propagation duration. It does not change the signal when the internal resampling ratio is 1.
:type time_pad_frac: `float | int`
:param time_taper: **Default `False`; keyword-only.** Whether to apply a nonperiodic Hann window during temporal resampling: after upsampling and before downsampling. Changes both the signal and its backpropagation transpose. Accepts only Booleans, not the integers 0/1. It does not change the signal when the internal resampling ratio is 1.
:type time_taper: `bool`
:param illumination: **Default `None`; keyword-only.** Optional fresh `ScalarIllumination` collector in 2D, requiring enabled gradient tracking and a trainable model. Collects detached source, receiver, and geometric-product statistics; it is sealed after forward and reduced after one backward. In 3D only `None` is accepted; any other value raises `NotImplementedError`. `None` allocates no illumination buffers. These statistics are not an exact Hessian and do not automatically precondition gradients.
:type illumination: `ScalarIllumination | None`
:returns: **`(receiver_amplitudes,)`**, a tuple with exactly one element. The recordings are a contiguous CUDA float32 tensor `[B,R,T]` on the same device as the model, at nominal times `0, dt, ..., (T-1)*dt`. They are pressure-like recordings whose scale depends on the source and unit system. Extract them with `[0]` or `[-1]`. No final wavefield or PML state is included.
:rtype: `tuple[torch.Tensor]`
```

(scalar-details)=
### Sources and autograd

The public equation convention is `u_tt = v² (Lap(u) - f)`. The user supplies fixed `f` in 2D and may train `f` in 3D; the internal source increment is `-v_source² * internal_dt² * U(f)`, where `U` is internal upsampling. Do not multiply by `-v² dt²` again externally. The velocity-scaling chain at source locations is retained in the model gradient.

Post-step recordings are aligned to the user clock before downsampling by prepending zero and dropping the final step. The transposes of temporal resampling and shifting remain in autograd; noncausal FFT filtering can cause ringing in the initial user samples.

Model gradients are conditional on a fixed extended model, CFL configuration, and PML. When replicate padding at the physical edges is regenerated, the current returned gradient does not include the transpose accumulation of the complete extension chain. It must not be treated as the total derivative of the entire boundary-rebuilding process; switching to full mode does not remove this limitation.

(scalar-examples)=
### Examples

The following is a call fragment assuming valid models, acquisition inputs, and the native library have already been prepared; it is not a standalone program. Every omitted optional parameter uses the default shown above.

```python
import starwave

receiver_amplitudes, = starwave.scalar(
    v, grid_spacing=10.0, dt=0.001,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
)
```

For complete input construction without external data files and one FWI update, see [Quickstart](quickstart.md) and [FWI](inversion/fwi.md). The default `pml_freq=25.0` does not automatically track the source frequency.


(scalar-time-sampling)=
### CFL and internal temporal resampling

Scalar uses the conservative coefficient 0.6, every model-axis spacing, and `max(abs(v))` (or the checked `max_vel`) to compute a time-step bound. The internal ratio `r` is a positive integer satisfying that bound, with `internal_dt=dt/r` and `internal_nt=T*r`. Unequal 3D spacings participate axis by axis. `accuracy` does not enable a different public CFL parameter. Time planning, the velocity envelope, and PML coefficients are fixed setup values outside autograd.

Sources are FFT-upsampled before source-location velocity scaling. Records are time-aligned before downsampling to `[B,R,T]`. Backward uses the actual transposes of these operations, not simple repetition or decimation. At ratio 1, valid taper/padding settings do not alter the signal. Time substeps cannot repair spatial dispersion.

(scalar-memory)=
### 3D memory: full and Radius-M boundary

`M=accuracy//2` is the pressure-Laplacian stencil radius, distinct from `boundary_buffer` and `pml_width`. Boundary mode stores six pressure faces of exactly M cells plus two terminal pressure fields, without a full time-volume history or automatic fallback to full. Full mode stores unscaled `Lap(u)` volume history at every internal step. Both modes retain propagation/adjoint workspaces; 3D CPML memory uses directional slabs. Source-only gradients also retain the selected history. Memory grows with internal time steps and shot count, so boundary mode does not guarantee production-scale 3D fits a GPU. See the exact payload formulas in {ref}`Scalar3D reconstruction <reconstruction-scalar3d>`.

(scalar-3d-example)=
### Runnable 3D call: first-order velocity and source gradients

With 7.0.0 installed and visible logical CUDA device 0 available, the following is a standalone small example. The model uses `[x,y,z]`, spacing is `[dx,dy,dz]`, and coordinates are grid indices. The loss only checks gradient wiring. Syntax and interface have been checked; this documentation update did not execute the GPU example, so these assertions are not numerical acceptance results.

```python
import torch
import starwave

starwave.prepare_native([0])
device = torch.device("cuda:0")
v = torch.full((24, 20, 16), 1800.0, device=device,
               dtype=torch.float32, requires_grad=True)
t = torch.arange(96, device=device, dtype=torch.float32) * 0.001
a = (torch.pi * 15.0 * (t - 0.04)).square()
source_amplitudes = ((1 - 2 * a) * torch.exp(-a)).reshape(1, 1, -1)
source_amplitudes = source_amplitudes.detach().requires_grad_()
source_locations = torch.tensor([[[12, 10, 3]]], device=device,
                                dtype=torch.long)
receiver_locations = torch.tensor(
    [[[i, 10, 3] for i in range(4, 20)]], device=device, dtype=torch.long)
receiver_amplitudes, = starwave.scalar(
    v, grid_spacing=(10.0, 12.0, 8.0), dt=0.001,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    accuracy=4, pml_freq=15.0, pml_width=8, boundary_buffer=5,
    memory="boundary", max_vel=2000.0,
    freq_taper_frac=0.0, time_pad_frac=0.0, time_taper=False,
    illumination=None,
)
assert receiver_amplitudes.shape == (1, 16, 96)
loss = receiver_amplitudes.square().mean()
loss.backward()
assert v.grad is not None and source_amplitudes.grad is not None
assert torch.isfinite(receiver_amplitudes).all()
assert torch.isfinite(v.grad).all()
assert torch.isfinite(source_amplitudes.grad).all()
```

(scalar-notes)=
### Notes

- Supports CUDA FP32 on the default stream only; each forward permits one first-order backward pass. Source gradients remain unsupported in 2D and are supported at first order in 3D. Higher-order derivatives, AMP, CUDA graphs, custom streams, CPU propagation, and public initial/final states are unsupported.
- CFL time substeps cannot compensate for inadequate spatial sampling. Spatial-resolution warnings are diagnostics, not certificates of accuracy or stability.
- For DataParallel, place the complete model in a Module and split acquisition inputs only along the shot dimension; see [Multi-GPU introduction](inversion/dataparallel.md).
- This page’s interface contract has been checked. Target-GPU numerical, performance, and FWI acceptance tests have not been completed.

(vrz)=
## VRZ Function

```{py:function} starwave.vrz(v: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, impedance: torch.Tensor | None=None, density: torch.Tensor | None=None, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, accuracy: int=8, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float | int=0.0, time_pad_frac: float | int=0.0, time_taper: bool=False, illumination: None=None) -> tuple[torch.Tensor]

Performs two-dimensional acoustic propagation using velocity and impedance (or density), with batched shots. Specify exactly one parameterization through `impedance` or `density`. The output supports one first-order gradient with respect to the selected model parameters; source waveforms and acquisition coordinates remain fixed.

:param v: **Required; positional or keyword.** Two-dimensional model `[N0,N1]`, with at least 2 cells in each dimension; CUDA float32. Typical unit: m/s. Coordinates `[i,j]` directly index `v[i,j]`; physical axis order is not inferred automatically. Velocity must be positive and finite; scalar’s signed/zero model domain is not accepted. Set `requires_grad=True` to request a model gradient.
:type v: `torch.Tensor`
:param grid_spacing: **Required; positional or keyword.** Positive, finite grid spacing, typically in m. Accepts one value or two equal values; scalar/VRZ do not support unequal spacing along the two axes. Booleans, zero/negative values, and mismatched sequence lengths are rejected.
:type grid_spacing: `float | int | list[float] | tuple[float, ...]`
:param dt: **Required; positional or keyword.** Positive, finite sampling interval of the user input/output, typically in s. Internally, the CFL condition may require a smaller time step; the returned sampling interval remains this value. Tensor and Boolean inputs are not accepted, and this parameter is not differentiated.
:type dt: `float | int`
:param impedance: **Default `None`; keyword-only.** Acoustic impedance `Z`, CUDA float32, with the same shape and device as `v`, positive and finite. Typical SI unit: kg/(m²·s), consistent with `v*rho`. Provide exactly one of this parameter and `density`; providing both or neither is invalid. Training this input uses an independent velocity/impedance parameterization.
:type impedance: `torch.Tensor | None`
:param density: **Default `None`; keyword-only.** Density `rho`, CUDA float32, with the same shape and device as `v`, positive and finite. Choose a consistent unit system; the SI unit is kg/m³. The interface uses the differentiable conversion `Z=v*rho`, and gradients include this chain rule. No implicit unit conversion is performed. Provide exactly one of this parameter and `impedance`.
:type density: `torch.Tensor | None`
:param source_amplitudes: **Required; keyword-only.** Fixed real-valued tensor `[B,S,T]`, with all three dimensions nonempty and `requires_grad=True` forbidden. Values must be finite and representable as float32; the tensor is converted to the model device and float32. This is the normalized forcing `f`, not a pressure increment per step; if the wavefield is in Pa and length is in m, its unit is Pa/m². A tensor of shape `[T]` or `[B,T]` alone is not accepted: explicitly expand the shot/source dimensions for a shared waveform.
:type source_amplitudes: `torch.Tensor`
:param source_locations: **Required; keyword-only.** Fixed integer tensor `[B,S,2]`; `[B,2]` is also accepted for a single source per shot. `torch.long` is recommended. Uses physical-grid indices in the input model axis order, without PML offsets or out-of-bounds values. Each source has a corresponding waveform; increments from coincident sources within a shot are added.
:type source_locations: `torch.Tensor`
:param receiver_locations: **Required; keyword-only.** Fixed integer tensor `[B,R,2]`, with `R>0`; `torch.long` is recommended. The shot count must match the sources. Uses physical-grid indices in the input model axis order. Duplicate positions are allowed. Receivers are required; an empty tensor cannot request propagation without recordings.
:type receiver_locations: `torch.Tensor`
:param accuracy: **Default `8`; keyword-only.** Spatial finite-difference order, one of `2,4,6,8`, not an error tolerance. Affects stencil extent, computational cost, and the minimum boundary buffer. Booleans and other orders are rejected.
:type accuracy: `int`
:param pml_freq: **Default `25.0`; keyword-only.** Positive, finite frequency used to configure the PML, typically in Hz; also used for spatial-sampling diagnostics. It does not generate a waveform or automatically measure the source frequency. Choose it for the actual experiment.
:type pml_freq: `float | int`
:param pml_width: **Default `20`; keyword-only.** Positive integer thickness in grid cells. A single value applies to every face; a sequence of four equal integers is also accepted, ordered as the beginning/end of the first axis, then the beginning/end of the second axis. Zero width, asymmetric face widths, and requesting a free surface with a zero-width face are unsupported.
:type pml_width: `int | list[int] | tuple[int, ...]`
:param boundary_buffer: **Default `5`; keyword-only.** Nonnegative buffer in spatial grid cells, not a number of time steps or a checkpoint interval. Boundary mode requires at least `accuracy // 2 + 1`, which is 5 at eighth order; full mode allows 0. Total padding on each side is `pml_width + boundary_buffer + accuracy // 2`.
:type boundary_buffer: `int`
:param memory: **Default `'boundary'`; keyword-only.** History strategy for model backpropagation. `"boundary"` stores boundary strips and reconstructs the wavefield; `"full"` stores the complete history. Full mode can exhaust GPU memory; there is no automatic fallback or `"checkpoint"` option. Forward propagation without a requested model gradient does not retain model-backpropagation history.
:type memory: `str`
:param max_vel: **Default `None`; keyword-only.** CFL/PML velocity envelope, typically in m/s. With `None`, each call replans from the current `max(v)`. An explicit value must be positive, finite, and cover the current maximum velocity; it is not a clipping limit. Planning and PML configuration are not differentiated. This primary-velocity constraint does not guarantee spatial stability for arbitrary impedance contrasts.
:type max_vel: `float | int | None`
:param freq_taper_frac: **Default `0.0`; keyword-only.** Finite fraction in `[0,1]` controlling the cosine taper of high-frequency FFT bins during temporal resampling; the bin count is obtained by truncating the proportional count to an integer. During resampling, the last positive rFFT bin is suppressed even when this fraction is 0, including for odd lengths; preservation of every frequency is not guaranteed. This setting does not change the signal when the internal resampling ratio is 1.
:type freq_taper_frac: `float | int`
:param time_pad_frac: **Default `0.0`; keyword-only.** Finite fraction in `[0,1]`. Appends `int(time_pad_frac*T)` user samples of zeros before resampling and removes them afterwards. Does not increase the public recording length or the actual propagation duration. It does not change the signal when the internal resampling ratio is 1.
:type time_pad_frac: `float | int`
:param time_taper: **Default `False`; keyword-only.** Whether to apply a nonperiodic Hann window during temporal resampling: after upsampling and before downsampling. Changes both the signal and its backpropagation transpose. Accepts only Booleans, not the integers 0/1. It does not change the signal when the internal resampling ratio is 1.
:type time_taper: `bool`
:param illumination: **Default `None`; keyword-only.** Compatibility keyword only; must be `None`. VRZ does not support illumination. Explicitly passing a collector or any other value raises `NotImplementedError`; do not reuse scalar’s illumination setup.
:type illumination: `None`
:returns: **`(receiver_amplitudes,)`**, a one-element tuple. Its only recording tensor has shape `[B,R,T]`, is contiguous CUDA float32 on the same device as the model, and has nominal times `0, dt, ..., (T-1)*dt`. These are pressure-like recordings, accessible with `[0]` or `[-1]`; no final wavefield or boundary state is included.
:rtype: `tuple[torch.Tensor]`
```

(vrz-details)=
### Parameterization, sources, and gradients

With `impedance=Z`, velocity and impedance are independent inputs. With `density=rho`, impedance is obtained internally from `v*rho`. The two gradient interpretations differ and cannot be interchanged without applying the chain rule. Use `requires_grad=True` for trainable inputs; other medium values still participate in forward propagation.

Sources use the same normalized-forcing convention as scalar: the internal increment is `-v_source² * internal_dt² * U(f)`, rather than a direct pressure increment. Do not apply additional time-step or velocity scaling. Recordings are aligned to user times before downsampling by prepending zero and removing the final step; FFT resampling may produce endpoint ringing.

Positivity, finiteness, coefficient representability, and CFL checks do not guarantee stability for arbitrarily strong impedance contrasts. Near-zero and strong-contrast models may be severely ill-conditioned.

Model gradients are conditional on a fixed extended model, CFL configuration, and PML. When replicate padding at the physical edges is regenerated, the current returned gradient does not include the transpose accumulation of the complete extension chain. It must not be treated as the total derivative of the entire boundary-rebuilding process; switching to full mode does not remove this limitation.

(vrz-examples)=
### Examples

The following is a call fragment assuming valid models, acquisition inputs, and the native library have already been prepared; it is not a standalone program. Every omitted optional parameter uses the default shown above.

```python
import starwave

receiver_amplitudes, = starwave.vrz(
    v, grid_spacing=10.0, dt=0.001,
    density=rho,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
)
```

For the impedance parameterization, replace `density=rho` with `impedance=Z`; do not keep both. See {ref}`VRZ modeling <wave-vrz>` for more background. A complete standalone GPU example remains to be added.

(vrz-notes)=
### Notes

- Supports CUDA FP32 on the default stream only; each forward permits one first-order backward pass. Source gradients, higher-order derivatives, AMP, CUDA graphs, custom streams, and public initial/final states are unsupported.
- CFL time substeps cannot compensate for inadequate spatial sampling. Spatial-resolution warnings are diagnostics, not certificates of accuracy or stability.
- For DataParallel, place the complete model in a Module and split acquisition inputs only along the shot dimension; see [Multi-GPU introduction](inversion/dataparallel.md).
- This page’s interface contract has been checked. Target-GPU numerical, performance, and FWI acceptance tests have not been completed.

(vti)=
## VTI Function

```{py:function} starwave.vti(vp: torch.Tensor, epsilon: torch.Tensor, delta: torch.Tensor, rho: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, source_fields: str | list[str] | tuple[str, ...]=('sH', 'sV'), receiver_fields: str | list[str] | tuple[str, ...]=('vz',), accuracy: int=4, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float | int=0.0, time_pad_frac: float | int=0.0, time_taper: bool=False) -> tuple[torch.Tensor, ...]

Propagates a Duveneck-type first-order acoustic VTI system, selecting 2D or 3D from `vp.ndim`. Outputs follow the selected receiver-component order. Each of the four models can independently request a first-order gradient in one backward pass. Fixed models still participate in all physical calculations; this interface is not full elastic VTI.

:param vp: **Required; positional or keyword.** Vertical P-wave velocity in m/s. CUDA float32, positive and finite, with shape `[nx,nz]` or `[nx,ny,nz]` and at least 2 cells per dimension. The last axis is always vertical. All four models must have matching shape, dtype, and device. Physical axis order is neither inferred nor converted automatically.
:type vp: `torch.Tensor`
:param epsilon: **Required; positional or keyword.** Dimensionless Thomsen epsilon, CUDA float32, with the same shape/device as `vp`. Must be finite and strictly greater than -0.5, and the coefficients and derivatives must be numerically representable. `epsilon<delta` is allowed, without implicit clipping; growing modes are a known limitation.
:type epsilon: `torch.Tensor`
:param delta: **Required; positional or keyword.** Dimensionless Thomsen delta, CUDA float32, with the same shape/device as `vp`. Must be finite and strictly greater than -0.5, and the coefficients and derivatives must be numerically representable. The input is not constrained to `delta<=epsilon`; the model is not modified implicitly.
:type delta: `torch.Tensor`
:param rho: **Required; positional or keyword.** Density in kg/m³. CUDA float32, with the same shape/device as `vp`, positive and finite; reciprocals, stiffnesses, and derivatives must be representable. There is no default density or implicit unit conversion. If the original data are confirmed to be in g/cm³, explicitly multiply them by 1000 before calling.
:type rho: `torch.Tensor`
:param grid_spacing: **Required; positional or keyword.** Positive, finite grid spacing in m. Accepts a single value or per-axis spacing `[dx,dz]` / `[dx,dy,dz]`; unequal spacing is allowed. Axis order must match the model. Booleans, zero/negative values, and mismatched sequence lengths are rejected.
:type grid_spacing: `float | int | list[float] | tuple[float, ...]`
:param dt: **Required; positional or keyword.** Positive, finite sampling interval of the user input/output, typically in s. Internally, the CFL condition may require a smaller time step; the returned sampling interval remains this value. Tensor and Boolean inputs are not accepted, and this parameter is not differentiated.
:type dt: `float | int`
:param source_amplitudes: **Required; keyword-only.** Fixed float32/float64 tensor `[B,S,T]`, in **Pa/s**, representing a stress rate held over each internal time step. All three dimensions must be nonempty; values must be finite and convertible to float32. `requires_grad=True` is forbidden, and integer/Boolean/complex sources are not accepted. Internally, `internal_dt*U(rate)` is applied; no additional external multiplication by dt is needed.
:type source_amplitudes: `torch.Tensor`
:param source_locations: **Required; keyword-only.** Fixed integer tensor `[B,S,D]`; `[B,D]` is accepted for a single source per shot. `D=vp.ndim`; `torch.long` is recommended. Provides physical-grid indices in the model’s `[x,z]` / `[x,y,z]` axis order, without PML offsets, and must be in bounds. Coincident sources add their increments.
:type source_locations: `torch.Tensor`
:param receiver_locations: **Required; keyword-only.** Fixed integer tensor `[B,R,D]`, with `R>0` and the same shot count as the sources; `torch.long` is recommended. Uses model axis order. These integers index the staggered grid of the selected component; no spatial interpolation to cell centers is performed. Duplicate positions are allowed; all positions must be in bounds.
:type receiver_locations: `torch.Tensor`
:param source_fields: **Default `('sH', 'sV')`; keyword-only.** Normal-stress components that receive the same source rate; only `sH` and `sV` are selectable. Accepts one name or a nonempty sequence of distinct names. By default, both receive the source, without dividing its amplitude by 2. Velocity-source components and per-source component mappings are unsupported.
:type source_fields: `str | list[str] | tuple[str, ...]`
:param receiver_fields: **Default `('vz',)`; keyword-only.** Receiver components and their return order; nonempty and without duplicates. Choices in 2D are `vx,vz,sH,sV`; 3D additionally supports `vy`. Velocities are in m/s, located half a grid cell in the positive direction of the corresponding axis; stresses are in Pa at cell centers. There is no `pressure` alias, and `vy` cannot be selected in 2D.
:type receiver_fields: `str | list[str] | tuple[str, ...]`
:param accuracy: **Default `4`; keyword-only.** Spatial finite-difference order on the staggered grid, one of `2,4,6,8`. Affects the stencil, CFL condition, and minimum buffer; it is not an error tolerance. Booleans and other orders are rejected.
:type accuracy: `int`
:param pml_freq: **Default `25.0`; keyword-only.** Positive, finite frequency used to configure the PML, typically in Hz; also used for spatial-sampling diagnostics. It does not generate a waveform or automatically measure the source frequency. Choose it for the actual experiment.
:type pml_freq: `float | int`
:param pml_width: **Default `20`; keyword-only.** Positive integer thickness in grid cells. A single value applies to every face; alternatively, use a sequence of `2*D` equal integers, ordered in beginning/end pairs for each axis. Asymmetric widths, zero-width faces, and free surfaces are unsupported.
:type pml_width: `int | list[int] | tuple[int, ...]`
:param boundary_buffer: **Default `5`; keyword-only.** Nonnegative buffer in spatial grid cells, not a number of time steps or a checkpoint interval. Boundary mode requires at least `accuracy // 2 + 1`, which is 5 at eighth order; full mode allows 0. Total padding on each side is `pml_width + boundary_buffer + accuracy // 2`.
:type boundary_buffer: `int`
:param memory: **Default `'boundary'`; keyword-only.** History strategy for model backpropagation. `"boundary"` stores boundary strips and reconstructs the wavefield; `"full"` stores the complete history. Full mode can exhaust GPU memory; there is no automatic fallback or `"checkpoint"` option. Forward propagation without a requested model gradient does not retain model-backpropagation history.
:type memory: `str`
:param max_vel: **Default `None`; keyword-only.** Anisotropic CFL/PML velocity envelope in m/s. With `None`, it is computed dynamically. An explicit value must be positive, finite, and cover `max(vp*sqrt(max(1,1+2*epsilon,sqrt(1+2*delta))))`. Covering only `max(vp)` is insufficient. Density contrast is still recomputed. Planning also depends on dimensionality, spacing, and finite-difference coefficients, and is not differentiated.
:type max_vel: `float | int | None`
:param freq_taper_frac: **Default `0.0`; keyword-only.** Finite fraction in `[0,1]` controlling the cosine taper of high-frequency FFT bins during temporal resampling; the bin count is obtained by truncating the proportional count to an integer. During resampling, the last positive rFFT bin is suppressed even when this fraction is 0, including for odd lengths; preservation of every frequency is not guaranteed. This setting does not change the signal when the internal resampling ratio is 1.
:type freq_taper_frac: `float | int`
:param time_pad_frac: **Default `0.0`; keyword-only.** Finite fraction in `[0,1]`. Appends `int(time_pad_frac*T)` user samples of zeros before resampling and removes them afterwards. Does not increase the public recording length or the actual propagation duration. It does not change the signal when the internal resampling ratio is 1.
:type time_pad_frac: `float | int`
:param time_taper: **Default `False`; keyword-only.** Whether to apply a nonperiodic Hann window during temporal resampling: after upsampling and before downsampling. Changes both the signal and its backpropagation transpose. Accepts only Booleans, not the integers 0/1. It does not change the signal when the internal resampling ratio is 1.
:type time_taper: `bool`
:returns: **`(records_0, ..., records_n)`**, with one contiguous CUDA float32 tensor `[B,R,T]` per `receiver_fields` entry, in exactly the requested component order. The default returns only `(vz_records,)`, the vertical particle velocity in m/s, not pressure; `sH/sV` are stresses in Pa. Tensors are on the same device as the model, at nominal user times `0, dt, ..., (T-1)*dt`, without final states.
:rtype: `tuple[torch.Tensor, ...]`
```

(vti-details)=
### Components, timing, and gradients

A positive stress rate adds positive stress to every selected stress component; internally, `internal_dt*U(rate)` is applied. There is no scalar-style `-vp² dt²` factor or model-dependent source scaling. Legacy per-step stress increments cannot be used directly as public rate inputs.

Stress recordings are time-aligned before downsampling by prepending zero and removing the final entry. Velocity recordings instead average the preceding and following half-time-step samples onto integer times, taking the initial preceding half step as zero. Both outputs are labeled with nominal user times, but their physical quantities and spatial locations differ. Resampling may produce endpoint ringing.

Each of `vp`, `epsilon`, `delta`, and `rho` requests a gradient only when its own `requires_grad=True`; fixed fields still participate in forward and adjoint propagation. No rule automatically freezes fields based on their values. `nn.Parameter` enables gradients by default; fixed fields can be registered as buffers.

`epsilon<delta` is not automatically repaired. A smaller dt does not guarantee removal of negative-stiffness growth or arbitrary PML instability. Covering the velocity envelope is only a necessary input condition.

Model gradients are conditional on a fixed extended model, CFL configuration, and PML. When replicate padding at the physical edges is regenerated, the current returned gradient does not include the transpose accumulation of the complete extension chain. It must not be treated as the total derivative of the entire boundary-rebuilding process; switching to full mode does not remove this limitation.

(vti-examples)=
### Examples

The following is a call fragment assuming valid models, acquisition inputs, and the native library have already been prepared; it is not a standalone program. Every omitted optional parameter uses the default shown above.

```python
import starwave

stress_h, velocity_z = starwave.vti(
    vp, epsilon, delta, rho,
    grid_spacing=[10.0, 10.0], dt=0.001,
    source_amplitudes=stress_rate,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    receiver_fields=("sH", "vz"),
)
```

This example uses 2D inputs. A 3D call needs all four models shaped `[nx,ny,nz]`, 3D coordinates, and matching spacing. Omitting `receiver_fields` returns only `vz`, so use single-variable tuple unpacking. VTI has no `illumination` parameter. A complete standalone GPU example remains to be added; see {ref}`VTI modeling <wave-vti>`.

(vti-notes)=
### Notes

- Supports CUDA FP32 on the default stream only; each forward permits one first-order backward pass. Source gradients, higher-order derivatives, AMP, CUDA graphs, custom streams, and public initial/final states are unsupported.
- CFL time substeps cannot compensate for inadequate spatial sampling. Spatial-resolution warnings are diagnostics, not certificates of accuracy or stability.
- For DataParallel, place the complete model in a Module and split acquisition inputs only along the shot dimension; see [Multi-GPU introduction](inversion/dataparallel.md).
- This page’s interface contract has been checked. Target-GPU numerical, performance, and FWI acceptance tests have not been completed.

(elastic)=
## Elastic Function

```{py:function} starwave.elastic(lamb: torch.Tensor, mu: torch.Tensor, buoyancy: torch.Tensor, grid_spacing: Union[float, Sequence[float]], dt: float, source_amplitudes_z: Optional[torch.Tensor]=None, source_amplitudes_y: Optional[torch.Tensor]=None, source_amplitudes_x: Optional[torch.Tensor]=None, source_amplitudes_p: Optional[torch.Tensor]=None, source_locations_z: Optional[torch.Tensor]=None, source_locations_y: Optional[torch.Tensor]=None, source_locations_x: Optional[torch.Tensor]=None, source_locations_p: Optional[torch.Tensor]=None, receiver_locations_z: Optional[torch.Tensor]=None, receiver_locations_y: Optional[torch.Tensor]=None, receiver_locations_x: Optional[torch.Tensor]=None, receiver_locations_p: Optional[torch.Tensor]=None, accuracy: int=4, pml_width: Union[int, Sequence[int]]=20, pml_freq: Optional[float]=None, max_vel: Optional[float]=None, survey_pad: Optional[Union[int, Sequence[Optional[int]]]]=None, vz_0: Optional[torch.Tensor]=None, vy_0: Optional[torch.Tensor]=None, vx_0: Optional[torch.Tensor]=None, sigmazz_0: Optional[torch.Tensor]=None, sigmayz_0: Optional[torch.Tensor]=None, sigmaxz_0: Optional[torch.Tensor]=None, sigmayy_0: Optional[torch.Tensor]=None, sigmaxy_0: Optional[torch.Tensor]=None, sigmaxx_0: Optional[torch.Tensor]=None, m_vzz_0: Optional[torch.Tensor]=None, m_vzy_0: Optional[torch.Tensor]=None, m_vzx_0: Optional[torch.Tensor]=None, m_vyz_0: Optional[torch.Tensor]=None, m_vxz_0: Optional[torch.Tensor]=None, m_vyy_0: Optional[torch.Tensor]=None, m_vyx_0: Optional[torch.Tensor]=None, m_vxy_0: Optional[torch.Tensor]=None, m_vxx_0: Optional[torch.Tensor]=None, m_sigmazzz_0: Optional[torch.Tensor]=None, m_sigmayzy_0: Optional[torch.Tensor]=None, m_sigmaxzx_0: Optional[torch.Tensor]=None, m_sigmayzz_0: Optional[torch.Tensor]=None, m_sigmaxzz_0: Optional[torch.Tensor]=None, m_sigmayyy_0: Optional[torch.Tensor]=None, m_sigmaxyy_0: Optional[torch.Tensor]=None, m_sigmaxyx_0: Optional[torch.Tensor]=None, m_sigmaxxx_0: Optional[torch.Tensor]=None, origin: Optional[Sequence[int]]=None, nt: Optional[int]=None, model_gradient_sampling_interval: int=1, freq_taper_frac: float=0.0, time_pad_frac: float=0.0, time_taper: bool=False, forward_callback: Optional[common.Callback]=None, callback_frequency: int=1, python_backend: Union[Literal['eager', 'jit', 'compile'], bool]=False, storage_mode: Literal['device', 'cpu', 'disk', 'none']='device', storage_path: str='.', storage_compression: bool=False, *, memory: Literal['full', 'boundary']='full') -> Tuple[torch.Tensor, ...]

2D/3D isotropic elastic propagation with the Deepwave 0.0.27-derived backend. Inputs are raw Lamé parameters and buoyancy, with multi-shot first-order differentiation of materials, sources and initial states. Every argument except the final memory is positional-or-keyword; only the first five are required.

:param lamb: **Required.** First Lamé parameter, typically Pa. Spatial shape `[Ny,Nx]` (2D) or `[Nz,Ny,Nx]` (3D), optionally with a leading model batch of 1 or B. CPU/CUDA float32 or float64, with finite values. Spatial shape, dtype and device must match mu and buoyancy; this input is never interpreted as vp.
:type lamb: `torch.Tensor`
:param mu: **Required.** Second Lamé parameter (shear modulus), typically Pa; the model-batch, axis, dtype and device rules for lamb apply. Trainable inputs retain their autograd chain.
:type mu: `torch.Tensor`
:param buoyancy: **Required.** Buoyancy, or inverse density, typically m³/kg; the model-shape rules for lamb apply. This is not density rho. Convert vp/vs/rho explicitly with the helper below.
:type buoyancy: `torch.Tensor`
:param grid_spacing: **Required.** Grid spacing, typically m: a positive scalar or a positive length-D sequence in model-axis order. Unequal spacings are supported.
:type grid_spacing: `Union[float, Sequence[float]]`
:param dt: **Required.** User source/receiver sampling interval, typically s. CFL preprocessing may subdivide it internally; velocity and stress use staggered clocks, as described below.
:type dt: `float`
:param source_amplitudes_z: **Default `None`.** 3D only. Tensor `[B,S,T]` of force density along z, typically N/m³. Pair it with source_locations_z; dtype/device must match the materials. Source gradients are supported. Supplied components need compatible shot counts and time lengths.
:type source_amplitudes_z: `Optional[torch.Tensor]`
:param source_amplitudes_y: **Default `None`.** 2D/3D. Tensor `[B,S,T]` of force density along y, typically N/m³. Pair it with source_locations_y; dtype/device must match the materials. Source gradients are supported. Supplied components need compatible shot counts and time lengths.
:type source_amplitudes_y: `Optional[torch.Tensor]`
:param source_amplitudes_x: **Default `None`.** 2D/3D. Tensor `[B,S,T]` of force density along x, typically N/m³. Pair it with source_locations_x; dtype/device must match the materials. Source gradients are supported. Supplied components need compatible shot counts and time lengths.
:type source_amplitudes_x: `Optional[torch.Tensor]`
:param source_amplitudes_p: **Default `None`.** 2D/3D. Tensor `[B,S,T]` of pressure rate, typically Pa/s, positive for compression. Pair it with source_locations_p; dtype/device must match the materials. Source gradients are supported. Supplied components need compatible shot counts and time lengths.
:type source_amplitudes_p: `Optional[torch.Tensor]`
:param source_locations_z: **Default `None`.** 3D only. Integer grid coordinates `[B,S,D]`, preferably torch.long, paired with source_amplitudes_z, in input model-axis order without PML offsets. Source locations for a component must be unique within a shot; `starwave.common.IGNORE_LOCATION` masks locations.
:type source_locations_z: `Optional[torch.Tensor]`
:param source_locations_y: **Default `None`.** 2D/3D. Integer grid coordinates `[B,S,D]`, preferably torch.long, paired with source_amplitudes_y, in input model-axis order without PML offsets. Source locations for a component must be unique within a shot; `starwave.common.IGNORE_LOCATION` masks locations.
:type source_locations_y: `Optional[torch.Tensor]`
:param source_locations_x: **Default `None`.** 2D/3D. Integer grid coordinates `[B,S,D]`, preferably torch.long, paired with source_amplitudes_x, in input model-axis order without PML offsets. Source locations for a component must be unique within a shot; `starwave.common.IGNORE_LOCATION` masks locations.
:type source_locations_x: `Optional[torch.Tensor]`
:param source_locations_p: **Default `None`.** 2D/3D. Integer grid coordinates `[B,S,D]`, preferably torch.long, paired with source_amplitudes_p, in input model-axis order without PML offsets. Source locations for a component must be unique within a shot; `starwave.common.IGNORE_LOCATION` masks locations.
:type source_locations_p: `Optional[torch.Tensor]`
:param receiver_locations_z: **Default `None`.** 3D only. Integer grid coordinates `[B,R,D]`, preferably torch.long, in model-axis order without PML offsets. Requests z-velocity records. If omitted, its return slot remains an empty tensor. Active receiver locations must be unique within a shot/component; `starwave.common.IGNORE_LOCATION` masks locations.
:type receiver_locations_z: `Optional[torch.Tensor]`
:param receiver_locations_y: **Default `None`.** 2D/3D. Integer grid coordinates `[B,R,D]`, preferably torch.long, in model-axis order without PML offsets. Requests y-velocity records. If omitted, its return slot remains an empty tensor. Active receiver locations must be unique within a shot/component; `starwave.common.IGNORE_LOCATION` masks locations.
:type receiver_locations_y: `Optional[torch.Tensor]`
:param receiver_locations_x: **Default `None`.** 2D/3D. Integer grid coordinates `[B,R,D]`, preferably torch.long, in model-axis order without PML offsets. Requests x-velocity records. If omitted, its return slot remains an empty tensor. Active receiver locations must be unique within a shot/component; `starwave.common.IGNORE_LOCATION` masks locations.
:type receiver_locations_x: `Optional[torch.Tensor]`
:param receiver_locations_p: **Default `None`.** 2D/3D. Integer grid coordinates `[B,R,D]`, preferably torch.long, in model-axis order without PML offsets. Requests pressure records. If omitted, its return slot remains an empty tensor. Active receiver locations must be unique within a shot/component; `starwave.common.IGNORE_LOCATION` masks locations.
:type receiver_locations_p: `Optional[torch.Tensor]`
:param accuracy: **Default `4`.** Spatial finite-difference order: `2,4,6,8`, for 2D/3D in both full and boundary modes.
:type accuracy: `int`
:param pml_width: **Default `20`.** PML width in cells: a nonnegative integer or length-2D sequence of low/high widths for each spatial axis. A zero face width removes that absorbing layer; it is not a general stress-free-surface switch.
:type pml_width: `Union[int, Sequence[int]]`
:param pml_freq: **Default `None`.** PML frequency, typically Hz. None selects 25 Hz with a warning; specify the dominant source frequency explicitly.
:type pml_freq: `Optional[float]`
:param max_vel: **Default `None`.** Maximum speed for CFL/PML, typically m/s. None uses the maximum reconstructed P/S speed. An explicit value should bound model speeds; it is neither a clipping limit nor a trainable parameter.
:type max_vel: `Optional[float]`
:param survey_pad: **Default `None`.** Extracts one shared subdomain around the sources/receivers of all shots in this call. Without initial fields, None keeps the whole model; an integer pads every face; a length-2D sequence gives low/high margins per axis, with None extending to that model edge. Margins are grid cells, before PML is added. Cropping can change results; retain the region needed by the waves. With initial fields, None can instead derive the subdomain from state shape and origin. survey_pad and origin cannot both be non-None.
:type survey_pad: `Optional[Union[int, Sequence[Optional[int]]]]`
:param vz_0: **Default `None`.** 3D only; initial velocity vz, typically m/s, at internal time −h/2 (h is the internal timestep). Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type vz_0: `Optional[torch.Tensor]`
:param vy_0: **Default `None`.** 2D/3D; initial velocity vy, typically m/s, at internal time −h/2 (h is the internal timestep). Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type vy_0: `Optional[torch.Tensor]`
:param vx_0: **Default `None`.** 2D/3D; initial velocity vx, typically m/s, at internal time −h/2 (h is the internal timestep). Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type vx_0: `Optional[torch.Tensor]`
:param sigmazz_0: **Default `None`.** 3D only; initial stress sigmazz, typically Pa, at internal time 0. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type sigmazz_0: `Optional[torch.Tensor]`
:param sigmayz_0: **Default `None`.** 3D only; initial stress sigmayz, typically Pa, at internal time 0. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type sigmayz_0: `Optional[torch.Tensor]`
:param sigmaxz_0: **Default `None`.** 3D only; initial stress sigmaxz, typically Pa, at internal time 0. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type sigmaxz_0: `Optional[torch.Tensor]`
:param sigmayy_0: **Default `None`.** 2D/3D; initial stress sigmayy, typically Pa, at internal time 0. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type sigmayy_0: `Optional[torch.Tensor]`
:param sigmaxy_0: **Default `None`.** 2D/3D; initial stress sigmaxy, typically Pa, at internal time 0. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type sigmaxy_0: `Optional[torch.Tensor]`
:param sigmaxx_0: **Default `None`.** 2D/3D; initial stress sigmaxx, typically Pa, at internal time 0. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type sigmaxx_0: `Optional[torch.Tensor]`
:param m_vzz_0: **Default `None`.** 3D only; initial PML memory variable m_vzz. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type m_vzz_0: `Optional[torch.Tensor]`
:param m_vzy_0: **Default `None`.** 3D only; initial PML memory variable m_vzy. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type m_vzy_0: `Optional[torch.Tensor]`
:param m_vzx_0: **Default `None`.** 3D only; initial PML memory variable m_vzx. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type m_vzx_0: `Optional[torch.Tensor]`
:param m_vyz_0: **Default `None`.** 3D only; initial PML memory variable m_vyz. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type m_vyz_0: `Optional[torch.Tensor]`
:param m_vxz_0: **Default `None`.** 3D only; initial PML memory variable m_vxz. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type m_vxz_0: `Optional[torch.Tensor]`
:param m_vyy_0: **Default `None`.** 2D/3D; initial PML memory variable m_vyy. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type m_vyy_0: `Optional[torch.Tensor]`
:param m_vyx_0: **Default `None`.** 2D/3D; initial PML memory variable m_vyx. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type m_vyx_0: `Optional[torch.Tensor]`
:param m_vxy_0: **Default `None`.** 2D/3D; initial PML memory variable m_vxy. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type m_vxy_0: `Optional[torch.Tensor]`
:param m_vxx_0: **Default `None`.** 2D/3D; initial PML memory variable m_vxx. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type m_vxx_0: `Optional[torch.Tensor]`
:param m_sigmazzz_0: **Default `None`.** 3D only; initial PML memory variable m_sigmazzz. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type m_sigmazzz_0: `Optional[torch.Tensor]`
:param m_sigmayzy_0: **Default `None`.** 3D only; initial PML memory variable m_sigmayzy. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type m_sigmayzy_0: `Optional[torch.Tensor]`
:param m_sigmaxzx_0: **Default `None`.** 3D only; initial PML memory variable m_sigmaxzx. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type m_sigmaxzx_0: `Optional[torch.Tensor]`
:param m_sigmayzz_0: **Default `None`.** 3D only; initial PML memory variable m_sigmayzz. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type m_sigmayzz_0: `Optional[torch.Tensor]`
:param m_sigmaxzz_0: **Default `None`.** 3D only; initial PML memory variable m_sigmaxzz. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type m_sigmaxzz_0: `Optional[torch.Tensor]`
:param m_sigmayyy_0: **Default `None`.** 2D/3D; initial PML memory variable m_sigmayyy. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type m_sigmayyy_0: `Optional[torch.Tensor]`
:param m_sigmaxyy_0: **Default `None`.** 2D/3D; initial PML memory variable m_sigmaxyy. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type m_sigmaxyy_0: `Optional[torch.Tensor]`
:param m_sigmaxyx_0: **Default `None`.** 2D/3D; initial PML memory variable m_sigmaxyx. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type m_sigmaxyx_0: `Optional[torch.Tensor]`
:param m_sigmaxxx_0: **Default `None`.** 2D/3D; initial PML memory variable m_sigmaxxx. Shape `[B,*propagation_domain]`, including PML but excluding finite-difference halos. Match material dtype/device; gradients are supported. None initializes zeros; staggered-grid edge masks still apply.
:type m_sigmaxxx_0: `Optional[torch.Tensor]`
:param origin: **Default `None`.** Length-D integer model coordinate for the origin of supplied initial fields, in model-axis order. Together with state shape and PML, it determines the continuation region; otherwise inferred by the upstream rules.
:type origin: `Optional[Sequence[int]]`
:param nt: **Default `None`.** Number of user time samples, required when no source amplitudes are supplied. With source amplitudes, duration is normally inferred from their final dimension; an explicit nt must equal that length. Source-free calls still need coordinates or initial fields sufficient to infer dimensions/shots.
:type nt: `Optional[int]`
:param model_gradient_sampling_interval: **Default `1`.** Material-gradient time-sampling interval, an integer ≥1. The internal accumulation stride is the CFL ratio q times this value. The default 1 is the simplest choice; larger values use sampled gradients rather than promising every-step equivalence. Current native paths process complete sampling groups; see storage and sampling below.
:type model_gradient_sampling_interval: `int`
:param freq_taper_frac: **Default `0.0`.** Fraction of the high-frequency end to cosine-taper during CFL-driven FFT resampling of sources and receivers; does not change user output length.
:type freq_taper_frac: `float`
:param time_pad_frac: **Default `0.0`.** Fractional zero-padding added before FFT resampling and removed afterwards for sources and receivers; does not extend the public record duration.
:type time_pad_frac: `float`
:param time_taper: **Default `False`.** Whether to apply a temporal Hann window to sources and receivers during resampling.
:type time_taper: `bool`
:param forward_callback: **Default `None`.** Observer receiving `starwave.common.CallbackState`. Native callbacks run on backend outer-loop groups; state.dt is internal h and state.step is the outer-loop index. Boundary snapshots are read-only.
:type forward_callback: `Optional[common.Callback]`
:param callback_frequency: **Default `1`.** Positive number of backend outer-loop groups between callbacks. A native group spans q × model_gradient_sampling_interval internal steps; it is not a fixed user-record sampling interval.
:type callback_frequency: `int`
:param python_backend: **Default `False`.** False selects C/CUDA; `"eager"`, `"jit"` and `"compile"` explicitly select PyTorch paths. True selects compile when a verified build declares OpenMP, otherwise jit; availability depends on PyTorch/toolchain. Python paths require full mode, device storage and no compression.
:type python_backend: `Union[Literal['eager', 'jit', 'compile'], bool]`
:param storage_mode: **Default `'device'`.** Intermediate storage: `"device"`, `"cpu"`, `"disk"`, or `"none"`. Offload/disk/none are native full options; none does not provide complete material gradients, as explained below.
:type storage_mode: `Literal['device', 'cpu', 'disk', 'none']`
:param storage_path: **Default `'.'`.** Directory for disk-mode temporary histories. Keeping the computation graph can keep these files alive; do not delete files still needed for backward.
:type storage_path: `str`
:param storage_compression: **Default `False`.** Lossy intermediate compression for native full mode; may change material gradients. Boundary and Python backends reject True.
:type storage_compression: `bool`
:param memory: **Default `'full'`.** StarWave keyword-only extension. `"full"` stores volume histories; `"boundary"` uses CUDA boundary storage and reconstruction. Both support 2D/3D and orders 2/4/6/8. Unsupported combinations fail rather than silently falling back to full.
:type memory: `Literal['full', 'boundary']`
:returns: Complete final wavefield/PML states and receiver tuple: 16 tensors in 2D, 31 in 3D. Requested records have shape `[B,R,T]`; absent receiver components remain empty tensors. Exact order is listed below.
:rtype: `Tuple[torch.Tensor, ...]`
```

(elastic-axes)=
### Model axes and staggered grid

The 2D API axes are `[y,x]`, and 3D axes are `[z,y,x]`. These names follow the input tensor's spatial axes; they do not determine the physical vertical direction independently of your array convention. A 2D `[depth,horizontal]` array can be passed directly: y then means depth, and a vertical force uses source_amplitudes_y. After an explicit transpose to `[horizontal,depth]`, depth corresponds to x; transform source/receiver components, coordinates, grid_spacing, PML faces and origin consistently. Neither material conversion transposes arrays.

Materials may be shared or per shot; sources and records have shapes `[B,S,T]` and `[B,R,T]`. The last coordinate dimension follows the same model axes. Components occupy different half-grid positions on the staggered grid, so they are not interchangeable colocated quantities. Velocity sources/receivers cannot use the last model index along their own component axis. See the [Deepwave elastic staggered grid](https://ausargeo.com/deepwave/elastic.html).

Sources y/x (and z in 3D) are force densities; source p is a pressure rate. Receivers y/x/z record velocity; receiver p records minus the mean normal stress. Nominal pressure samples are at `t*dt`, and velocity samples at `(t-0.5)*dt`. CFL subdivision uses the internal half-step and FFT resampling without a separate half-step retiming operation. Do not reuse scalar's forcing or recording clock as the elastic contract.

(elastic-returns)=
### Complete return order

2D returns 16 tensors, with fixed p/y/x receiver slots at the end:

```text
(vy, vx, sigmayy, sigmaxy, sigmaxx,
 m_vyy, m_vyx, m_vxy, m_vxx,
 m_sigmayyy, m_sigmaxyy, m_sigmaxyx, m_sigmaxxx,
 receiver_amplitudes_p, receiver_amplitudes_y, receiver_amplitudes_x)
```

3D returns 31 tensors, with fixed p/z/y/x receiver slots at the end:

```text
(vz, vy, vx, sigmazz, sigmayz, sigmaxz, sigmayy, sigmaxy, sigmaxx,
 m_vzz, m_vzy, m_vzx, m_vyz, m_vxz, m_vyy, m_vyx, m_vxy, m_vxx,
 m_sigmazzz, m_sigmayzy, m_sigmaxzx, m_sigmayzz, m_sigmaxzz,
 m_sigmayyy, m_sigmaxyy, m_sigmaxyx, m_sigmaxxx,
 receiver_amplitudes_p, receiver_amplitudes_z,
 receiver_amplitudes_y, receiver_amplitudes_x)
```

Final states have shape `[B,*propagation_domain_including_PML]`, without finite-difference halos, for the actual domain selected by survey_pad, origin and initial fields. They can be reused through the corresponding `_0` arguments for continuation. Each requested receiver tensor is `[B,R,T]`; absent components retain shape-`(0,)` empty tensor slots. Consequently `outputs[-1]` always means the x component, not whichever record you requested. Tensor dtype/device match the materials.

(elastic-memory)=
### Storage and gradient sampling

For the reconstruction order, material gradients and storage formulas, see {ref}`Wavefield Reconstruction <reconstruction>`.

- `memory="full"` is the default, supporting 2D/3D CPU/CUDA float32/float64. Native `storage_mode="device"` stores histories on the current device; `"cpu"` uses host storage for CUDA and is normalized to device storage for CPU models; `"disk"` uses storage_path. `storage_compression=True` is lossy.
- `storage_mode="none"` disables propagation-history contributions to material gradients and cannot provide full material gradients for inversion. Material-dependent force-source scaling can still contribute partial gradients. Trainable materials trigger a warning; source/initial-state gradients are a separate path.
- `memory="boundary"` is a StarWave extension: CUDA only, with `python_backend=False`, `storage_mode="device"` and `storage_compression=False`. Every cropped physical extent must exceed accuracy; active staggered buoyancy must be nonzero when its gradient is requested. CPU, offload and compression are unsupported, with no automatic full fallback. Memory/speed benefits depend on domain shape, PML and sampling.
- Keep `model_gradient_sampling_interval=1` for the simplest behavior; when CFL ratio q>1, the material-gradient stride is still q, rather than every internal step. For larger integers, current native full/boundary paths execute complete sampling groups only: an incomplete tail is not propagated, and a positive duration shorter than one group raises an error (possibly during backward when callbacks are used). If sampling is needed, use an interval no larger than user nt that divides nt, and assess the sampled gradient for your application.
- Ordinary backward releases tensor histories saved by autograd; retaining a graph can extend their lifetime. Disk temporary files live with the corresponding output/loss graph. Log detached values, and do not delete files still needed for backward.

(elastic-examples)=
### Minimal example

With [StarWave 4.0.0 installed](installation.md) and logical CUDA device 0 available, this builds a `[depth,horizontal]` model and applies force along depth. It demonstrates vp/vs/rho → explicit conversion → elastic → a first-order Vs gradient. This documentation update did not execute the GPU example; it is not a numerical or convergence acceptance test.

```python
import torch
import starwave

# The input array axes are [depth, horizontal], mapped to elastic [y, x].
device = torch.device("cuda:0")
starwave.prepare_elastic([0])
vp = torch.full((48, 64), 2200.0, device=device)
vs = torch.full_like(vp, 1100.0, requires_grad=True)
rho = torch.full_like(vp, 2000.0)
lamb, mu, buoyancy = starwave.common.vpvsrho_to_lambmubuoyancy(vp, vs, rho)

dt, nt, freq = 0.001, 100, 15.0
t = torch.arange(nt, device=device) * dt
phase = torch.pi * freq * (t - 0.04)
wavelet = (1.0 - 2.0 * phase.square()) * torch.exp(-phase.square())
force_y = wavelet.reshape(1, 1, nt)  # [shot, source, time], force density
source_y = torch.tensor([[[8, 32]]], device=device, dtype=torch.long)
receiver_y = torch.tensor([[[8, 20], [8, 32], [8, 44]]],
                          device=device, dtype=torch.long)
outputs = starwave.elastic(
    lamb, mu, buoyancy, grid_spacing=(10.0, 10.0), dt=dt,
    source_amplitudes_y=force_y, source_locations_y=source_y,
    receiver_locations_y=receiver_y, pml_freq=freq, accuracy=4,
    memory="boundary",
)
record_y = outputs[-2]  # [1, 3, 100]; the final slot is x, even when absent
loss = record_y.square().mean()
loss.backward()
assert vs.grad is not None
```

For CPU, change device to `torch.device("cpu")`, call `starwave.prepare_elastic()` and use `memory="full"`. Build 3D models in `[z,y,x]` order; if the first axis is depth, a vertical force uses z, coordinates have a final dimension of 3, and p/z/y/x records are `outputs[-4:]`.

(elastic-notes)=
### Notes

- Native propagation provides first-order AD, without a higher-order guarantee. The conversion helpers retain the PyTorch chain but do not add higher-order propagation derivatives. Raw materials are not physically repaired; invalid square roots, ill-conditioned models or insufficient resolution can still produce invalid results.
- There is no public backward_callback, free_surface, boundary_buffer, source_fields or receiver_fields argument; use the actual component keywords. Elastic nt/storage options cannot be added to scalar, VRZ or VTI calls.
- Before DataParallel, call {py:func}`starwave.prepare_elastic` on the main thread, keep complete materials in a Module and split acquisition along shots only. Elastic has a separate preparation entry point from the existing propagators. Successful preparation is not multi-GPU numerical acceptance.
- This update checked the published wheel's interface and Python 3.10 syntax without running a GPU. Existing user-supplied RTX 4060 source-test reports and published-wheel CPU/host checks have their own scopes; they do not establish A30, multi-GPU, long FWI or target-GPU performance acceptance.

(elastic-conversions)=
### Material conversions

```{py:function} starwave.common.vpvsrho_to_lambmubuoyancy(vp: torch.Tensor, vs: torch.Tensor, rho: torch.Tensor, eps: float=1e-15) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

:param vp: **Required.** P-wave speed, typically m/s.
:type vp: `torch.Tensor`
:param vs: **Required.** S-wave speed, typically m/s.
:type vs: `torch.Tensor`
:param rho: **Required.** Density, typically kg/m³.
:type rho: `torch.Tensor`
:param eps: **Default `1e-15`.** Regularizer in the inverse-density/density denominator; used as supplied, without automatic unit scaling.
:type eps: `float`
:returns: `(lamb, mu, buoyancy)`; each expression follows ordinary PyTorch broadcasting/type promotion independently, preserving spatial axes and the autograd chain. Supply matching full model shapes before propagation.
:rtype: `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
```

```{py:function} starwave.common.lambmubuoyancy_to_vpvsrho(lamb: torch.Tensor, mu: torch.Tensor, buoyancy: torch.Tensor, eps: float=1e-15) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

:param lamb: **Required.** First Lamé parameter, typically Pa.
:type lamb: `torch.Tensor`
:param mu: **Required.** Shear modulus, typically Pa.
:type mu: `torch.Tensor`
:param buoyancy: **Required.** Inverse density, typically m³/kg.
:type buoyancy: `torch.Tensor`
:param eps: **Default `1e-15`.** Regularizer in the inverse-density/density denominator; used as supplied, without automatic unit scaling.
:type eps: `float`
:returns: `(vp, vs, rho)`; each expression follows ordinary PyTorch broadcasting/type promotion independently, preserving spatial axes and the autograd chain. Supply matching full model shapes before propagation.
:rtype: `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
```

Forward algebra is `lamb=(vp**2-2*vs**2)*rho`, `mu=vs**2*rho`, and `buoyancy=rho/(rho**2+eps)`. The inverse first computes `vs=sqrt(mu*buoyancy)`, then `vp=sqrt(lamb*buoyancy+2*vs**2)` and `rho=buoyancy/(buoyancy**2+eps)`. The regularizer makes round trips generally approximate; squaring loses negative velocity signs. PyTorch broadcasting is preserved, but the helpers do not repair physical signs, clamp square-root inputs, detach or mutate inputs. The propagator still requires matching spatial shapes.

(native-runtime)=
## Native runtime

(native-status)=
### starwave.native_status

```{py:function} starwave.native_status() -> dict

Queries current native-library status without compiling or propagating.

:returns: Status dictionary. Common entries include `library_exists`, `library_loaded`, `torch_version`, `torch_cuda_version`, and `cuda_available`. Neither existence nor successful loading demonstrates numerical acceptance.
:rtype: `dict`
```

(prepare-native)=
### starwave.prepare_native

```{py:function} starwave.prepare_native(device_ids: list[int] | tuple[int, ...]) -> dict

Validates and preloads an existing native library on the main thread for subsequent propagation or DataParallel use. Does not compile.

:param device_ids: Required. Nonempty, duplicate-free collection of nonnegative visible logical CUDA device IDs; Booleans are rejected. Use the device-visibility mapping of the current process.
:type device_ids: `list[int] | tuple[int, ...]`
:returns: Dictionary containing native status, `selected_device_ids`, and preparation messages. Successful initialization is not a GPU numerical test.
:rtype: `dict`
```

(native-notes)=
## Native runtime notes

Use `native_status()` to diagnose installation and loading. Call `prepare_native()` on the main thread before propagation or DataParallel begins. Neither function compiles the library or replaces GPU numerical acceptance. Logical device `0` in the example must be visible to the current process.

(native-examples)=
## Native runtime examples

After installing the public wheel and a matching PyTorch version, query the status. The subsequent preparation call requires an available CUDA device.

```python
import starwave

status = starwave.native_status()
print(status)
prepared = starwave.prepare_native([0])
```

(prepare-elastic)=
## starwave.prepare_elastic

```{py:function} starwave.prepare_elastic(device_ids: list[int] | tuple[int, ...] | None=None) -> dict

Load the separate elastic native library on the main thread. Prepare explicit logical devices before CUDA/DataParallel use. Does not compile or execute propagation tests.

:param device_ids: **Default `None`.** None loads the elastic library without selecting or preparing a CUDA device list, suitable for CPU use; a nonempty list/tuple selects distinct, nonnegative visible logical CUDA integer IDs (bool is rejected). CUDA requests validate the devices and build capability.
:type device_ids: `list[int] | tuple[int, ...] | None`
:returns: Dictionary with elastic-library status, selected_device_ids and a preparation message; successful loading is not numerical or DataParallel acceptance.
:rtype: `dict`
```

`starwave.elastic_native_status()` reads elastic-library status. The existing `prepare_native()` serves scalar/VRZ/VTI; elastic-only use does not require preparing those propagators.

(other-exports)=
## Other exported names and scope

`ScalarIllumination`, `IlluminationFields`, and `precondition_gradient` are verified exported names related to scalar illumination. This edition does not yet provide a complete lifecycle tutorial for them; the propagator pages explain the usage boundaries of the `illumination` parameter.

StarWave 7.0.0 has no public `starwave.Scalar` wrapper class. The tutorials define their own Module wrapper. Do not assume another library’s classes exist in StarWave, or add elastic state, `nt`, or storage options to scalar/VRZ/VTI calls.

{ref}`genindex`
