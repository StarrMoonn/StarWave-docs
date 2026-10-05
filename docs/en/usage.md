# Usage

```{container} sw-page-toc

**On this page**

- [Scalar Function](#scalar)
- [VRZ Function](#vrz)
- [VTI Function](#vti)
- [starwave.native_status](#native-status)
- [starwave.prepare_native](#prepare-native)
- [Other exported names and scope](#other-exports)
```

This reference targets the public StarWave **2.0.0** wheel. Each propagator includes its actual signature, every parameter, return values, gradient scope, notes, and a call example. Parameter types describe accepted runtime values; signatures retain the actual keyword-only boundaries and defaults.

The page organization follows the Sphinx Python API style of the [official Deepwave Usage documentation](https://ausargeo.com/deepwave/usage). All descriptions are newly written from StarWave’s actual contracts. The libraries’ parameter sets, source units, return structures, and differentiability scopes are not interchangeable.

(propagators)=
## Propagators at a glance

- {py:func}`starwave.scalar`: 2D scalar acoustics; velocity model `v`; returns a one-element tuple of recordings.

- {py:func}`starwave.vrz`: 2D variable-density acoustics; `v` plus exactly one `impedance` / `density` parameterization.

- {py:func}`starwave.vti`: 2D/3D acoustic VTI; `vp, epsilon, delta, rho`; returns recordings in the selected component order.

Notation: `B` is the number of shots, `S` the sources per shot, `R` the receivers per shot, `T` the number of user time samples, and `D` the number of spatial dimensions. Sources and receivers use integer grid indices in the physical model, not coordinates in meters, and do not include PML offsets.

(scalar)=
## Scalar Function

```{py:function} starwave.scalar(v: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, accuracy: int=8, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float | int=0.0, time_pad_frac: float | int=0.0, time_taper: bool=False, illumination: ScalarIllumination | None=None) -> tuple[torch.Tensor]

Propagates scalar acoustic waves on a two-dimensional, equally spaced grid, supporting a batch of independent shots. The output supports one first-order gradient with respect to the velocity model; source waveforms, coordinates, and numerical settings are not trainable inputs. Every call starts with a fresh propagation state and does not return the final wavefield.

:param v: **Required; positional or keyword.** Two-dimensional model `[N0,N1]`, with at least 2 cells in each dimension; CUDA float32. Typical unit: m/s. Coordinates `[i,j]` directly index `v[i,j]`; physical axis order is not inferred automatically. All values must be finite. Negative and zero values are accepted, but this does not guarantee physical validity. Model-axis transformations preserve the autograd chain; `requires_grad=True` requests a velocity gradient.
:type v: `torch.Tensor`
:param grid_spacing: **Required; positional or keyword.** Positive, finite grid spacing, typically in m. Accepts one value or two equal values; scalar/VRZ do not support unequal spacing along the two axes. Booleans, zero/negative values, and mismatched sequence lengths are rejected.
:type grid_spacing: `float | int | list[float] | tuple[float, ...]`
:param dt: **Required; positional or keyword.** Positive, finite sampling interval of the user input/output, typically in s. Internally, the CFL condition may require a smaller time step; the returned sampling interval remains this value. Tensor and Boolean inputs are not accepted, and this parameter is not differentiated.
:type dt: `float | int`
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
:param max_vel: **Default `None`; keyword-only.** CFL/PML velocity envelope, typically in m/s. With `None`, each call replans from the current model’s `max(abs(v))`. An explicit value must be positive, finite, and cover that maximum; it is not a clipping limit. An all-zero model requires an explicit positive value. Extrema, time substeps, and PML configuration are not differentiated.
:type max_vel: `float | int | None`
:param freq_taper_frac: **Default `0.0`; keyword-only.** Finite fraction in `[0,1]` controlling the cosine taper of high-frequency FFT bins during temporal resampling; the bin count is obtained by truncating the proportional count to an integer. During resampling, the last positive rFFT bin is suppressed even when this fraction is 0, including for odd lengths; preservation of every frequency is not guaranteed. This setting does not change the signal when the internal resampling ratio is 1.
:type freq_taper_frac: `float | int`
:param time_pad_frac: **Default `0.0`; keyword-only.** Finite fraction in `[0,1]`. Appends `int(time_pad_frac*T)` user samples of zeros before resampling and removes them afterwards. Does not increase the public recording length or the actual propagation duration. It does not change the signal when the internal resampling ratio is 1.
:type time_pad_frac: `float | int`
:param time_taper: **Default `False`; keyword-only.** Whether to apply a nonperiodic Hann window during temporal resampling: after upsampling and before downsampling. Changes both the signal and its backpropagation transpose. Accepts only Booleans, not the integers 0/1. It does not change the signal when the internal resampling ratio is 1.
:type time_taper: `bool`
:param illumination: **Default `None`; keyword-only.** Optional fresh `ScalarIllumination` collector. Requires gradient tracking to be enabled and a trainable model to be present. Collects detached source, receiver, and geometric-product statistics; it is sealed after forward and reduced after one backward. `None` allocates no illumination buffers. This parameter does not automatically precondition gradients, and the statistics are not an exact Hessian.
:type illumination: `ScalarIllumination | None`
:returns: **`(receiver_amplitudes,)`**, a tuple with exactly one element. The recordings are a contiguous CUDA float32 tensor `[B,R,T]` on the same device as the model, at nominal times `0, dt, ..., (T-1)*dt`. They are pressure-like recordings whose scale depends on the source and unit system. Extract them with `[0]` or `[-1]`. No final wavefield or PML state is included.
:rtype: `tuple[torch.Tensor]`
```

(scalar-details)=
### Sources and autograd

The public equation convention is `u_tt = v² (Lap(u) - f)`. The user provides fixed `f`; the internal source increment is `-v_source² * internal_dt² * U(f)`, where `U` is internal upsampling. Do not multiply by `-v² dt²` again externally. The velocity-scaling chain at source locations is retained in the model gradient.

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

(scalar-notes)=
### Notes

- Supports CUDA FP32 on the default stream only; each forward permits one first-order backward pass. Source gradients, higher-order derivatives, AMP, CUDA graphs, custom streams, and public initial/final states are unsupported.
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

For the impedance parameterization, replace `density=rho` with `impedance=Z`; do not keep both. See [VRZ modeling](modeling/vrz.md) for more background. A complete standalone GPU example remains to be added.

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

This example uses 2D inputs. A 3D call needs all four models shaped `[nx,ny,nz]`, 3D coordinates, and matching spacing. Omitting `receiver_fields` returns only `vz`, so use single-variable tuple unpacking. VTI has no `illumination` parameter. A complete standalone GPU example remains to be added; see [VTI modeling](modeling/vti.md).

(vti-notes)=
### Notes

- Supports CUDA FP32 on the default stream only; each forward permits one first-order backward pass. Source gradients, higher-order derivatives, AMP, CUDA graphs, custom streams, and public initial/final states are unsupported.
- CFL time substeps cannot compensate for inadequate spatial sampling. Spatial-resolution warnings are diagnostics, not certificates of accuracy or stability.
- For DataParallel, place the complete model in a Module and split acquisition inputs only along the shot dimension; see [Multi-GPU introduction](inversion/dataparallel.md).
- This page’s interface contract has been checked. Target-GPU numerical, performance, and FWI acceptance tests have not been completed.

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

(other-exports)=
## Other exported names and scope

`ScalarIllumination`, `IlluminationFields`, and `precondition_gradient` are verified exported names related to scalar illumination. This edition does not yet provide a complete lifecycle tutorial for them; the propagator pages explain the usage boundaries of the `illumination` parameter.

StarWave 2.0.0 has no public `starwave.Scalar` wrapper class. The tutorials define their own Module wrapper. Do not add another library’s class names, state parameters, `nt`, or storage options directly to these calls.

{ref}`genindex`
