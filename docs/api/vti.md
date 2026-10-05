# vti: 2D and 3D acoustic VTI

```{contents} On this page
:local:
:depth: 2
```

## Function

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

## Components, timing, and gradients

A positive stress rate adds positive stress to every selected stress component; internally, `internal_dt*U(rate)` is applied. There is no scalar-style `-vp² dt²` factor or model-dependent source scaling. Legacy per-step stress increments cannot be used directly as public rate inputs.

Stress recordings are time-aligned before downsampling by prepending zero and removing the final entry. Velocity recordings instead average the preceding and following half-time-step samples onto integer times, taking the initial preceding half step as zero. Both outputs are labeled with nominal user times, but their physical quantities and spatial locations differ. Resampling may produce endpoint ringing.

Each of `vp`, `epsilon`, `delta`, and `rho` requests a gradient only when its own `requires_grad=True`; fixed fields still participate in forward and adjoint propagation. No rule automatically freezes fields based on their values. `nn.Parameter` enables gradients by default; fixed fields can be registered as buffers.

`epsilon<delta` is not automatically repaired. A smaller dt does not guarantee removal of negative-stiffness growth or arbitrary PML instability. Covering the velocity envelope is only a necessary input condition.

Model gradients are conditional on a fixed extended model, CFL configuration, and PML. When replicate padding at the physical edges is regenerated, the current returned gradient does not include the transpose accumulation of the complete extension chain. It must not be treated as the total derivative of the entire boundary-rebuilding process; switching to full mode does not remove this limitation.

## Examples

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

This example uses 2D inputs. A 3D call needs all four models shaped `[nx,ny,nz]`, 3D coordinates, and matching spacing. Omitting `receiver_fields` returns only `vz`, so use single-variable tuple unpacking. VTI has no `illumination` parameter. A complete standalone GPU example remains to be added; see [VTI modeling](../modeling/vti.md).

## Notes

- Supports CUDA FP32 on the default stream only; each forward permits one first-order backward pass. Source gradients, higher-order derivatives, AMP, CUDA graphs, custom streams, and public initial/final states are unsupported.
- CFL time substeps cannot compensate for inadequate spatial sampling. Spatial-resolution warnings are diagnostics, not certificates of accuracy or stability.
- For DataParallel, place the complete model in a Module and split acquisition inputs only along the shot dimension; see [Multi-GPU introduction](../inversion/dataparallel.md).
- This page’s interface contract has been checked. Target-GPU numerical, performance, and FWI acceptance tests have not been completed.
