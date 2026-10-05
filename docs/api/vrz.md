# vrz: 2D variable-density acoustics

```{contents} On this page
:local:
:depth: 2
```

## Function

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

## Parameterization, sources, and gradients

With `impedance=Z`, velocity and impedance are independent inputs. With `density=rho`, impedance is obtained internally from `v*rho`. The two gradient interpretations differ and cannot be interchanged without applying the chain rule. Use `requires_grad=True` for trainable inputs; other medium values still participate in forward propagation.

Sources use the same normalized-forcing convention as scalar: the internal increment is `-v_source² * internal_dt² * U(f)`, rather than a direct pressure increment. Do not apply additional time-step or velocity scaling. Recordings are aligned to user times before downsampling by prepending zero and removing the final step; FFT resampling may produce endpoint ringing.

Positivity, finiteness, coefficient representability, and CFL checks do not guarantee stability for arbitrarily strong impedance contrasts. Near-zero and strong-contrast models may be severely ill-conditioned.

Model gradients are conditional on a fixed extended model, CFL configuration, and PML. When replicate padding at the physical edges is regenerated, the current returned gradient does not include the transpose accumulation of the complete extension chain. It must not be treated as the total derivative of the entire boundary-rebuilding process; switching to full mode does not remove this limitation.

## Examples

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

For the impedance parameterization, replace `density=rho` with `impedance=Z`; do not keep both. See [VRZ modeling](../modeling/vrz.md) for more background. A complete standalone GPU example remains to be added.

## Notes

- Supports CUDA FP32 on the default stream only; each forward permits one first-order backward pass. Source gradients, higher-order derivatives, AMP, CUDA graphs, custom streams, and public initial/final states are unsupported.
- CFL time substeps cannot compensate for inadequate spatial sampling. Spatial-resolution warnings are diagnostics, not certificates of accuracy or stability.
- For DataParallel, place the complete model in a Module and split acquisition inputs only along the shot dimension; see [Multi-GPU introduction](../inversion/dataparallel.md).
- This page’s interface contract has been checked. Target-GPU numerical, performance, and FWI acceptance tests have not been completed.
