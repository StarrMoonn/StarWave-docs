# scalar: 2D scalar acoustics

```{contents} On this page
:local:
:depth: 2
```

## Function

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

## Sources and autograd

The public equation convention is `u_tt = v² (Lap(u) - f)`. The user provides fixed `f`; the internal source increment is `-v_source² * internal_dt² * U(f)`, where `U` is internal upsampling. Do not multiply by `-v² dt²` again externally. The velocity-scaling chain at source locations is retained in the model gradient.

Post-step recordings are aligned to the user clock before downsampling by prepending zero and dropping the final step. The transposes of temporal resampling and shifting remain in autograd; noncausal FFT filtering can cause ringing in the initial user samples.

Model gradients are conditional on a fixed extended model, CFL configuration, and PML. When replicate padding at the physical edges is regenerated, the current returned gradient does not include the transpose accumulation of the complete extension chain. It must not be treated as the total derivative of the entire boundary-rebuilding process; switching to full mode does not remove this limitation.

## Examples

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

For complete input construction without external data files and one FWI update, see [Quickstart](../quickstart.md) and [FWI](../inversion/fwi.md). The default `pml_freq=25.0` does not automatically track the source frequency.

## Notes

- Supports CUDA FP32 on the default stream only; each forward permits one first-order backward pass. Source gradients, higher-order derivatives, AMP, CUDA graphs, custom streams, and public initial/final states are unsupported.
- CFL time substeps cannot compensate for inadequate spatial sampling. Spatial-resolution warnings are diagnostics, not certificates of accuracy or stability.
- For DataParallel, place the complete model in a Module and split acquisition inputs only along the shot dimension; see [Multi-GPU introduction](../inversion/dataparallel.md).
- This page’s interface contract has been checked. Target-GPU numerical, performance, and FWI acceptance tests have not been completed.
