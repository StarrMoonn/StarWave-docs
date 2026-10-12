# Usage

GSLS: [visco_gsls](#visco-gsls). Independent source extension: [VRR Function](#vrr), outside the 7.1.0 wheel.

```{container} sw-page-toc

**On this page**

- [Scalar Function](#scalar)
- [Born Function](#born)
- [VRZ Function](#vrz)
- [VRR Function](#vrr)
- [VTI Function](#vti)
- [Elastic Function](#elastic)
- [GSLS Function](#visco-gsls)
- [Material conversions](#elastic-conversions)
- [starwave.native_status](#native-status)
- [starwave.prepare_native](#prepare-native)
- [starwave.prepare_elastic](#prepare-elastic)
- [Other exported names and scope](#other-exports)
```

This reference targets reviewed V17 source `0.1.0.dev17`, including Born, the merged flat-top boundary behavior and import-time native initialization. [Release V17](https://github.com/StarrMoonn/StarWave/releases/tag/V17) is published; PyPI `8.0.0` publication remains pending; the published `7.1.0` wheel is the V16 baseline and does not gain these changes through a documentation update. See [Status](status.md). Existing positional arguments and native defaults are preserved; explicit [PyTorch backend](pytorch-backend.md) options retain their equation-specific limits. Complete signatures, parameters, returns and limits follow.

See [Release Notes](release-notes.md) for 6.0.0 internal storage and PML-transpose maintenance, and [Scalar](modeling/scalar.md) for 2D storage changes. These introduce no new public parameters; source upgrades require rebuilt matching native libraries.

The page organization follows the Sphinx Python API style of the [official Deepwave Usage documentation](https://ausargeo.com/deepwave/usage). All descriptions are newly written from StarWave’s actual contracts. The libraries’ parameter sets, source units, return structures, and differentiability scopes are not interchangeable.

(propagators)=
## Propagators at a glance

- {py:func}`starwave.scalar`: `2D/3D` scalar acoustics; velocity model `v`; returns a one-element tuple of recordings.

- {py:func}`starwave.scalar_born`: `1D/2D/3D` constant-density acoustic Born; velocity perturbation δv, background/scattered records and terminal states. V17 source; 8.0.0 publication pending.

- {py:func}`starwave.vrz`: `2D` variable-density acoustics; `v` plus exactly one `impedance` / `density` parameterization.

- {py:func}`starwave_vrr.vrr`: `2D` acoustic vector reflectivity; independent `v, rx, rz` fields. Separately installed source extension, outside the 7.1.0 wheel.

- {py:func}`starwave.vti`: `2D/3D` acoustic VTI; `vp, epsilon, delta, rho`; returns recordings in the selected component order.

- {py:func}`starwave.elastic`: `2D/3D` isotropic elasticity; `lamb, mu, buoyancy`; complete final states plus p/velocity records.

- {py:func}`starwave.visco_gsls`: 2D generalized-standard-linear-solid viscoacoustics, fixed variable density and native full/checkpoint; see the [GSLS Function](#visco-gsls) section below.

Notation: `B` is the number of shots, `S` the sources per shot, `R` the receivers per shot, `T` the number of user time samples, and `D` the number of spatial dimensions. Sources and receivers use integer grid indices in the physical model, not coordinates in meters, and do not include PML offsets.

(scalar)=
## Scalar Function

```{py:function} starwave.scalar(v: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, accuracy: int=8, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float | int=0.0, time_pad_frac: float | int=0.0, time_taper: bool=False, illumination: ScalarIllumination | None=None, backend: str | None=None, execution: str='eager', checkpoint_interval: int | None=None, compile_steps: int=1) -> tuple[torch.Tensor]

Propagates scalar acoustic waves on a 2D or 3D grid selected by `v.ndim`, with a batch of independent shots. Both dimensions support one first-order velocity gradient; 3D also supports source-waveform gradients. Coordinates and numerical settings are not differentiable. Each call starts with fresh propagation state and does not return final wavefields.

:param v: **Required; positional or keyword.** Two-dimensional `[N0,N1]` or three-dimensional `[N0,N1,N2]` model, with at least 2 cells per dimension; float32 (native CUDA; Torch CPU/CUDA), typically in m/s. `v.ndim` selects the dimension. Coordinates `[i,j]` / `[i,j,k]` directly index `v[i,j]` / `v[i,j,k]`; the 3D examples use `[nx,ny,nz]`, with no automatic physical-axis permutation. All values must be finite. Signed and zero values are accepted without guaranteeing physical validity. Model-axis transformations preserve autograd; `requires_grad=True` requests a velocity gradient.
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
:param pml_width: **Default `20`; keyword-only.** Positive integer thickness in cells, applied to every face, or four equal positive widths in 2D / six in 3D. Face order is low/high for each public model axis. V17 also accepts only the top face zero: `[A,A,0,A]` for `[x,z]`, `[A,A,A,A,0,A]` for `[x,y,z]`, with all other widths equal positive integers A. Other zero faces, all-zero, unequal remaining widths, negative/noninteger/bool values are rejected. See the zero-halo geometry below.
:type pml_width: `int | list[int] | tuple[int, ...]`
:param boundary_buffer: **Default `5`; keyword-only.** Nonnegative buffer in spatial grid cells, not a number of time steps or a checkpoint interval. Boundary mode requires at least `accuracy // 2 + 1`, which is 5 at eighth order; full mode allows 0. All absorbing sides use total padding `pml_width + boundary_buffer + accuracy // 2`; a top-zero face uses only `accuracy // 2` halo cells, without top PML or buffer.
:type boundary_buffer: `int`
:param memory: **Default `'boundary'`; keyword-only.** Backpropagation-history strategy. `"boundary"` stores pressure strips and two terminal pressure fields for reconstruction; the six 3D faces each have width `M=accuracy//2` (Radius-M). In 3D, `"full"` stores unscaled `Lap(u)` history over the whole padded volume at every internal step and is intended for small comparisons. Source-only 3D gradients also retain the selected history; a forward with neither velocity nor source gradients retains none. Full mode can exhaust GPU memory. Native has no automatic fallback, CPU/disk offload or checkpoint. Torch separately supports full/checkpoint (no boundary), with autograd graphs or complete-state checkpoints rather than this native Lap(u) layout.
:type memory: `str`
:param max_vel: **Default `None`; keyword-only.** CFL/PML velocity envelope, typically in m/s. With `None`, each call replans from the current model’s `max(abs(v))`. An explicit value must be positive, finite, and cover that maximum; it is not a clipping limit. An all-zero model requires an explicit positive value. Extrema, time substeps, and PML configuration are not differentiated.
:type max_vel: `float | int | None`
:param freq_taper_frac: **Default `0.0`; keyword-only.** Finite fraction in `[0,1]` controlling the cosine taper of high-frequency FFT bins during temporal resampling; the bin count is obtained by truncating the proportional count to an integer. During resampling, the last positive rFFT bin is suppressed even when this fraction is 0, including for odd lengths; preservation of every frequency is not guaranteed. This setting does not change the signal when the internal resampling ratio is 1.
:type freq_taper_frac: `float | int`
:param time_pad_frac: **Default `0.0`; keyword-only.** Finite fraction in `[0,1]`. Appends `int(time_pad_frac*T)` user samples of zeros before resampling and removes them afterwards. Does not increase the public recording length or the actual propagation duration. It does not change the signal when the internal resampling ratio is 1.
:type time_pad_frac: `float | int`
:param time_taper: **Default `False`; keyword-only.** Whether to apply a nonperiodic Hann window during temporal resampling: after upsampling and before downsampling. Changes both the signal and its backpropagation transpose. Accepts only Booleans, not the integers 0/1. It does not change the signal when the internal resampling ratio is 1.
:type time_taper: `bool`
:param illumination: **Default `None`; keyword-only.** Optional fresh `ScalarIllumination` collector in 2D, requiring enabled gradient tracking and a trainable model. Collects detached source, receiver, and geometric-product statistics; it is sealed after forward and reduced after one backward. In 3D only `None` is accepted; any other value raises `NotImplementedError`. `None` allocates no illumination buffers. These statistics are not an exact Hessian and do not automatically precondition gradients. Must be None for Torch or a scalar top-zero free surface.
:type illumination: `ScalarIllumination | None`
:param backend: **Default `None`; keyword-only.** None preserves native behavior and never automatically selects Torch. `'cuda'` explicitly requires CUDA models; `'torch'` selects CPU/CUDA tensor propagation and autograd. Device follows inputs without implicit model transfer. `'cpu'` is not a backend selector. MPS is supported only for Scalar2D accuracy=4 float32; Scalar3D and other scalar MPS combinations are rejected. Explicitly select full/checkpoint for scalar/VRZ/VTI Torch calls.
:type backend: `str | None`
:param execution: **Default `'eager'`; keyword-only.** Torch accepts `'eager'` or `'compile'`. Compile uses PyTorch Inductor/AOTAutograd and raises on failure without eager fallback. Native propagation only accepts eager.
:type execution: `str`
:param checkpoint_interval: **Default `None`; keyword-only.** Torch checkpoint segment length in internal time steps: a positive integer, excluding bool; None resolves to 32. Keep None for full or native propagation. Longer segments increase the local replay graph and need not reduce peak memory.
:type checkpoint_interval: `int | None`
:param compile_steps: **Default `1`; keyword-only.** Positive integer, excluding bool; consecutive internal steps per Torch execution block. Try 4 for compile; the final partial block is retained. Does not change dt, nt, shots or loss. Native propagation accepts only 1.
:type compile_steps: `int`
:returns: **`(receiver_amplitudes,)`**, a tuple with exactly one element. The recordings are a contiguous float32 (native CUDA; Torch CPU/CUDA) tensor `[B,R,T]` on the same device as the model, at nominal times `0, dt, ..., (T-1)*dt`. They are pressure-like recordings whose scale depends on the source and unit system. Extract them with `[0]` or `[-1]`. No final wavefield or PML state is included.
:rtype: `tuple[torch.Tensor]`
```

**Torch path:** set `backend="torch"` on the same main entry point and explicitly choose `memory="full"` or `"checkpoint"` for CPU/CUDA without requiring a native build. Import still initializes available native libraries. Native history-layout/stream restrictions below describe the native path. See the [PyTorch backend](pytorch-backend.md) for all options and the memory table.

(scalar-details)=
### Sources and autograd

The public equation convention is `u_tt = v² (Lap(u) - f)`. The user supplies fixed `f` in 2D and may train `f` in 3D; the internal source increment is `-v_source² * internal_dt² * U(f)`, where `U` is internal upsampling. Do not multiply by `-v² dt²` again externally. The velocity-scaling chain at source locations is retained in the model gradient.

Post-step recordings are aligned to the user clock before downsampling by prepending zero and dropping the final step. The transposes of temporal resampling and shifting remain in autograd; noncausal FFT filtering can cause ringing in the initial user samples.

Model gradients are conditional on a fixed extended model, CFL configuration, and PML. When replicate padding at the physical edges is regenerated, the current returned gradient does not include the transpose accumulation of the complete extension chain. It must not be treated as the total derivative of the entire boundary-rebuilding process; switching to full mode does not remove this limitation.

With top-zero `pml_width`, the `M=accuracy//2` cells above the physical grid are fixed zero, while user `z=0` remains active. Coordinates retain their original model indices; do not add a halo offset. There is no top PML/buffer. This is a truncated-stencil zero-halo closure, not odd reflection: second order has an effective pressure-release plane one cell above the first physical sample; higher orders do not inherit that image construction or surface convergence order. Native CUDA full/boundary and Torch full/checkpoint preserve the same source/time and frozen-extension model-gradient limits. Scalar2D native free-surface full uses full-grid history rather than its ordinary compact history; boundary retains the required top strips without a full fallback. Illumination is rejected with top-zero. Scalar2D MPS eligibility remains float32, accuracy=4; actual MPS free-surface acceptance is still separate.

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
### Native 3D memory: full and Radius-M boundary

`M=accuracy//2` is the pressure-Laplacian stencil radius, distinct from `boundary_buffer` and `pml_width`. Boundary mode stores six pressure faces of exactly M cells plus two terminal pressure fields, without a full time-volume history or automatic fallback to full. Full mode stores unscaled `Lap(u)` volume history at every internal step. Both modes retain propagation/adjoint workspaces; 3D CPML memory uses directional slabs. Source-only gradients also retain the selected history. Memory grows with internal time steps and shot count, so boundary mode does not guarantee production-scale 3D fits a GPU. See the exact payload formulas in {ref}`Scalar3D reconstruction <reconstruction-scalar3d>`.

(scalar-3d-example)=
### Runnable 3D call: first-order velocity and source gradients

With V17 source, its matching rebuilt native library, and visible logical CUDA device 0 available, the following is a standalone small example. The model uses `[x,y,z]`, spacing is `[dx,dy,dz]`, and coordinates are grid indices. The loss only checks gradient wiring. Syntax and interface have been checked; this documentation update did not execute the GPU example, so these assertions are not numerical acceptance results.

```python
import torch
import starwave

starwave.prepare_native([0])  # Optional explicit device check.
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
    illumination=None, backend=None, execution="eager",
    checkpoint_interval=None, compile_steps=1,
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

- Native supports CUDA FP32 on the default stream only; each forward permits one first-order backward pass. Source gradients remain unsupported in 2D and are supported at first order in 3D. Higher-order derivatives, AMP, CUDA graphs, custom streams, CPU propagation, and public initial/final states are unsupported.
- CFL time substeps cannot compensate for inadequate spatial sampling. Spatial-resolution warnings are diagnostics, not certificates of accuracy or stability.
- For DataParallel, place the complete model in a Module and split acquisition inputs only along the shot dimension; see [Multi-GPU introduction](inversion/dataparallel.md).
- This page’s interface contract has been checked. Target-GPU numerical, performance, and FWI acceptance tests have not been completed.

(born)=
## Born Function

V17 source API; [Release V17](https://github.com/StarrMoonn/StarWave/releases/tag/V17) is published and PyPI 8.0.0 remains pending. Published 7.1.0 does not contain Born. This constant-density acoustic Born operator is derived from [Deepwave 0.0.27](https://github.com/ar4/deepwave/tree/dc2f8af71e1fb67676c50a91a22bdd059bac8772). `scatter` is velocity perturbation δv, in the units of `v`, not relative squared velocity or squared slowness. Its background operator differs from `starwave.scalar`; use the returned background records for compatible observations. No installed Deepwave package is required.

```{py:function} starwave.scalar_born(v: torch.Tensor, scatter: torch.Tensor, grid_spacing: Union[float, Sequence[float]], dt: float, source_amplitudes: Optional[torch.Tensor]=None, source_locations: Optional[torch.Tensor]=None, receiver_locations: Optional[torch.Tensor]=None, bg_receiver_locations: Optional[torch.Tensor]=None, accuracy: int=4, pml_width: Union[int, Sequence[int]]=20, pml_freq: Optional[float]=None, max_vel: Optional[float]=None, survey_pad: Optional[Union[int, Sequence[Optional[int]]]]=None, wavefield_0: Optional[torch.Tensor]=None, wavefield_m1: Optional[torch.Tensor]=None, psiz_m1: Optional[torch.Tensor]=None, psiy_m1: Optional[torch.Tensor]=None, psix_m1: Optional[torch.Tensor]=None, zetaz_m1: Optional[torch.Tensor]=None, zetay_m1: Optional[torch.Tensor]=None, zetax_m1: Optional[torch.Tensor]=None, wavefield_sc_0: Optional[torch.Tensor]=None, wavefield_sc_m1: Optional[torch.Tensor]=None, psiz_sc_m1: Optional[torch.Tensor]=None, psiy_sc_m1: Optional[torch.Tensor]=None, psix_sc_m1: Optional[torch.Tensor]=None, zetaz_sc_m1: Optional[torch.Tensor]=None, zetay_sc_m1: Optional[torch.Tensor]=None, zetax_sc_m1: Optional[torch.Tensor]=None, origin: Optional[Sequence[int]]=None, nt: Optional[int]=None, model_gradient_sampling_interval: int=1, freq_taper_frac: float=0.0, time_pad_frac: float=0.0, time_taper: bool=False, forward_callback: Optional[common.Callback]=None, backward_callback: Optional[common.Callback]=None, callback_frequency: int=1, python_backend: Union[Literal['eager', 'jit', 'compile'], bool]=False, storage_mode: Literal['device', 'cpu', 'disk', 'none']='device', storage_path: str='.', storage_compression: bool=False, *, memory: Literal['full', 'boundary', 'checkpoint']='full', backend=None, execution='eager', checkpoint_interval=None, compile_steps=1) -> Tuple[torch.Tensor, ...]

:param v: **Required.** Positional-or-keyword. Background velocity tensor with 1–3 spatial axes, optionally model-batched; finite float32/64.
:type v: `torch.Tensor`
:param scatter: **Required.** Positional-or-keyword. Velocity perturbation δv, same spatial shape/dtype/device as v; not relative squared velocity or squared slowness.
:type scatter: `torch.Tensor`
:param grid_spacing: **Required.** Positional-or-keyword. Positive scalar or one positive spacing per spatial axis.
:type grid_spacing: `Union[float, Sequence[float]]`
:param dt: **Required.** Positional-or-keyword. User source/receiver sample interval; CFL may subdivide it internally.
:type dt: `float`
:param source_amplitudes: **Default `None`.** Positional-or-keyword. Optional forcing [shot,source,time], matching model dtype/device; source length supplies nt.
:type source_amplitudes: `Optional[torch.Tensor]`
:param source_locations: **Default `None`.** Positional-or-keyword. Optional int64 [shot,source,ndim] physical-grid indices; shared preparation supports common.IGNORE_LOCATION.
:type source_locations: `Optional[torch.Tensor]`
:param receiver_locations: **Default `None`.** Positional-or-keyword. Optional scattered-field receiver indices, int64 [shot,receiver,ndim].
:type receiver_locations: `Optional[torch.Tensor]`
:param bg_receiver_locations: **Default `None`.** Positional-or-keyword. Optional background-field receiver indices, int64 [shot,receiver,ndim].
:type bg_receiver_locations: `Optional[torch.Tensor]`
:param accuracy: **Default `4`.** Positional-or-keyword. Spatial finite-difference order: 2, 4, 6 or 8.
:type accuracy: `int`
:param pml_width: **Default `20`.** Positional-or-keyword. Nonnegative scalar or 2*ndim face widths in spatial-axis/side order. Zero selects a fixed-zero ghost closure on that side.
:type pml_width: `Union[int, Sequence[int]]`
:param pml_freq: **Default `None`.** Positional-or-keyword. Nonnegative PML reference frequency in Hz; zero is accepted. None uses the upstream 25 Hz default with a warning; above-Nyquist values also warn.
:type pml_freq: `Optional[float]`
:param max_vel: **Default `None`.** Positional-or-keyword. Optional fixed CFL/PML velocity bound; None derives it from abs(v). An explicit value is converted to its absolute value; zero rejects, while a value below max(abs(v)) warns rather than rejects. Use a covering bound for valid planning; planning is not differentiated.
:type max_vel: `Optional[float]`
:param survey_pad: **Default `None`.** Positional-or-keyword. Optional geometry-based domain crop padding; an integer or per-face sequence, with None entries retaining that side of the model.
:type survey_pad: `Optional[Union[int, Sequence[Optional[int]]]]`
:param wavefield_0: **Default `None`.** Positional-or-keyword. Optional background current initial wavefield; includes PML, excludes finite-difference halo; differentiable.
:type wavefield_0: `Optional[torch.Tensor]`
:param wavefield_m1: **Default `None`.** Positional-or-keyword. Optional background previous initial wavefield; includes PML, excludes finite-difference halo; differentiable.
:type wavefield_m1: `Optional[torch.Tensor]`
:param psiz_m1: **Default `None`.** Positional-or-keyword. Optional background psiz PML initial memory (3D only); differentiable active entries, ignored non-PML entries have zero derivative.
:type psiz_m1: `Optional[torch.Tensor]`
:param psiy_m1: **Default `None`.** Positional-or-keyword. Optional background psiy PML initial memory (2D/3D); differentiable active entries, ignored non-PML entries have zero derivative.
:type psiy_m1: `Optional[torch.Tensor]`
:param psix_m1: **Default `None`.** Positional-or-keyword. Optional background psix PML initial memory (all dimensions); differentiable active entries, ignored non-PML entries have zero derivative.
:type psix_m1: `Optional[torch.Tensor]`
:param zetaz_m1: **Default `None`.** Positional-or-keyword. Optional background zetaz PML initial memory (3D only); differentiable active entries, ignored non-PML entries have zero derivative.
:type zetaz_m1: `Optional[torch.Tensor]`
:param zetay_m1: **Default `None`.** Positional-or-keyword. Optional background zetay PML initial memory (2D/3D); differentiable active entries, ignored non-PML entries have zero derivative.
:type zetay_m1: `Optional[torch.Tensor]`
:param zetax_m1: **Default `None`.** Positional-or-keyword. Optional background zetax PML initial memory (all dimensions); differentiable active entries, ignored non-PML entries have zero derivative.
:type zetax_m1: `Optional[torch.Tensor]`
:param wavefield_sc_0: **Default `None`.** Positional-or-keyword. Optional scattered current initial wavefield; includes PML, excludes finite-difference halo; differentiable.
:type wavefield_sc_0: `Optional[torch.Tensor]`
:param wavefield_sc_m1: **Default `None`.** Positional-or-keyword. Optional scattered previous initial wavefield; includes PML, excludes finite-difference halo; differentiable.
:type wavefield_sc_m1: `Optional[torch.Tensor]`
:param psiz_sc_m1: **Default `None`.** Positional-or-keyword. Optional scattered psiz PML initial memory (3D only); differentiable active entries, ignored non-PML entries have zero derivative.
:type psiz_sc_m1: `Optional[torch.Tensor]`
:param psiy_sc_m1: **Default `None`.** Positional-or-keyword. Optional scattered psiy PML initial memory (2D/3D); differentiable active entries, ignored non-PML entries have zero derivative.
:type psiy_sc_m1: `Optional[torch.Tensor]`
:param psix_sc_m1: **Default `None`.** Positional-or-keyword. Optional scattered psix PML initial memory (all dimensions); differentiable active entries, ignored non-PML entries have zero derivative.
:type psix_sc_m1: `Optional[torch.Tensor]`
:param zetaz_sc_m1: **Default `None`.** Positional-or-keyword. Optional scattered zetaz PML initial memory (3D only); differentiable active entries, ignored non-PML entries have zero derivative.
:type zetaz_sc_m1: `Optional[torch.Tensor]`
:param zetay_sc_m1: **Default `None`.** Positional-or-keyword. Optional scattered zetay PML initial memory (2D/3D); differentiable active entries, ignored non-PML entries have zero derivative.
:type zetay_sc_m1: `Optional[torch.Tensor]`
:param zetax_sc_m1: **Default `None`.** Positional-or-keyword. Optional scattered zetax PML initial memory (all dimensions); differentiable active entries, ignored non-PML entries have zero derivative.
:type zetax_sc_m1: `Optional[torch.Tensor]`
:param origin: **Default `None`.** Positional-or-keyword. Physical-model origin of supplied initial states, in grid indices and spatial-axis order.
:type origin: `Optional[Sequence[int]]`
:param nt: **Default `None`.** Positional-or-keyword. User sample count when sources are absent; must agree with supplied source length. Zero is allowed. A source-free call needs batch-defining geometry or an initial state.
:type nt: `Optional[int]`
:param model_gradient_sampling_interval: **Default `1`.** Positional-or-keyword. Positive integer M measured in internal propagation steps. Native M>1 requests approximate sampled model gradients; unified Torch requires 1; legacy tensor AD remains exact.
:type model_gradient_sampling_interval: `int`
:param freq_taper_frac: **Default `0.0`.** Positional-or-keyword. Frequency taper fraction for FFT time resampling.
:type freq_taper_frac: `float`
:param time_pad_frac: **Default `0.0`.** Positional-or-keyword. Time padding fraction for FFT resampling.
:type time_pad_frac: `float`
:param time_taper: **Default `False`.** Positional-or-keyword. Apply the optional time taper during resampling.
:type time_taper: `bool`
:param forward_callback: **Default `None`.** Positional-or-keyword. Optional common.Callback observer before forward blocks; see callback clock and mutation limits below.
:type forward_callback: `Optional[common.Callback]`
:param backward_callback: **Default `None`.** Positional-or-keyword. Observer supported by native full and native CUDA boundary; rejected by Python reference and unified Torch.
:type backward_callback: `Optional[common.Callback]`
:param callback_frequency: **Default `1`.** Positional-or-keyword. Positive callback interval on the user-sample clock, independent of M.
:type callback_frequency: `int`
:param python_backend: **Default `False`.** Positional-or-keyword. Legacy False/eager/jit/compile/True selector; do not combine with explicit cuda or unified torch. True uses available OpenMP build metadata to select compile or jit.
:type python_backend: `Union[Literal['eager', 'jit', 'compile'], bool]`
:param storage_mode: **Default `'device'`.** Positional-or-keyword. Native full history placement: device/cpu/disk/none. Other routes require device; none omits propagation contributions to model gradients.
:type storage_mode: `Literal['device', 'cpu', 'disk', 'none']`
:param storage_path: **Default `'.'`.** Positional-or-keyword. Directory for native full disk history; not a model/source transfer setting.
:type storage_path: `str`
:param storage_compression: **Default `False`.** Positional-or-keyword. Native full history quantization; can approximate model gradients. Other routes require False.
:type storage_compression: `bool`
:param memory: **Default `'full'`.** Keyword-only. Native CPU: full. Native CUDA: full/boundary. Unified Torch: full/checkpoint. Legacy reference: full.
:type memory: `Literal['full', 'boundary', 'checkpoint']`
:param backend: **Default `None`.** Keyword-only. None keeps native-by-input-device or legacy python_backend selection; cuda requires CUDA; torch is explicit CPU/CUDA tensor execution. No cpu/native_cpu selector or fallback.
:type backend: `str | None`
:param execution: **Default `'eager'`.** Keyword-only. Unified Torch eager/compile; all other routes require eager. Compile failures propagate.
:type execution: `str | None`
:param checkpoint_interval: **Default `None`.** Keyword-only. Positive internal-step segment length for unified Torch checkpoint only; None selects 32.
:type checkpoint_interval: `int | None`
:param compile_steps: **Default `1`.** Keyword-only. Positive internal steps per bounded Torch compile chunk; other routes require 1.
:type compile_steps: `int | None`
:returns: Full terminal background/scattered/PML state tuple, then background and scattered records; 10/14/18 tensors for 1D/2D/3D.
:rtype: `Tuple[torch.Tensor, ...]`
```

(born-backends)=
### Backends and history storage

| Selection | Device / dtype | Memory | Execution |
|---|---|---|---|
| `backend=None`, `python_backend=False` | CPU float32/64 | `full` | native |
| `backend=None` with CUDA inputs, or `backend="cuda"` | CUDA float32/64 | `full`, `boundary` | native |
| `backend="torch"` | CPU/CUDA float32/64 | `full`, `checkpoint` | `eager`, `compile` |
| `backend=None`, `python_backend="eager"/"jit"/"compile"` | CPU/CUDA float32/64 | `full` | legacy reference |

All routes support 1D/2D/3D and spatial orders 2/4/6/8. Models, source and initial states must agree in floating dtype/device. MPS, AMP, implicit transfers of floating model/source/state inputs and silent fallbacks are unsupported. Geometry is normalized to integer indices on the model device. Native/reference routes require `execution="eager"`, `compile_steps=1`, `checkpoint_interval=None`. Unified Torch requires `python_backend=False`, gradient interval 1, device storage and no compression. Positive integer chunk/interval settings exclude bool. `python_backend=True` uses available native build metadata for the OpenMP-dependent compile/jit choice; it does not verify runtime source hashes.

Native full supports device/cpu/disk/none history and optional quantization. CPU history on native CPU is normalized to device storage. Compression can approximate model gradients. `storage_mode="none"` omits propagation contributions to model gradients, although source scaling can still contribute; it is not a correct full-gradient substitute. Native CUDA has only full and boundary strategies, not native checkpoint.

Boundary is CUDA-only, device storage, uncompressed, and requires each cropped physical extent to exceed `accuracy`. It saves/reconstructs every internal step, including when model gradients are sampled. Boundary memory need not be smaller on small volumes, thick PML or heavily subsampled full history. All native derivatives are first order.

Torch checkpoints retain complete background/scattered/PML state and replay forward segments; they do not drop initial-state/source/model gradients. The first explicit compile can be costly and errors propagate. Born CPU Inductor uses a scoped fusion restriction to avoid a verified backward-fusion problem; CUDA retains normal fusion. This is actual compilation, not an eager fallback or a performance claim.

(born-conventions)=
### Models, source and boundaries

Axes are `[x]`, `[y,x]`, `[z,y,x]`; coordinates index physical cells in that order. Shared, singleton-batched and per-shot models follow the common batch inference. For source-free propagation, provide an initial state or source geometry that establishes batch size as well as `nt`; models and receiver locations alone do not establish a shot batch. Active source positions and the positions within each receiver set must be unique per shot; ignored locations do not participate in this check. Source and receiver sets may overlap. Initial states include PML and exclude the FD halo. z memories are active in 3D, y in 2D/3D and x in every dimension; ignored non-PML memory entries have zero derivatives.

A zero PML face leaves fixed-zero FD ghost cells. The first physical row remains active. Second order places the negative-image wall one cell outside that row; higher orders retain the upstream zero-ghost closure, without a matching odd-image high-order surface claim. It is not elastic traction-free physics.

```text
u_next  = v² dt_internal² L(u) + 2u - u_previous - v² dt_internal² f
us_next = v² dt_internal² L(us) + 2us - us_previous
          + 2v δv dt_internal² L(u) - 2v δv dt_internal² f
```

Background and scattered fields share the pinned scalar PML operator L. The scattered source includes the derivative of velocity-dependent source scaling. Velocity padding is replicated but scatter padding is zero: edge-nonzero δv is not the derivative of a wrapper that re-replicates perturbed velocity into PML. For a finite-difference linearization test, keep the exterior fixed or use perturbations that vanish at the physical boundary. Use fixed covering `max_vel` to keep numerical planning unchanged.

(born-gradients)=
### Internal-step model-gradient sampling

For every positive interval M, native sampling counts internal propagation steps. Default `M=1` stores/images every internal step and matches the first-order discrete model VJP with uncompressed, enabled history and fixed planning, up to floating-point/reconstruction error. `M>1` explicitly requests approximate sampled/weighted model gradients; full allocates `ceil(N_internal/M)` history slots and keeps weight M even on a final partial interval. It does not omit forward/adjoint steps, receiver samples, terminal states or source/initial-state adjoints.

Deepwave 0.0.27 uses effective stride `CFL_ratio * k` for its user interval k. To request that sampling stride here, pass `M=CFL_ratio*k`; StarWave’s default internal stride 1 is a deliberate gradient-accuracy adaptation, not identical upstream sampling semantics. Unified Torch rejects M other than 1; legacy tensor reference accepts the argument but computes complete tensor AD.

Example: `dt=0.003`, `nt=2000`, CFL ratio 2 means 4000 internal steps at 0.0015 s for both full and boundary, with 2000 returned samples. M=2 reduces model imaging sampling only; boundary still reconstructs all 4000 steps. It does not promise 2× speedup.

(born-callbacks)=
### Callback clocks and restrictions

Forward callbacks receive `starwave.common.CallbackState` before a block. Native full and native CUDA boundary also support backward callbacks with model-gradient views; Python reference and unified Torch reject them. Torch checkpoint callbacks execute once in real forward, never during backward replay. Boundary callbacks are checked read-only snapshots: mutations are rejected. Use observer-only callbacks generally.

Callback frequency and `state.step` use the user-sample clock independently of M. `state.dt` is the internal CFL timestep, so `state.step * state.dt` is not physical time when the ratio exceeds one. Preserve the user dt when constructing physical timestamps.

(born-returns)=
### Complete return order

The tuple contains `4+4*ndim` final-state tensors, followed by two record tensors; inactive spatial axes have no placeholder tensors:

1. Background current and previous wavefield
2. Background psi for active axes in z/y/x order, then zeta in the same order
3. Scattered current and previous wavefield
4. Scattered psi for active axes, then scattered zeta
5. Background records `out[-2]`, then scattered records `out[-1]`

There are 10/14/18 tensors in 1D/2D/3D. Records are `[shot,receiver,time]` and observe the current field before update; omitted receivers yield empty tensors. Final states include PML and exclude the FD halo. Continuation must keep internal sampling consistent: independently FFT-resampled source segments are not generally equivalent to one whole-signal resampling.

(born-module)=
### ScalarBorn

```{py:class} starwave.ScalarBorn(v: torch.Tensor, scatter: torch.Tensor, grid_spacing: Union[float, Sequence[float]], v_requires_grad: bool=False, scatter_requires_grad: bool=False, storage_mode: Literal['device', 'cpu', 'disk', 'none']='device', storage_path: str='.', storage_compression: bool=False, *, memory: Literal['full', 'boundary', 'checkpoint']='full', backend=None, execution='eager', checkpoint_interval=None, compile_steps=1)

:param v: **Required.** Positional-or-keyword. Background velocity tensor with 1–3 spatial axes, optionally model-batched; finite float32/64.
:type v: `torch.Tensor`
:param scatter: **Required.** Positional-or-keyword. Velocity perturbation δv, same spatial shape/dtype/device as v; not relative squared velocity or squared slowness.
:type scatter: `torch.Tensor`
:param grid_spacing: **Required.** Positional-or-keyword. Positive scalar or one positive spacing per spatial axis.
:type grid_spacing: `Union[float, Sequence[float]]`
:param v_requires_grad: **Default `False`.** Positional-or-keyword. Whether the module-owned velocity Parameter requires gradients.
:type v_requires_grad: `bool`
:param scatter_requires_grad: **Default `False`.** Positional-or-keyword. Whether the module-owned scatter Parameter requires gradients.
:type scatter_requires_grad: `bool`
:param storage_mode: **Default `'device'`.** Positional-or-keyword. Native full history placement: device/cpu/disk/none. Other routes require device; none omits propagation contributions to model gradients.
:type storage_mode: `Literal['device', 'cpu', 'disk', 'none']`
:param storage_path: **Default `'.'`.** Positional-or-keyword. Directory for native full disk history; not a model/source transfer setting.
:type storage_path: `str`
:param storage_compression: **Default `False`.** Positional-or-keyword. Native full history quantization; can approximate model gradients. Other routes require False.
:type storage_compression: `bool`
:param memory: **Default `'full'`.** Keyword-only. Native CPU: full. Native CUDA: full/boundary. Unified Torch: full/checkpoint. Legacy reference: full.
:type memory: `Literal['full', 'boundary', 'checkpoint']`
:param backend: **Default `None`.** Keyword-only. None keeps native-by-input-device or legacy python_backend selection; cuda requires CUDA; torch is explicit CPU/CUDA tensor execution. No cpu/native_cpu selector or fallback.
:type backend: `str | None`
:param execution: **Default `'eager'`.** Keyword-only. Unified Torch eager/compile; all other routes require eager. Compile failures propagate.
:type execution: `str | None`
:param checkpoint_interval: **Default `None`.** Keyword-only. Positive internal-step segment length for unified Torch checkpoint only; None selects 32.
:type checkpoint_interval: `int | None`
:param compile_steps: **Default `1`.** Keyword-only. Positive internal steps per bounded Torch compile chunk; other routes require 1.
:type compile_steps: `int | None`
:returns: Module with its own v/scatter Parameter wrappers; no storage clone.
:rtype: `starwave.ScalarBorn`
```

The module owns Parameter wrappers without cloning data. Clone inputs first if independent storage is required; gradients attach to module parameters, not original tensor histories. The unannotated selectors remain unannotated in the displayed signature; type fields state accepted values.

(born-forward)=
### ScalarBorn.forward

```{py:method} starwave.ScalarBorn.forward(dt: float, source_amplitudes: Optional[torch.Tensor]=None, source_locations: Optional[torch.Tensor]=None, receiver_locations: Optional[torch.Tensor]=None, bg_receiver_locations: Optional[torch.Tensor]=None, accuracy: int=4, pml_width: Union[int, Sequence[int]]=20, pml_freq: Optional[float]=None, max_vel: Optional[float]=None, survey_pad: Optional[Union[int, Sequence[Optional[int]]]]=None, wavefield_0: Optional[torch.Tensor]=None, wavefield_m1: Optional[torch.Tensor]=None, psiz_m1: Optional[torch.Tensor]=None, psiy_m1: Optional[torch.Tensor]=None, psix_m1: Optional[torch.Tensor]=None, zetaz_m1: Optional[torch.Tensor]=None, zetay_m1: Optional[torch.Tensor]=None, zetax_m1: Optional[torch.Tensor]=None, wavefield_sc_0: Optional[torch.Tensor]=None, wavefield_sc_m1: Optional[torch.Tensor]=None, psiz_sc_m1: Optional[torch.Tensor]=None, psiy_sc_m1: Optional[torch.Tensor]=None, psix_sc_m1: Optional[torch.Tensor]=None, zetaz_sc_m1: Optional[torch.Tensor]=None, zetay_sc_m1: Optional[torch.Tensor]=None, zetax_sc_m1: Optional[torch.Tensor]=None, origin: Optional[Sequence[int]]=None, nt: Optional[int]=None, model_gradient_sampling_interval: int=1, freq_taper_frac: float=0.0, time_pad_frac: float=0.0, time_taper: bool=False, forward_callback: Optional[common.Callback]=None, backward_callback: Optional[common.Callback]=None, callback_frequency: int=1, python_backend: Union[Literal['eager', 'jit', 'compile'], bool]=False, *, memory=None, backend=None, execution=None, checkpoint_interval=None, compile_steps=None) -> Tuple[torch.Tensor, ...]

:param dt: **Required.** Positional-or-keyword. User source/receiver sample interval; CFL may subdivide it internally.
:type dt: `float`
:param source_amplitudes: **Default `None`.** Positional-or-keyword. Optional forcing [shot,source,time], matching model dtype/device; source length supplies nt.
:type source_amplitudes: `Optional[torch.Tensor]`
:param source_locations: **Default `None`.** Positional-or-keyword. Optional int64 [shot,source,ndim] physical-grid indices; shared preparation supports common.IGNORE_LOCATION.
:type source_locations: `Optional[torch.Tensor]`
:param receiver_locations: **Default `None`.** Positional-or-keyword. Optional scattered-field receiver indices, int64 [shot,receiver,ndim].
:type receiver_locations: `Optional[torch.Tensor]`
:param bg_receiver_locations: **Default `None`.** Positional-or-keyword. Optional background-field receiver indices, int64 [shot,receiver,ndim].
:type bg_receiver_locations: `Optional[torch.Tensor]`
:param accuracy: **Default `4`.** Positional-or-keyword. Spatial finite-difference order: 2, 4, 6 or 8.
:type accuracy: `int`
:param pml_width: **Default `20`.** Positional-or-keyword. Nonnegative scalar or 2*ndim face widths in spatial-axis/side order. Zero selects a fixed-zero ghost closure on that side.
:type pml_width: `Union[int, Sequence[int]]`
:param pml_freq: **Default `None`.** Positional-or-keyword. Nonnegative PML reference frequency in Hz; zero is accepted. None uses the upstream 25 Hz default with a warning; above-Nyquist values also warn.
:type pml_freq: `Optional[float]`
:param max_vel: **Default `None`.** Positional-or-keyword. Optional fixed CFL/PML velocity bound; None derives it from abs(v). An explicit value is converted to its absolute value; zero rejects, while a value below max(abs(v)) warns rather than rejects. Use a covering bound for valid planning; planning is not differentiated.
:type max_vel: `Optional[float]`
:param survey_pad: **Default `None`.** Positional-or-keyword. Optional geometry-based domain crop padding; an integer or per-face sequence, with None entries retaining that side of the model.
:type survey_pad: `Optional[Union[int, Sequence[Optional[int]]]]`
:param wavefield_0: **Default `None`.** Positional-or-keyword. Optional background current initial wavefield; includes PML, excludes finite-difference halo; differentiable.
:type wavefield_0: `Optional[torch.Tensor]`
:param wavefield_m1: **Default `None`.** Positional-or-keyword. Optional background previous initial wavefield; includes PML, excludes finite-difference halo; differentiable.
:type wavefield_m1: `Optional[torch.Tensor]`
:param psiz_m1: **Default `None`.** Positional-or-keyword. Optional background psiz PML initial memory (3D only); differentiable active entries, ignored non-PML entries have zero derivative.
:type psiz_m1: `Optional[torch.Tensor]`
:param psiy_m1: **Default `None`.** Positional-or-keyword. Optional background psiy PML initial memory (2D/3D); differentiable active entries, ignored non-PML entries have zero derivative.
:type psiy_m1: `Optional[torch.Tensor]`
:param psix_m1: **Default `None`.** Positional-or-keyword. Optional background psix PML initial memory (all dimensions); differentiable active entries, ignored non-PML entries have zero derivative.
:type psix_m1: `Optional[torch.Tensor]`
:param zetaz_m1: **Default `None`.** Positional-or-keyword. Optional background zetaz PML initial memory (3D only); differentiable active entries, ignored non-PML entries have zero derivative.
:type zetaz_m1: `Optional[torch.Tensor]`
:param zetay_m1: **Default `None`.** Positional-or-keyword. Optional background zetay PML initial memory (2D/3D); differentiable active entries, ignored non-PML entries have zero derivative.
:type zetay_m1: `Optional[torch.Tensor]`
:param zetax_m1: **Default `None`.** Positional-or-keyword. Optional background zetax PML initial memory (all dimensions); differentiable active entries, ignored non-PML entries have zero derivative.
:type zetax_m1: `Optional[torch.Tensor]`
:param wavefield_sc_0: **Default `None`.** Positional-or-keyword. Optional scattered current initial wavefield; includes PML, excludes finite-difference halo; differentiable.
:type wavefield_sc_0: `Optional[torch.Tensor]`
:param wavefield_sc_m1: **Default `None`.** Positional-or-keyword. Optional scattered previous initial wavefield; includes PML, excludes finite-difference halo; differentiable.
:type wavefield_sc_m1: `Optional[torch.Tensor]`
:param psiz_sc_m1: **Default `None`.** Positional-or-keyword. Optional scattered psiz PML initial memory (3D only); differentiable active entries, ignored non-PML entries have zero derivative.
:type psiz_sc_m1: `Optional[torch.Tensor]`
:param psiy_sc_m1: **Default `None`.** Positional-or-keyword. Optional scattered psiy PML initial memory (2D/3D); differentiable active entries, ignored non-PML entries have zero derivative.
:type psiy_sc_m1: `Optional[torch.Tensor]`
:param psix_sc_m1: **Default `None`.** Positional-or-keyword. Optional scattered psix PML initial memory (all dimensions); differentiable active entries, ignored non-PML entries have zero derivative.
:type psix_sc_m1: `Optional[torch.Tensor]`
:param zetaz_sc_m1: **Default `None`.** Positional-or-keyword. Optional scattered zetaz PML initial memory (3D only); differentiable active entries, ignored non-PML entries have zero derivative.
:type zetaz_sc_m1: `Optional[torch.Tensor]`
:param zetay_sc_m1: **Default `None`.** Positional-or-keyword. Optional scattered zetay PML initial memory (2D/3D); differentiable active entries, ignored non-PML entries have zero derivative.
:type zetay_sc_m1: `Optional[torch.Tensor]`
:param zetax_sc_m1: **Default `None`.** Positional-or-keyword. Optional scattered zetax PML initial memory (all dimensions); differentiable active entries, ignored non-PML entries have zero derivative.
:type zetax_sc_m1: `Optional[torch.Tensor]`
:param origin: **Default `None`.** Positional-or-keyword. Physical-model origin of supplied initial states, in grid indices and spatial-axis order.
:type origin: `Optional[Sequence[int]]`
:param nt: **Default `None`.** Positional-or-keyword. User sample count when sources are absent; must agree with supplied source length. Zero is allowed. A source-free call needs batch-defining geometry or an initial state.
:type nt: `Optional[int]`
:param model_gradient_sampling_interval: **Default `1`.** Positional-or-keyword. Positive integer M measured in internal propagation steps. Native M>1 requests approximate sampled model gradients; unified Torch requires 1; legacy tensor AD remains exact.
:type model_gradient_sampling_interval: `int`
:param freq_taper_frac: **Default `0.0`.** Positional-or-keyword. Frequency taper fraction for FFT time resampling.
:type freq_taper_frac: `float`
:param time_pad_frac: **Default `0.0`.** Positional-or-keyword. Time padding fraction for FFT resampling.
:type time_pad_frac: `float`
:param time_taper: **Default `False`.** Positional-or-keyword. Apply the optional time taper during resampling.
:type time_taper: `bool`
:param forward_callback: **Default `None`.** Positional-or-keyword. Optional common.Callback observer before forward blocks; see callback clock and mutation limits below.
:type forward_callback: `Optional[common.Callback]`
:param backward_callback: **Default `None`.** Positional-or-keyword. Observer supported by native full and native CUDA boundary; rejected by Python reference and unified Torch.
:type backward_callback: `Optional[common.Callback]`
:param callback_frequency: **Default `1`.** Positional-or-keyword. Positive callback interval on the user-sample clock, independent of M.
:type callback_frequency: `int`
:param python_backend: **Default `False`.** Positional-or-keyword. Legacy False/eager/jit/compile/True selector; do not combine with explicit cuda or unified torch. True uses available OpenMP build metadata to select compile or jit.
:type python_backend: `Union[Literal['eager', 'jit', 'compile'], bool]`
:param memory: **Default `None`.** Keyword-only. None inherits the constructor setting, including explicit None; a non-None value overrides it. Native CPU: full. Native CUDA: full/boundary. Unified Torch: full/checkpoint. Legacy reference: full.
:type memory: `str | None`
:param backend: **Default `None`.** Keyword-only. None inherits the constructor setting, including explicit None; a non-None value overrides it. None keeps native-by-input-device or legacy python_backend selection; cuda requires CUDA; torch is explicit CPU/CUDA tensor execution. No cpu/native_cpu selector or fallback.
:type backend: `str | None`
:param execution: **Default `None`.** Keyword-only. None inherits the constructor setting, including explicit None; a non-None value overrides it. Unified Torch eager/compile; all other routes require eager. Compile failures propagate.
:type execution: `str | None`
:param checkpoint_interval: **Default `None`.** Keyword-only. None inherits the constructor setting, including explicit None; a non-None value overrides it. Positive internal-step segment length for unified Torch checkpoint only; None selects 32.
:type checkpoint_interval: `int | None`
:param compile_steps: **Default `None`.** Keyword-only. None inherits the constructor setting, including explicit None; a non-None value overrides it. Positive internal steps per bounded Torch compile chunk; other routes require 1.
:type compile_steps: `int | None`
:returns: Full terminal background/scattered/PML state tuple, then background and scattered records; 10/14/18 tensors for 1D/2D/3D.
:rtype: `Tuple[torch.Tensor, ...]`
```

Forward None overrides inherit constructor settings; they cannot reset a stored checkpoint interval/backend. Storage mode/path/compression remain constructor settings and have no per-call override.

(born-example)=
### Small CPU example

```python
import torch
from starwave import scalar_born

v = torch.full((20, 24), 1500.0, dtype=torch.float64, requires_grad=True)
scatter = torch.full_like(v, 10.0, requires_grad=True)
source = torch.zeros((1, 1, 12), dtype=v.dtype)
source[0, 0, 1] = 1.0
source.requires_grad_()
locations = torch.tensor([[[8, 10]]], dtype=torch.int64)
out = scalar_born(
    v, scatter, 10.0, 0.0005,
    source_amplitudes=source, source_locations=locations,
    receiver_locations=locations, bg_receiver_locations=locations,
    accuracy=4, pml_width=4, pml_freq=20.0, max_vel=1600.0,
    backend="torch", memory="checkpoint", checkpoint_interval=4,
)
assert len(out) == 14
assert out[-2].shape == out[-1].shape == (1, 1, 12)
loss = out[-2].square().sum() + out[-1].square().sum()
loss.backward()
assert all(x.grad is not None and torch.isfinite(x.grad).all()
           for x in (v, scatter, source))
```

(born-build)=
### Native build and import initialization

```bash
# Complete authorized V17 source; use the actual GPU SM.
python compile_born.py --arch 80
# CPU-only native full, alternatively:
python compile_born.py --cpu-only
```

Born is in the main Python package, with a separately built native library. In frozen V17 (`27c2e14b184b94d8031a5b3a83c25b1ed3d4ac67`), root `compile_all.py` builds core/elastic/GSLS; Born requires the separate command above. Current dev commit `3f2b24daba0fbe6956389baf95dc910405a7ed4a` also includes Born in `python compile_all.py --arch 80`, building core/elastic/GSLS/Born together. This later change does not alter the frozen V17 archive. Standalone `compile_born.py --cpu-only` remains available; VRR is installed and built separately. Pure Torch requires neither Born native compilation nor a GPU for CPU inputs.

After explicit compilation, V17 `import starwave` loads available core/elastic/Born/GSLS libraries and completes shared-dispatch bindings before returning. Optional Born boundary-entry signatures are bound by their calling path before invocation. An installed/importable independent VRR extension is prepared too, without changing its namespace. Ordinary first parallel propagation needs no serial warm-up or mandatory manual prepare call. Import does not compile, initialize CUDA, select devices, allocate wavefields or propagate. Missing builds are skipped for supported pure-Torch use; an existing invalid build is not silently replaced. Configure build paths before import, rebuild after updates and restart Python/Jupyter. Runtime does not monitor source/library changes or re-hash sources for each propagation. Optional prepare helpers remain available for explicit device checks.

(born-scope)=
### Version and validation scope

This section documents frozen V17 source `27c2e14b184b94d8031a5b3a83c25b1ed3d4ac67` (`0.1.0.dev17`), available as [Release V17](https://github.com/StarrMoonn/StarWave/releases/tag/V17). Source release, native compilation, public-wheel publication, clean installation, CPU numerics and target-device GPU/multi-GPU validation are separate evidence. A documentation build establishes none of the latter by itself. PyPI 8.0.0 remains pending until the release record identifies its verified artifact; the 7.1.0 wheel does not gain Born through this documentation update. Native higher-order AD, MPS, AMP and performance guarantees are outside this contract.

(vrz)=
## VRZ Function

```{py:function} starwave.vrz(v: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, impedance: torch.Tensor | None=None, density: torch.Tensor | None=None, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, accuracy: int=8, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float | int=0.0, time_pad_frac: float | int=0.0, time_taper: bool=False, illumination: None=None, backend: str | None=None, execution: str='eager', checkpoint_interval: int | None=None, compile_steps: int=1) -> tuple[torch.Tensor]

Performs two-dimensional acoustic propagation using velocity and impedance (or density), with batched shots. Specify exactly one parameterization through `impedance` or `density`. The output supports one first-order gradient with respect to the selected model parameters; source waveforms and acquisition coordinates remain fixed.

:param v: **Required; positional or keyword.** Two-dimensional model `[N0,N1]`, with at least 2 cells in each dimension; float32 (native CUDA; Torch CPU/CUDA). Typical unit: m/s. Coordinates `[i,j]` directly index `v[i,j]`; physical axis order is not inferred automatically. Velocity must be positive and finite; scalar’s signed/zero model domain is not accepted. Set `requires_grad=True` to request a model gradient.
:type v: `torch.Tensor`
:param grid_spacing: **Required; positional or keyword.** Positive, finite grid spacing, typically in m. Accepts one value or two equal values; scalar/VRZ do not support unequal spacing along the two axes. Booleans, zero/negative values, and mismatched sequence lengths are rejected.
:type grid_spacing: `float | int | list[float] | tuple[float, ...]`
:param dt: **Required; positional or keyword.** Positive, finite sampling interval of the user input/output, typically in s. Internally, the CFL condition may require a smaller time step; the returned sampling interval remains this value. Tensor and Boolean inputs are not accepted, and this parameter is not differentiated.
:type dt: `float | int`
:param impedance: **Default `None`; keyword-only.** Acoustic impedance `Z`, float32 (native CUDA; Torch CPU/CUDA), with the same shape and device as `v`, positive and finite. Typical SI unit: kg/(m²·s), consistent with `v*rho`. Provide exactly one of this parameter and `density`; providing both or neither is invalid. Training this input uses an independent velocity/impedance parameterization.
:type impedance: `torch.Tensor | None`
:param density: **Default `None`; keyword-only.** Density `rho`, float32 (native CUDA; Torch CPU/CUDA), with the same shape and device as `v`, positive and finite. Choose a consistent unit system; the SI unit is kg/m³. The interface uses the differentiable conversion `Z=v*rho`, and gradients include this chain rule. No implicit unit conversion is performed. Provide exactly one of this parameter and `impedance`.
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
:param pml_width: **Default `20`; keyword-only.** Positive integer thickness in cells or four equal positive widths, in `[x-low,x-high,z-low,z-high]` order. V17 also accepts `[A,A,0,A]` for a flat top, where A is one positive integer width on the remaining faces. Other zero faces, all-zero, unequal remaining widths, negative/noninteger/bool values are rejected. The first physical row remains active; see below.
:type pml_width: `int | list[int] | tuple[int, ...]`
:param boundary_buffer: **Default `5`; keyword-only.** Nonnegative buffer in spatial grid cells, not a number of time steps or a checkpoint interval. Boundary mode requires at least `accuracy // 2 + 1`, which is 5 at eighth order; full mode allows 0. All absorbing sides use total padding `pml_width + boundary_buffer + accuracy // 2`; a top-zero face uses only `accuracy // 2` halo cells, without top PML or buffer. Torch VRZ additionally requires at least accuracy//2 in both full/checkpoint.
:type boundary_buffer: `int`
:param memory: **Default `'boundary'`; keyword-only.** History strategy for model backpropagation. `"boundary"` stores boundary strips and reconstructs the wavefield; `"full"` stores the complete history. Full mode can exhaust GPU memory; native has no automatic fallback or checkpoint. Torch separately supports full/checkpoint (no boundary), with a different graph/checkpoint layout. Forward propagation without a requested model gradient does not retain model-backpropagation history.
:type memory: `str`
:param max_vel: **Default `None`; keyword-only.** CFL/PML velocity envelope, typically in m/s. With `None`, each call replans from the current `max(v)`. An explicit value must be positive, finite, and cover the current maximum velocity; it is not a clipping limit. Planning and PML configuration are not differentiated. This primary-velocity constraint does not guarantee spatial stability for arbitrary impedance contrasts.
:type max_vel: `float | int | None`
:param freq_taper_frac: **Default `0.0`; keyword-only.** Finite fraction in `[0,1]` controlling the cosine taper of high-frequency FFT bins during temporal resampling; the bin count is obtained by truncating the proportional count to an integer. During resampling, the last positive rFFT bin is suppressed even when this fraction is 0, including for odd lengths; preservation of every frequency is not guaranteed. This setting does not change the signal when the internal resampling ratio is 1.
:type freq_taper_frac: `float | int`
:param time_pad_frac: **Default `0.0`; keyword-only.** Finite fraction in `[0,1]`. Appends `int(time_pad_frac*T)` user samples of zeros before resampling and removes them afterwards. Does not increase the public recording length or the actual propagation duration. It does not change the signal when the internal resampling ratio is 1.
:type time_pad_frac: `float | int`
:param time_taper: **Default `False`; keyword-only.** Whether to apply a nonperiodic Hann window during temporal resampling: after upsampling and before downsampling. Changes both the signal and its backpropagation transpose. Accepts only Booleans, not the integers 0/1. It does not change the signal when the internal resampling ratio is 1.
:type time_taper: `bool`
:param illumination: **Default `None`; keyword-only.** Compatibility keyword only; must be `None`. VRZ does not support illumination. Explicitly passing a collector or any other value raises `NotImplementedError`; do not reuse scalar’s illumination setup. Must be None for Torch.
:type illumination: `None`
:param backend: **Default `None`; keyword-only.** None preserves native behavior and never automatically selects Torch. `'cuda'` explicitly requires CUDA models; `'torch'` selects CPU/CUDA tensor propagation and autograd. Device follows inputs without implicit model transfer. `'cpu'` and MPS are rejected. Explicitly select full/checkpoint for scalar/VRZ/VTI Torch calls.
:type backend: `str | None`
:param execution: **Default `'eager'`; keyword-only.** Torch accepts `'eager'` or `'compile'`. Compile uses PyTorch Inductor/AOTAutograd and raises on failure without eager fallback. Native propagation only accepts eager.
:type execution: `str`
:param checkpoint_interval: **Default `None`; keyword-only.** Torch checkpoint segment length in internal time steps: a positive integer, excluding bool; None resolves to 32. Keep None for full or native propagation. Longer segments increase the local replay graph and need not reduce peak memory.
:type checkpoint_interval: `int | None`
:param compile_steps: **Default `1`; keyword-only.** Positive integer, excluding bool; consecutive internal steps per Torch execution block. Try 4 for compile; the final partial block is retained. Does not change dt, nt, shots or loss. Native propagation accepts only 1.
:type compile_steps: `int`
:returns: **`(receiver_amplitudes,)`**, a one-element tuple. Its only recording tensor has shape `[B,R,T]`, is contiguous float32 (native CUDA; Torch CPU/CUDA) on the same device as the model, and has nominal times `0, dt, ..., (T-1)*dt`. These are pressure-like recordings, accessible with `[0]` or `[-1]`; no final wavefield or boundary state is included.
:rtype: `tuple[torch.Tensor]`
```

**Torch path:** set `backend="torch"` on the same main entry point and explicitly choose `memory="full"` or `"checkpoint"` for CPU/CUDA without requiring a native build. Import still initializes available native libraries. Native history-layout/stream restrictions below describe the native path. See the [PyTorch backend](pytorch-backend.md) for all options and the memory table.

(vrz-details)=
### Parameterization, sources, and gradients

With `impedance=Z`, velocity and impedance are independent inputs. With `density=rho`, impedance is obtained internally from `v*rho`. The two gradient interpretations differ and cannot be interchanged without applying the chain rule. Use `requires_grad=True` for trainable inputs; other medium values still participate in forward propagation.

Sources use the same normalized-forcing convention as scalar: the internal increment is `-v_source² * internal_dt² * U(f)`, rather than a direct pressure increment. Do not apply additional time-step or velocity scaling. Recordings are aligned to user times before downsampling by prepending zero and removing the final step; FFT resampling may produce endpoint ringing.

Positivity, finiteness, coefficient representability, and CFL checks do not guarantee stability for arbitrarily strong impedance contrasts. Near-zero and strong-contrast models may be severely ill-conditioned.

Model gradients are conditional on a fixed extended model, CFL configuration, and PML. When replicate padding at the physical edges is regenerated, the current returned gradient does not include the transpose accumulation of the complete extension chain. It must not be treated as the total derivative of the entire boundary-rebuilding process; switching to full mode does not remove this limitation.

The top-zero form `[A,A,0,A]` uses the scalar/VRZ zero-halo convention: the M exterior cells are zero, user `z=0` remains active, and top PML/buffer is removed. Do not offset source/receiver coordinates. Second order has its effective pressure-release plane one cell above the first physical row; a higher interior order does not establish an odd-image high-order surface. CUDA full/boundary and Torch full/checkpoint retain fixed public sources, the velocity-dependent source chain and the existing cropped model-gradient limitation.

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

- Native supports CUDA FP32 on the default stream only; each forward permits one first-order backward pass. Source gradients, higher-order derivatives, AMP, CUDA graphs, custom streams, and public initial/final states are unsupported.
- CFL time substeps cannot compensate for inadequate spatial sampling. Spatial-resolution warnings are diagnostics, not certificates of accuracy or stability.
- For DataParallel, place the complete model in a Module and split acquisition inputs only along the shot dimension; see [Multi-GPU introduction](inversion/dataparallel.md).
- This page’s interface contract has been checked. Target-GPU numerical, performance, and FWI acceptance tests have not been completed.

(vrr)=
## VRR Function

```{py:function} starwave_vrr.vrr(v: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, rx: torch.Tensor, rz: torch.Tensor, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, accuracy: int=8, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float=0.0, time_pad_frac: float=0.0, time_taper: bool=False, backend: str='cuda', execution: str='eager', checkpoint_interval: int | None=None, compile_steps: int=1) -> tuple[torch.Tensor]

Independent **source extension** integrated in the reviewed dev source, distribution `starwave-vrr` (`0.1.0.dev1`), import `starwave_vrr`. It is not part of the published V16 / StarWave 7.1.0 wheel. There is no `starwave.vrr` alias. Obtain complete authorized source containing `extensions/vrr`; this public manual does not distribute the private implementation. See [installation](installation.md) for the separate main package.

:param v: **Required.** Positive finite velocity [nx,nz], m/s.
:type v: `torch.Tensor`
:param grid_spacing: **Required.** Positive uniform spacing: scalar or equal pair, metres.
:type grid_spacing: `float | int | list[float] | tuple[float, ...]`
:param dt: **Required.** Positive user sampling interval, seconds.
:type dt: `float | int`
:param rx: **Required.** Keyword-only. Finite signed x reflectivity-vector field, same shape/device/dtype as v; 1/m.
:type rx: `torch.Tensor`
:param rz: **Required.** Keyword-only. Finite signed z reflectivity-vector field, same shape/device/dtype as v; 1/m.
:type rz: `torch.Tensor`
:param source_amplitudes: **Required.** Keyword-only. Forcing [B,S,T]; explicitly expand shared wavelets across shots/sources.
:type source_amplitudes: `torch.Tensor`
:param source_locations: **Required.** Keyword-only. Integer [B,S,2], or [B,2] for one source per shot.
:type source_locations: `torch.Tensor`
:param receiver_locations: **Required.** Keyword-only. Integer [B,R,2]; geometry indexes unpadded physical models.
:type receiver_locations: `torch.Tensor`
:param accuracy: **Default `8`.** Keyword-only. Interior spatial finite-difference order: 2, 4, 6 or 8.
:type accuracy: `int`
:param pml_freq: **Default `25.0`.** Keyword-only. Positive CPML reference frequency, Hz.
:type pml_freq: `float | int`
:param pml_width: **Default `20`.** Keyword-only. Positive scalar, equal positive four-face widths, or [w,w,0,w] with positive integer w.
:type pml_width: `int | list[int] | tuple[int, ...]`
:param boundary_buffer: **Default `5`.** Keyword-only. Spatial cells; at least accuracy//2+1, including full mode.
:type boundary_buffer: `int`
:param memory: **Default `"boundary"`.** Keyword-only. CUDA: full/boundary. Torch: full/checkpoint; select explicitly.
:type memory: `str`
:param max_vel: **Default `None`.** Keyword-only. Detached current velocity maximum when omitted; explicit positive bound must cover v.
:type max_vel: `float | int | None`
:param freq_taper_frac: **Default `0.0`.** Keyword-only. FFT resampling frequency-taper fraction.
:type freq_taper_frac: `float`
:param time_pad_frac: **Default `0.0`.** Keyword-only. FFT resampling time-padding fraction.
:type time_pad_frac: `float`
:param time_taper: **Default `False`.** Keyword-only. Optional time taper during resampling.
:type time_taper: `bool`
:param backend: **Default `"cuda"`.** Keyword-only. cuda or torch; no automatic backend fallback.
:type backend: `str`
:param execution: **Default `"eager"`.** Keyword-only. Torch: eager/compile. Native CUDA accepts eager only.
:type execution: `str`
:param checkpoint_interval: **Default `None`.** Keyword-only. Torch checkpoint only; None means 32 internal steps, otherwise a positive integer.
:type checkpoint_interval: `int | None`
:param compile_steps: **Default `1`.** Keyword-only. Positive internal steps per compiled chunk; native CUDA accepts only 1.
:type compile_steps: `int`
:returns: Single-element `(record,)` tuple; record shape `[B,R,T]`. Use `result[-1]`; no final states or illumination.
:rtype: tuple[torch.Tensor]
```

(vrr-install)=
### Install and build

From the complete source root, use the intended Python environment with compatible dependencies already installed:

```bash
python -m pip install --no-deps --no-build-isolation -e ./extensions/vrr
python -c "import starwave_vrr; print(starwave_vrr.__file__)"
# Native CUDA only; replace 80 with the actual GPU SM.
python -m starwave_vrr.build --plan --arch 80
python -m starwave_vrr.build --arch 80
```

For `backend="torch"`, omit both build commands. Import and propagation do not implicitly build. Root `compile_all.py` builds the main package only. `--plan` prints a plan without proving compilation; `--arch` accepts `80` or `8.0`, SM 60 or newer. Optional `--nvcc` and `--ccbin` select installed compilers.

Standalone nvcc/C++17 compilation does not use Torch C++ headers or libtorch and can be planned/run with CPU-only Torch. Actual GPU execution still requires CUDA-enabled Torch, a supported GPU and compatible toolkit/driver/runtime. This is not a promise that arbitrary version combinations work. Native C ABI 3 is independent of the main package. Rebuild this exact source and restart Python after source/library changes; old ABI 1/2 binaries cannot be reused. The default cache is `~/.cache/starwave-vrr/`; `STARWAVE_VRR_CACHE` can select another extension cache.

Unregister only this extension with `python -m pip uninstall starwave-vrr`; source/cache cleanup is separate. Editable registration does not replace `starwave`.

(vrr-backends)=
### Backends and gradients

| Backend | Device / dtype | Memory | Execution |
|---|---|---|---|
| `cuda` (default) | CUDA float32 | `full`, `boundary` (default) | `eager` selector; native execution |
| `torch` | CPU/CUDA float32/float64 | `full`, `checkpoint` | `eager`, `compile` |

Selecting Torch alone does not change the boundary default: also pass `memory="full"` or `"checkpoint"`. Invalid combinations raise rather than silently falling back. Torch compile explicitly uses installed Inductor and propagates compilation errors. Native CUDA requires `execution="eager"`, `compile_steps=1`, `checkpoint_interval=None`.

CUDA full stores the six-state trajectory (two pressure levels and four CPML memories). Boundary stores shell history and final states and reconstructs the interior; history still grows with time and shell size. Torch checkpoint stores complete six-state segment boundaries and recomputes forward segments during backward. It is not native boundary reconstruction.

One first-order backward supports independent `v`, `rx`, `rz` and source amplitudes, including source velocity scaling and the full replicated-padding transpose. Geometry and planning inputs are not trainable. MPS, AMP/autocast, CUDA graphs, native checkpoint, Torch boundary, VRR3D and higher-order AD are unsupported. `starwave_vrr.vrr_reference` is an explicitly selected CPU/CUDA float32/64 full-history reference; it has no backend/execution/checkpoint/compile selectors and is not a fallback.

(vrr-physics)=
### Models, source and time conventions

```text
p_tt = v² (p_xx + p_zz) + (v v_x - 2 v² rx) p_x + (v v_z - 2 v² rz) p_z
```

The 2D acoustic vector-reflectivity model uses public `[x,z]` arrays, depth last, and integer `[ix,iz]` geometry. Transpose every conventional `[z,x]` model. `rx/rz` are signed inverse-length fields, not density, Q or dimensionless reflection coefficients. For positive smooth impedance `Z=v*rho`, the continuum relation is `r=0.5*gradient(log Z)`; discrete finite differences need not satisfy the chain/product rules, so this is not discrete VRZ equivalence. No automatic density conversion is provided. Independent fields need not represent a physically realizable impedance.

If pressure has unit P, source forcing has unit P/m²: injection is `-v_source² * dt_internal² * U(source)`, where U is time upsampling. Initial states are zero. Public records apply downsampling to the pre-update-aligned sequence `[0, raw[:-1]]` at user times `0,dt,...`; FFT resampling can ring, so the first public sample need not be exactly zero for every substep ratio.

Internal planning uses `max_dt=0.6*h/(max_vel*sqrt(2))` and `ratio=max(1,ceil(dt/max_dt))`. This principal-speed bound does not guarantee stability for arbitrary reflectivity, velocity gradients or CPML. Use a fixed covering `max_vel` for optimization and finite differences, test the actual time horizon, and refine space/time. The propagator does not clamp/smooth models, repair losses or choose an optimizer.

(vrr-top)=
### Flat top boundary

Face order is `[x-low,x-high,z-low,z-high]`. Only `pml_width=[w,w,0,w]` with positive integer w selects a flat top; a positive scalar or equal positive four-face widths retain all-face absorption. Other zero faces, multiple/all-zero widths and unequal positive absorbers are rejected.

The top removes PML/buffer but keeps `M=accuracy//2` zero halo cells. Physical `z=0` remains active. This is a zero-halo closure, not an odd image extension: at second order the pressure-release plane is the first ghost row; higher interior FD order does not establish matching surface convergence order. No new free-surface switch, arbitrary face or topography support is implied.

(vrr-example)=
### Small synthetic example

```python
import torch
from starwave_vrr import vrr

# A small synthetic CPU example; all models use [x,z].
v = torch.full((32, 32), 1500.0, dtype=torch.float64, requires_grad=True)
rx = torch.zeros_like(v, requires_grad=True)
rz = torch.zeros_like(v, requires_grad=True)
t = torch.arange(40, dtype=v.dtype) * 0.001
phase = torch.pi * 25.0 * (t - 0.02)
source = ((1 - 2 * phase.square()) * torch.exp(-phase.square())).reshape(1, 1, -1)
source = source.detach().requires_grad_()
src = torch.tensor([[[16, 12]]], dtype=torch.int64)
rec = torch.tensor([[[12, 12], [20, 12]]], dtype=torch.int64)
record = vrr(
    v, 10.0, 0.001, rx=rx, rz=rz,
    source_amplitudes=source, source_locations=src, receiver_locations=rec,
    accuracy=4, pml_freq=25.0, pml_width=8, boundary_buffer=5,
    memory="checkpoint", max_vel=1600.0,
    freq_taper_frac=0.0, time_pad_frac=0.0, time_taper=False,
    backend="torch", execution="eager", checkpoint_interval=10, compile_steps=1,
)[-1]
assert record.shape == (1, 2, 40)
record.square().mean().backward()
assert all(x.grad is not None and torch.isfinite(x.grad).all()
           for x in (v, rx, rz, source))
```

For native CUDA, move every model, source and geometry tensor to the same CUDA device, use float32 models/sources, build first, and explicitly select `backend="cuda", memory="boundary", execution="eager", checkpoint_interval=None, compile_steps=1`. The CPU example is a wiring check, not a GPU or physical-convergence validation.

(vrr-status)=
### Source and validation scope

The current reviewed V17 source is `27c2e14b184b94d8031a5b3a83c25b1ed3d4ac67` (`0.1.0.dev17`), including import-time native initialization. The earlier reviewed dev baseline `904c9106bdb883325df8586b7c3412f7814a9737` retained the VRR runtime integrated at `6b9ec41d3bef1a8a3593a63f124c6ee4139955b0`. This dev extension is later than frozen V16 source `5a224dbb1e86154ff034bbedaf67c0e4cf7a1b6a`, which underlies public 7.1.0. Its reviewed VRR feature input is `e8ccf9b0fe70edd023fc4c6d0319abae6c203c65`; the dev integration preserves the existing main-package free-surface work and keeps the extension separately packaged. Release V17 now identifies the frozen source above; the separately packaged VRR extension does not thereby become part of the 7.1.0 wheel, and PyPI 8.0.0 remains pending.

The physical parameterization follows [Whitmore et al., EAGE 2020](https://doi.org/10.3997/2214-4609.202010332). The tensor top convention follows [SWEEP dev](https://github.com/DeepWave-KAUST/sweep/tree/32235f796ffa5b272ca81b9016f8cc85e25ecdf1) commit `32235f796ffa5b272ca81b9016f8cc85e25ecdf1`; the native VRR CUDA step/VJP and boundary reconstruction are independent implementations, not an upstream CUDA port.

The reviewed source records user-supplied RTX 4060 Laptop and A30 numerical reports, including native full/boundary and input gradients; CPU/Inductor, host loading, actual GPU, multi-GPU, long FWI and performance are separate evidence categories. This documentation build does not rerun or certify those GPU results, and no result is transferred to the 7.1.0 wheel or arbitrary hardware/physical regimes. A first-order backward is not a Born/LSRTM or Hessian API.

(vti)=
## VTI Function

```{py:function} starwave.vti(vp: torch.Tensor, epsilon: torch.Tensor, delta: torch.Tensor, rho: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, source_fields: str | list[str] | tuple[str, ...]=('sH', 'sV'), receiver_fields: str | list[str] | tuple[str, ...]=('vz',), accuracy: int=4, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float | int=0.0, time_pad_frac: float | int=0.0, time_taper: bool=False, backend: str | None=None, execution: str='eager', checkpoint_interval: int | None=None, compile_steps: int=1) -> tuple[torch.Tensor, ...]

Propagates a Duveneck-type first-order acoustic VTI system, selecting 2D or 3D from `vp.ndim`. Outputs follow the selected receiver-component order. Each of the four models can independently request a first-order gradient in one backward pass. Fixed models still participate in all physical calculations; this interface is not full elastic VTI.

:param vp: **Required; positional or keyword.** Vertical P-wave velocity in m/s. float32 (native CUDA; Torch CPU/CUDA), positive and finite, with shape `[nx,nz]` or `[nx,ny,nz]` and at least 2 cells per dimension. The last axis is always vertical. All four models must have matching shape, dtype, and device. Physical axis order is neither inferred nor converted automatically.
:type vp: `torch.Tensor`
:param epsilon: **Required; positional or keyword.** Dimensionless Thomsen epsilon, float32 (native CUDA; Torch CPU/CUDA), with the same shape/device as `vp`. Must be finite and strictly greater than -0.5, and the coefficients and derivatives must be numerically representable. `epsilon<delta` is allowed, without implicit clipping; growing modes are a known limitation.
:type epsilon: `torch.Tensor`
:param delta: **Required; positional or keyword.** Dimensionless Thomsen delta, float32 (native CUDA; Torch CPU/CUDA), with the same shape/device as `vp`. Must be finite and strictly greater than -0.5, and the coefficients and derivatives must be numerically representable. The input is not constrained to `delta<=epsilon`; the model is not modified implicitly.
:type delta: `torch.Tensor`
:param rho: **Required; positional or keyword.** Density in kg/m³. float32 (native CUDA; Torch CPU/CUDA), with the same shape/device as `vp`, positive and finite; reciprocals, stiffnesses, and derivatives must be representable. There is no default density or implicit unit conversion. If the original data are confirmed to be in g/cm³, explicitly multiply them by 1000 before calling.
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
:param pml_width: **Default `20`; keyword-only.** Positive integer thickness in cells or `2*D` equal positive widths in low/high axis pairs. V17 also accepts `[A,A,0,A]` in 2D `[x,z]` or `[A,A,A,A,0,A]` in 3D `[x,y,z]`, with equal positive remaining widths A. Only the top face may be zero; other zero faces, all-zero, unequal remaining widths, negative/noninteger/bool values are rejected. This selects the projected normal-traction closure below.
:type pml_width: `int | list[int] | tuple[int, ...]`
:param boundary_buffer: **Default `5`; keyword-only.** Nonnegative buffer in spatial grid cells, not a number of time steps or a checkpoint interval. Boundary mode requires at least `accuracy // 2 + 1`, which is 5 at eighth order; full mode allows 0. All absorbing sides use total padding `pml_width + boundary_buffer + accuracy // 2`; a top-zero face uses only `accuracy // 2` halo cells, without top PML or buffer.
:type boundary_buffer: `int`
:param memory: **Default `'boundary'`; keyword-only.** History strategy for model backpropagation. `"boundary"` stores boundary strips and reconstructs the wavefield; `"full"` stores the complete history. Full mode can exhaust GPU memory; native has no automatic fallback or checkpoint. Torch separately supports full/checkpoint (no boundary), with a different graph/checkpoint layout. Forward propagation without a requested model gradient does not retain model-backpropagation history.
:type memory: `str`
:param max_vel: **Default `None`; keyword-only.** Anisotropic CFL/PML velocity envelope in m/s. With `None`, it is computed dynamically. An explicit value must be positive, finite, and cover `max(vp*sqrt(max(1,1+2*epsilon,sqrt(1+2*delta))))`. Covering only `max(vp)` is insufficient. Density contrast is still recomputed. Planning also depends on dimensionality, spacing, and finite-difference coefficients, and is not differentiated.
:type max_vel: `float | int | None`
:param freq_taper_frac: **Default `0.0`; keyword-only.** Finite fraction in `[0,1]` controlling the cosine taper of high-frequency FFT bins during temporal resampling; the bin count is obtained by truncating the proportional count to an integer. During resampling, the last positive rFFT bin is suppressed even when this fraction is 0, including for odd lengths; preservation of every frequency is not guaranteed. This setting does not change the signal when the internal resampling ratio is 1.
:type freq_taper_frac: `float | int`
:param time_pad_frac: **Default `0.0`; keyword-only.** Finite fraction in `[0,1]`. Appends `int(time_pad_frac*T)` user samples of zeros before resampling and removes them afterwards. Does not increase the public recording length or the actual propagation duration. It does not change the signal when the internal resampling ratio is 1.
:type time_pad_frac: `float | int`
:param time_taper: **Default `False`; keyword-only.** Whether to apply a nonperiodic Hann window during temporal resampling: after upsampling and before downsampling. Changes both the signal and its backpropagation transpose. Accepts only Booleans, not the integers 0/1. It does not change the signal when the internal resampling ratio is 1.
:type time_taper: `bool`
:param backend: **Default `None`; keyword-only.** None preserves native behavior and never automatically selects Torch. `'cuda'` explicitly requires CUDA models; `'torch'` selects CPU/CUDA tensor propagation and autograd. Device follows inputs without implicit model transfer. `'cpu'` and MPS are rejected. Explicitly select full/checkpoint for scalar/VRZ/VTI Torch calls.
:type backend: `str | None`
:param execution: **Default `'eager'`; keyword-only.** Torch accepts `'eager'` or `'compile'`. Compile uses PyTorch Inductor/AOTAutograd and raises on failure without eager fallback. Native propagation only accepts eager.
:type execution: `str`
:param checkpoint_interval: **Default `None`; keyword-only.** Torch checkpoint segment length in internal time steps: a positive integer, excluding bool; None resolves to 32. Keep None for full or native propagation. Longer segments increase the local replay graph and need not reduce peak memory.
:type checkpoint_interval: `int | None`
:param compile_steps: **Default `1`; keyword-only.** Positive integer, excluding bool; consecutive internal steps per Torch execution block. Try 4 for compile; the final partial block is retained. Does not change dt, nt, shots or loss. Native propagation accepts only 1.
:type compile_steps: `int`
:returns: **`(records_0, ..., records_n)`**, with one contiguous float32 (native CUDA; Torch CPU/CUDA) tensor `[B,R,T]` per `receiver_fields` entry, in exactly the requested component order. The default returns only `(vz_records,)`, the vertical particle velocity in m/s, not pressure; `sH/sV` are stresses in Pa. Tensors are on the same device as the model, at nominal user times `0, dt, ..., (T-1)*dt`, without final states.
:rtype: `tuple[torch.Tensor, ...]`
```

**Torch path:** set `backend="torch"` on the same main entry point and explicitly choose `memory="full"` or `"checkpoint"` for CPU/CUDA without requiring a native build. Import still initializes available native libraries. Native history-layout/stream restrictions below describe the native path. See the [PyTorch backend](pytorch-backend.md) for all options and the memory table.

(vti-details)=
### Components, timing, and gradients

A positive stress rate adds positive stress to every selected stress component; internally, `internal_dt*U(rate)` is applied. There is no scalar-style `-vp² dt²` factor or model-dependent source scaling. Legacy per-step stress increments cannot be used directly as public rate inputs.

Stress recordings are time-aligned before downsampling by prepending zero and removing the final entry. Velocity recordings instead average the preceding and following half-time-step samples onto integer times, taking the initial preceding half step as zero. Both outputs are labeled with nominal user times, but their physical quantities and spatial locations differ. Resampling may produce endpoint ringing.

Each of `vp`, `epsilon`, `delta`, and `rho` requests a gradient only when its own `requires_grad=True`; fixed fields still participate in forward and adjoint propagation. No rule automatically freezes fields based on their values. `nn.Parameter` enables gradients by default; fixed fields can be registered as buffers.

`epsilon<delta` is not automatically repaired. A smaller dt does not guarantee removal of negative-stiffness growth or arbitrary PML instability. Covering the velocity envelope is only a necessary input condition.

Model gradients are conditional on a fixed extended model, CFL configuration, and PML. When replicate padding at the physical edges is regenerated, the current returned gradient does not include the transpose accumulation of the complete extension chain. It must not be treated as the total derivative of the entire boundary-rebuilding process; switching to full mode does not remove this limitation.

Top-zero `pml_width` imposes `sV=0` at user `z=0`. Eliminating normal strain gives the surface constitutive coefficient `diag(2*rho*vp**2*(epsilon-delta),0)`; odd normal-stress and even face-velocity images form an exact negative-transpose derivative pair. This is a local pseudo-acoustic closure, not full elastic VTI or a claim of the interior high order at the surface. Sources including `sV` at `z=0` are rejected, including the default two-component source; surface `sV` receivers are zero. `sH` sources and velocity receivers retain native staggering. Native full/boundary keep the top strip; Torch full/checkpoint use the same closure. The projected material chain is differentiated, but the frozen-extension/cropped model VJP and the `epsilon<delta` instability remain.

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

- Native supports CUDA FP32 on the default stream only; each forward permits one first-order backward pass. Source gradients, higher-order derivatives, AMP, CUDA graphs, custom streams, and public initial/final states are unsupported.
- CFL time substeps cannot compensate for inadequate spatial sampling. Spatial-resolution warnings are diagnostics, not certificates of accuracy or stability.
- For DataParallel, place the complete model in a Module and split acquisition inputs only along the shot dimension; see [Multi-GPU introduction](inversion/dataparallel.md).
- This page’s interface contract has been checked. Target-GPU numerical, performance, and FWI acceptance tests have not been completed.

(elastic)=
## Elastic Function

```{py:function} starwave.elastic(lamb: torch.Tensor, mu: torch.Tensor, buoyancy: torch.Tensor, grid_spacing: Union[float, Sequence[float]], dt: float, source_amplitudes_z: Optional[torch.Tensor]=None, source_amplitudes_y: Optional[torch.Tensor]=None, source_amplitudes_x: Optional[torch.Tensor]=None, source_amplitudes_p: Optional[torch.Tensor]=None, source_locations_z: Optional[torch.Tensor]=None, source_locations_y: Optional[torch.Tensor]=None, source_locations_x: Optional[torch.Tensor]=None, source_locations_p: Optional[torch.Tensor]=None, receiver_locations_z: Optional[torch.Tensor]=None, receiver_locations_y: Optional[torch.Tensor]=None, receiver_locations_x: Optional[torch.Tensor]=None, receiver_locations_p: Optional[torch.Tensor]=None, accuracy: int=4, pml_width: Union[int, Sequence[int]]=20, pml_freq: Optional[float]=None, max_vel: Optional[float]=None, survey_pad: Optional[Union[int, Sequence[Optional[int]]]]=None, vz_0: Optional[torch.Tensor]=None, vy_0: Optional[torch.Tensor]=None, vx_0: Optional[torch.Tensor]=None, sigmazz_0: Optional[torch.Tensor]=None, sigmayz_0: Optional[torch.Tensor]=None, sigmaxz_0: Optional[torch.Tensor]=None, sigmayy_0: Optional[torch.Tensor]=None, sigmaxy_0: Optional[torch.Tensor]=None, sigmaxx_0: Optional[torch.Tensor]=None, m_vzz_0: Optional[torch.Tensor]=None, m_vzy_0: Optional[torch.Tensor]=None, m_vzx_0: Optional[torch.Tensor]=None, m_vyz_0: Optional[torch.Tensor]=None, m_vxz_0: Optional[torch.Tensor]=None, m_vyy_0: Optional[torch.Tensor]=None, m_vyx_0: Optional[torch.Tensor]=None, m_vxy_0: Optional[torch.Tensor]=None, m_vxx_0: Optional[torch.Tensor]=None, m_sigmazzz_0: Optional[torch.Tensor]=None, m_sigmayzy_0: Optional[torch.Tensor]=None, m_sigmaxzx_0: Optional[torch.Tensor]=None, m_sigmayzz_0: Optional[torch.Tensor]=None, m_sigmaxzz_0: Optional[torch.Tensor]=None, m_sigmayyy_0: Optional[torch.Tensor]=None, m_sigmaxyy_0: Optional[torch.Tensor]=None, m_sigmaxyx_0: Optional[torch.Tensor]=None, m_sigmaxxx_0: Optional[torch.Tensor]=None, origin: Optional[Sequence[int]]=None, nt: Optional[int]=None, model_gradient_sampling_interval: int=1, freq_taper_frac: float=0.0, time_pad_frac: float=0.0, time_taper: bool=False, forward_callback: Optional[common.Callback]=None, callback_frequency: int=1, python_backend: Union[Literal['eager', 'jit', 'compile'], bool]=False, storage_mode: Literal['device', 'cpu', 'disk', 'none']='device', storage_path: str='.', storage_compression: bool=False, *, memory: Literal['full', 'boundary', 'checkpoint']='full', backend: str | None=None, execution: str='eager', checkpoint_interval: int | None=None, compile_steps: int=1) -> Tuple[torch.Tensor, ...]

2D/3D isotropic elastic propagation with the Deepwave 0.0.27-derived backend. Inputs are raw Lamé parameters and buoyancy, with multi-shot first-order differentiation of materials, sources and initial states. memory and backend/execution/checkpoint_interval/compile_steps are keyword-only; remaining arguments are positional-or-keyword. Only the first five are required.

:param lamb: **Required.** First Lamé parameter, typically Pa. Spatial shape `[Ny,Nx]` (2D) or `[Nz,Ny,Nx]` (3D), optionally with a leading model batch of 1 or B. CPU/float32 (native CUDA; Torch CPU/CUDA) or float64, with finite values. Spatial shape, dtype and device must match mu and buoyancy; this input is never interpreted as vp.
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
:param pml_width: **Default `20`.** Positive integer thickness in cells or length-2D positive face widths in low/high axis order; positive faces may differ. V17 accepts top-zero `[0,A,B,C]` for 2D `[y,x]` and `[0,A,B,C,D,E]` for 3D `[z,y,x]`, with every remaining width positive. Other zero faces and all-zero are rejected. This selects improved-vacuum preparation on the first model axis: row 0 is fixed vacuum and row 1 is the first solid stress row. Top-zero rejects survey_pad, origin and supplied initial fields; see below.
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
:param model_gradient_sampling_interval: **Default `1`.** Material-gradient time-sampling interval, an integer ≥1. The internal accumulation stride is the CFL ratio q times this value. The default 1 is the simplest choice; larger values use sampled gradients rather than promising every-step equivalence. Current native paths process complete sampling groups; see storage and sampling below. Torch accepts only 1 and differentiates all internal steps.
:type model_gradient_sampling_interval: `int`
:param freq_taper_frac: **Default `0.0`.** Fraction of the high-frequency end to cosine-taper during CFL-driven FFT resampling of sources and receivers; does not change user output length.
:type freq_taper_frac: `float`
:param time_pad_frac: **Default `0.0`.** Fractional zero-padding added before FFT resampling and removed afterwards for sources and receivers; does not extend the public record duration.
:type time_pad_frac: `float`
:param time_taper: **Default `False`.** Whether to apply a temporal Hann window to sources and receivers during resampling.
:type time_taper: `bool`
:param forward_callback: **Default `None`.** Observer receiving `starwave.common.CallbackState`. Native callbacks run on backend outer-loop groups; state.dt is internal h and state.step is the outer-loop index. Boundary snapshots are read-only. Torch accepts None only.
:type forward_callback: `Optional[common.Callback]`
:param callback_frequency: **Default `1`.** Positive number of backend outer-loop groups between callbacks. A native group spans q × model_gradient_sampling_interval internal steps; it is not a fixed user-record sampling interval.
:type callback_frequency: `int`
:param python_backend: **Default `False`.** With backend=None, False selects C/CUDA; `"eager"`, `"jit"` and `"compile"` explicitly select PyTorch paths. True selects compile when a verified build declares OpenMP, otherwise jit; availability depends on PyTorch/toolchain. The existing python_backend compatibility paths require full mode, device storage and no compression. With new backend="torch", only False or eager is accepted; select compilation with execution="compile".
:type python_backend: `Union[Literal['eager', 'jit', 'compile'], bool]`
:param storage_mode: **Default `'device'`.** Intermediate storage: `"device"`, `"cpu"`, `"disk"`, or `"none"`. Offload/disk/none are native full options; none does not provide complete material gradients, as explained below. Torch accepts device only.
:type storage_mode: `Literal['device', 'cpu', 'disk', 'none']`
:param storage_path: **Default `'.'`.** Directory for disk-mode temporary histories. Keeping the computation graph can keep these files alive; do not delete files still needed for backward.
:type storage_path: `str`
:param storage_compression: **Default `False`.** Lossy intermediate compression for native full mode; may change material gradients. Boundary and Python backends reject True. Torch accepts False only.
:type storage_compression: `bool`
:param memory: **Default `'full'`.** StarWave keyword-only extension. `"full"` stores volume histories; `"boundary"` uses CUDA boundary storage and reconstruction. Both support 2D/3D and orders 2/4/6/8. Unsupported combinations fail rather than silently falling back to full. Torch additionally supports checkpoint only through backend="torch"; Torch boundary is unsupported.
:type memory: `Literal['full', 'boundary', 'checkpoint']`
:param backend: **Default `None`; keyword-only.** None preserves native behavior and never automatically selects Torch. `'cuda'` explicitly requires CUDA models; `'torch'` selects CPU/CUDA tensor propagation and autograd. Device follows inputs without implicit model transfer. `'cpu'` and MPS are rejected. Explicitly select full/checkpoint for scalar/VRZ/VTI Torch calls.
:type backend: `str | None`
:param execution: **Default `'eager'`; keyword-only.** Torch accepts `'eager'` or `'compile'`. Compile uses PyTorch Inductor/AOTAutograd and raises on failure without eager fallback. Native propagation only accepts eager.
:type execution: `str`
:param checkpoint_interval: **Default `None`; keyword-only.** Torch checkpoint segment length in internal time steps: a positive integer, excluding bool; None resolves to 32. Keep None for full or native propagation. Longer segments increase the local replay graph and need not reduce peak memory.
:type checkpoint_interval: `int | None`
:param compile_steps: **Default `1`; keyword-only.** Positive integer, excluding bool; consecutive internal steps per Torch execution block. Try 4 for compile; the final partial block is retained. Does not change dt, nt, shots or loss. Native propagation accepts only 1.
:type compile_steps: `int`
:returns: Complete final wavefield/PML states and receiver tuple: 16 tensors in 2D, 31 in 3D. Requested records have shape `[B,R,T]`; absent receiver components remain empty tensors. Exact order is listed below.
:rtype: `Tuple[torch.Tensor, ...]`
```

**Torch path:** set `backend="torch"` on the same main entry point and explicitly choose `memory="full"` or `"checkpoint"` for CPU/CUDA without requiring a native build. Import still initializes available native libraries. Native history-layout/stream restrictions below describe the native path. See the [PyTorch backend](pytorch-backend.md) for all options and the memory table.

(elastic-axes)=
### Model axes and staggered grid

The 2D API axes are `[y,x]`, and 3D axes are `[z,y,x]`. These names follow the input tensor's spatial axes; they do not determine the physical vertical direction independently of your array convention. A 2D `[depth,horizontal]` array can be passed directly: y then means depth, and a vertical force uses source_amplitudes_y. After an explicit transpose to `[horizontal,depth]`, depth corresponds to x; transform source/receiver components, coordinates, grid_spacing, PML faces and origin consistently. Neither material conversion transposes arrays.

Materials may be shared or per shot; sources and records have shapes `[B,S,T]` and `[B,R,T]`. The last coordinate dimension follows the same model axes. Components occupy different half-grid positions on the staggered grid, so they are not interchangeable colocated quantities. Velocity sources/receivers cannot use the last model index along their own component axis. See the [Deepwave elastic staggered grid](https://ausargeo.com/deepwave/elastic.html).

Sources y/x (and z in 3D) are force densities; source p is a pressure rate. Receivers y/x/z record velocity; receiver p records minus the mean normal stress. Nominal pressure samples are at `t*dt`, and velocity samples at `(t-0.5)*dt`. CFL subdivision uses the internal half-step and FFT resampling without a separate half-step retiming operation. Do not reuse scalar's forcing or recording clock as the elastic contract.

For a top-zero face, the first model axis is the depth axis. The improved-vacuum preparation replaces row 0 lambda, mu and buoyancy by fixed zeros without modifying caller tensors; their material gradients on that row are zero. The first solid stress row is row 1, with normal-velocity interface buoyancy following the vacuum/solid averaging. Normal body-force sources at the staggered interface remain allowed; pressure sources in the vacuum row are rejected. This is not a collocated clamp of every traction on the first solid row or arbitrary topography. Top-zero requires positive buoyancy below row 0 and rejects `survey_pad`, `origin` and every supplied initial field. Native CUDA full/boundary, native CPU full and Torch full/checkpoint retain their individual restrictions. All-positive PML retains the existing route, including caller-provided vacuum materials; no source/receiver index shift is added.

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

- `memory="full"` is the default, supporting 2D/3D CPU/float32 (native CUDA; Torch CPU/CUDA)/float64. Native `storage_mode="device"` stores histories on the current device; `"cpu"` uses host storage for CUDA and is normalized to device storage for CPU models; `"disk"` uses storage_path. `storage_compression=True` is lossy.
- `storage_mode="none"` disables propagation-history contributions to material gradients and cannot provide full material gradients for inversion. Material-dependent force-source scaling can still contribute partial gradients. Trainable materials trigger a warning; source/initial-state gradients are a separate path.
- `memory="boundary"` is a StarWave extension: CUDA only, with `python_backend=False`, `storage_mode="device"` and `storage_compression=False`. Every cropped physical extent must exceed accuracy; active staggered buoyancy must be nonzero when its gradient is requested, except the fixed top-vacuum tangential slots of the top-zero preparation. CPU, offload and compression are unsupported, with no automatic full fallback. Memory/speed benefits depend on domain shape, PML and sampling.
- Keep `model_gradient_sampling_interval=1` for the simplest behavior; when CFL ratio q>1, the material-gradient stride is still q, rather than every internal step. For larger integers, current native full/boundary paths execute complete sampling groups only: an incomplete tail is not propagated, and a positive duration shorter than one group raises an error (possibly during backward when callbacks are used). If sampling is needed, use an interval no larger than user nt that divides nt, and assess the sampled gradient for your application.
- Ordinary backward releases tensor histories saved by autograd; retaining a graph can extend their lifetime. Disk temporary files live with the corresponding output/loss graph. Log detached values, and do not delete files still needed for backward.

Torch supports full/checkpoint with `model_gradient_sampling_interval=1`, `forward_callback=None`, `storage_mode="device"`, `storage_compression=False`, and python_backend False or eager only. It differentiates every internal step. Native samples material gradients at CFL step_ratio × sampling_interval, so exact native/Torch material-gradient comparisons also require step_ratio=1.

(elastic-examples)=
### Minimal example

With [StarWave 4.0.0 installed](installation.md) and logical CUDA device 0 available, this builds a `[depth,horizontal]` model and applies force along depth. It demonstrates vp/vs/rho → explicit conversion → elastic → a first-order Vs gradient. This documentation update did not execute the GPU example; it is not a numerical or convergence acceptance test.

```python
import torch
import starwave

# The input array axes are [depth, horizontal], mapped to elastic [y, x].
device = torch.device("cuda:0")
starwave.prepare_elastic([0])  # Optional explicit device check.
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

For native CPU, build its matching library, change device to `torch.device("cpu")` and use `memory="full"`; `starwave.prepare_elastic()` is optional. Build 3D models in `[z,y,x]` order; if the first axis is depth, a vertical force uses z, coordinates have a final dimension of 3, and p/z/y/x records are `outputs[-4:]`.

(elastic-notes)=
### Notes

- Native propagation provides first-order AD, without a higher-order guarantee. The conversion helpers retain the PyTorch chain but do not add higher-order propagation derivatives. Raw materials are not physically repaired; invalid square roots, ill-conditioned models or insufficient resolution can still produce invalid results.
- There is no public backward_callback, free_surface, boundary_buffer, source_fields or receiver_fields argument; use the actual component keywords. Elastic nt/storage options cannot be added to scalar, VRZ or VTI calls.
- V17 import initializes the available elastic library before DataParallel; {py:func}`starwave.prepare_elastic` remains an optional explicit device check. Keep complete materials in a Module and split acquisition along shots only. Successful initialization is not multi-GPU numerical acceptance.
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

(visco-gsls)=
## GSLS Function

This section describes V17 source `0.1.0.dev17`, including the merged top-zero pressure-release boundary and import lifecycle. The two-dimensional `starwave.visco_gsls` is the current viscoacoustic entry point; a single mechanism uses explicit `mode="sls_compat", n_mechanisms=1`. The standalone `visco_sls`, `prepare_visco_sls` and `visco_sls_native_status` APIs are removed. Published [PyPI 7.1.0](https://pypi.org/project/starwave/7.1.0/) includes the earlier V16 GSLS API, without the V17 boundary/initialization changes; historical `7.0.0` has no GSLS. Release V17 is published; PyPI `8.0.0` remains pending. See [documentation status](status.md) and [installation](installation.md).

Supported: fixed spatially varying density, float32/float64, and first-order Vp/Q/source/provider-parameter gradients. Native CPU/CUDA support full/checkpoint; explicit Torch supports full/checkpoint and eager/compile. All main propagators gain execution options while native defaults remain unchanged; see the [PyTorch backend](pytorch-backend.md).

```{py:function} starwave.visco_gsls(vp: torch.Tensor, q: torch.Tensor, rho: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, float], dt: float | int, *, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, f_ref: float | int, accuracy: int=4, pml_width: int | list[int] | tuple[int, int]=20, memory: str='full', max_vel: float | int | None=None, backend: str='cuda', mode: str='hao1', frequency_band: tuple[float, float] | list[float] | None=None, n_mechanisms: int | None=None, fit_tolerance: float | int | None=0.05, strict: bool=False, material_model: Callable | None=None, checkpoint_interval: int | None=None, execution: str='eager', compile_steps: int=1) -> tuple[torch.Tensor]

:param vp: **Required; positional-or-keyword.** Positive finite float32/64 tensor `[z,x]`; phase speed at f_ref, m/s; optionally trainable
:type vp: `torch.Tensor`
:param q: **Required; positional-or-keyword.** Matching positive tensor `[z,x]`, including +infinity. `hao1` (default): paper nominal Q0, not exact Qref; passivity Q0>S. `band_fit`: spatial fitting target, not exact Qref. `sls_compat`: exact modulus Q at f_ref. Optionally trainable
:type q: `torch.Tensor`
:param rho: **Required; positional-or-keyword.** Matching fixed positive finite density tensor, kg/m³; spatial variation supported, `requires_grad=True` rejected
:type rho: `torch.Tensor`
:param grid_spacing: **Required; positional-or-keyword.** Positive scalar or `(dz,dx)` in metres
:type grid_spacing: `float | int | list[float] | tuple[float, float]`
:param dt: **Required; positional-or-keyword.** Positive seconds; conservative CFL guard uses unrelaxed speed and density contrast; no resampling
:type dt: `float | int`
:param source_amplitudes: **Required; keyword-only.** Matching float tensor `[shot,source,time]`; second-order forcing Pa/m²; optional gradients
:type source_amplitudes: `torch.Tensor`
:param source_locations: **Required; keyword-only.** Integer `[shot,source,2]`, unpadded `[z,x]` coordinates; duplicates add. Top-zero requires source z>=1
:type source_locations: `torch.Tensor`
:param receiver_locations: **Required; keyword-only.** Integer `[shot,receiver,2]`; duplicates repeat samples. Top-zero surface pressure receivers return zero
:type receiver_locations: `torch.Tensor`
:param f_ref: **Required; keyword-only.** Positive Hz; exact phase-reference frequency, also used for fixed PML setup
:type f_ref: `float | int`
:param accuracy: **Default `4`; keyword-only.** Staggered FD order 2,4,6,8
:type accuracy: `int`
:param pml_width: **Default `20`; keyword-only.** Positive integer or positive `(z,x)` pair for symmetric absorbing faces; alternatively `[0,bottom,left,right]` selects the flat top, with all remaining widths positive and allowed to differ. Zero scalar/pair entries, all-positive four-face lists, other zero faces, bool and negative/noninteger values are rejected. Replicate extension retains its full accumulating gradient
:type pml_width: `int | list[int] | tuple[int, int]`
:param memory: **Default `'full'`; keyword-only.** full or checkpoint with native CPU/CUDA; Torch also supports full/checkpoint; boundary is unsupported
:type memory: `str`
:param max_vel: **Default `None`; keyword-only.** Optional fixed unrelaxed-speed envelope, m/s. Required for any trainable material coefficients (including closure parameters) with nonzero PML. Too-small values reject
:type max_vel: `float | int | None`
:param backend: **Default `'cuda'`; keyword-only.** Explicit `cuda`, `native_cpu` or `torch`; no fallback. CUDA/native require built matching library
:type backend: `str`
:param mode: **Default `'hao1'`; keyword-only.** Default `hao1`; explicit `band_fit` or `sls_compat`
:type mode: `str`
:param frequency_band: **Default `None`; keyword-only.** Required design `(scale,200*scale)` Hz for hao1 (Table 4/Eq.45); required arbitrary positive ordered fitting band for band_fit. A hao1 source working band may be a subset of the design band
:type frequency_band: `tuple[float, float] | list[float] | None`
:param n_mechanisms: **Default `None`; keyword-only.** Default None selects 5 for hao1 or 3 for band_fit. Explicit hao1 requires 5; band_fit supports integers 2–6; sls_compat requires explicit 1
:type n_mechanisms: `int | None`
:param fit_tolerance: **Default `0.05`; keyword-only.** Positive relative inverse-Q tolerance; default dense-grid warning above 0.05; None skips check
:type fit_tolerance: `float | int | None`
:param strict: **Default `False`; keyword-only.** If true, exceeding fit_tolerance raises rather than warns
:type strict: `bool`
:param material_model: **Default `None`; keyword-only.** Optional advanced callable `(vp_padded,q_padded,dt,f_ref)->(c_rel2,strength,A,B)`; overrides built-in material options. See [provider protocol](#visco-gsls-provider)
:type material_model: `Callable | None`
:param checkpoint_interval: **Default `None`; keyword-only.** Positive integer segment length, checkpoint only. Native None balances complete-state and scratch history memory using input shape; Torch None is 32 internal steps. Neither uses GPU type
:type checkpoint_interval: `int | None`
:param execution: **Default `'eager'`; keyword-only.** Torch accepts `'eager'` or `'compile'`. Compile uses PyTorch Inductor/AOTAutograd and raises on failure without eager fallback. Native propagation only accepts eager.
:type execution: `str`
:param compile_steps: **Default `1`; keyword-only.** Positive integer, excluding bool; consecutive internal steps per Torch execution block. Try 4 for compile; the final partial block is retained. Does not change dt, nt, shots or loss. Native propagation accepts only 1.
:type compile_steps: `int`
:returns: One-element tuple `(pressure_records,)`, shape `[B,R,T]`, in Pa, with the model dtype/device.
:rtype: `tuple[torch.Tensor]`
```

<!-- api-doc:visco_gsls:signature:start -->
```text
starwave.visco_gsls(vp, q, rho, grid_spacing, dt, *, source_amplitudes,
           source_locations, receiver_locations, f_ref, accuracy=4,
           pml_width=20, memory="full", max_vel=None, backend="cuda",
           mode="hao1", frequency_band=None, n_mechanisms=None,
           fit_tolerance=0.05, strict=False, material_model=None,
           checkpoint_interval=None, execution="eager", compile_steps=1)
```
<!-- api-doc:visco_gsls:signature:end -->

The first five parameters are positional-or-keyword; all others are keyword-only. `source_amplitudes`, `source_locations`, `receiver_locations` and `f_ref` are required. The four-entry V17 top-zero form may be a list or tuple, under the restrictions below. The table lists every parameter and signature default.

<!-- api-doc:visco_gsls:parameters:start -->
| Parameter | Calling kind | Default when omitted | Type, shape and units | Purpose and behavior | Allowed values, rejections and limits |
|---|---|---|---|---|---|
| `vp` | `positional-or-keyword` | `required` | Tensor [nz,nx]; m/s | Reference phase speed| Positive finite float32/64 tensor `[z,x]`; phase speed at f_ref, m/s; optionally trainable |
| `q` | `positional-or-keyword` | `required` | Tensor [nz,nx]; dimensionless | Mode-specific Q| Matching positive tensor `[z,x]`, including +infinity. `hao1` (default): paper nominal Q0, not exact Qref; passivity Q0>S. `band_fit`: spatial fitting target, not exact Qref. `sls_compat`: exact modulus Q at f_ref. Optionally trainable |
| `rho` | `positional-or-keyword` | `required` | Tensor [nz,nx]; kg/m³ | Fixed density| Matching fixed positive finite density tensor, kg/m³; spatial variation supported, `requires_grad=True` rejected |
| `grid_spacing` | `positional-or-keyword` | `required` | real or pair; m | Grid spacing| Positive scalar or `(dz,dx)` in metres |
| `dt` | `positional-or-keyword` | `required` | real; s | Fixed time step| Positive seconds; conservative CFL guard uses unrelaxed speed and density contrast; no resampling |
| `source_amplitudes` | `keyword-only` | `required` | Tensor [B,S,T]; Pa/m² | Second-order forcing| Matching float tensor `[shot,source,time]`; second-order forcing Pa/m²; optional gradients |
| `source_locations` | `keyword-only` | `required` | int32/int64 Tensor [B,S,2]; cells | Unpadded source indices| Integer `[shot,source,2]`, unpadded `[z,x]` coordinates; duplicates add. Top-zero requires source z>=1 |
| `receiver_locations` | `keyword-only` | `required` | int32/int64 Tensor [B,R,2]; cells | Unpadded receiver indices| Integer `[shot,receiver,2]`; duplicates repeat samples. Top-zero surface pressure receivers return zero |
| `f_ref` | `keyword-only` | `required` | real; Hz | Phase reference| Positive Hz; exact phase-reference frequency, also used for fixed PML setup |
| `accuracy` | `keyword-only` | `4` | integer; dimensionless | Staggered spatial order| Staggered FD order 2,4,6,8 |
| `pml_width` | `keyword-only` | `20` | integer, pair or top-zero face list; cells | Absorbing PML or flat top| Positive integer or positive `(z,x)` pair for symmetric absorbing faces; alternatively `[0,bottom,left,right]` selects the flat top, with all remaining widths positive and allowed to differ. Zero scalar/pair entries, all-positive four-face lists, other zero faces, bool and negative/noninteger values are rejected. Replicate extension retains its full accumulating gradient |
| `memory` | `keyword-only` | `'full'` | str | Native history strategy| full or checkpoint with native CPU/CUDA; Torch also supports full/checkpoint; boundary is unsupported |
| `max_vel` | `keyword-only` | `None` | None or real; m/s | Unrelaxed envelope| Optional fixed unrelaxed-speed envelope, m/s. Required for any trainable material coefficients (including closure parameters) with nonzero PML. Too-small values reject |
| `backend` | `keyword-only` | `'cuda'` | str | Explicit implementation| Explicit `cuda`, `native_cpu` or `torch`; no fallback. CUDA/native require built matching library |
| `mode` | `keyword-only` | `'hao1'` | str | Material mapping| Default `hao1`; explicit `band_fit` or `sls_compat` |
| `frequency_band` | `keyword-only` | `None` | None or positive pair; Hz | Design/fitting band| Required design `(scale,200*scale)` Hz for hao1 (Table 4/Eq.45); required arbitrary positive ordered fitting band for band_fit. A hao1 source working band may be a subset of the design band |
| `n_mechanisms` | `keyword-only` | `None` | None or integer | Mechanism count| Default None selects 5 for hao1 or 3 for band_fit. Explicit hao1 requires 5; band_fit supports integers 2–6; sls_compat requires explicit 1 |
| `fit_tolerance` | `keyword-only` | `0.05` | None or positive real | Relative inverse-Q tolerance| Positive relative inverse-Q tolerance; default dense-grid warning above 0.05; None skips check |
| `strict` | `keyword-only` | `False` | bool | Fit warning/error policy| If true, exceeding fit_tolerance raises rather than warns |
| `material_model` | `keyword-only` | `None` | None or callable | Material provider| Optional advanced callable `(vp_padded,q_padded,dt,f_ref)->(c_rel2,strength,A,B)`; overrides built-in material options. See [provider protocol](#visco-gsls-provider) |
| `checkpoint_interval` | `keyword-only` | `None` | None or positive integer; time steps | Replay segment length| Positive integer segment length, checkpoint only. Native None balances complete-state and scratch history memory using input shape; Torch None is 32 internal steps. Neither uses GPU type |
| `execution` | `keyword-only` | `'eager'` | str | Torch execution | Torch eager/compile; native eager only. |
| `compile_steps` | `keyword-only` | `1` | positive integer | Steps per execution block | Torch positive integer, excluding bool; complete tail; native 1 only. |
<!-- api-doc:visco_gsls:parameters:end -->

(visco-gsls-modes)=
### Modes, Q and reference velocity

| mode | Meaning of q | Mechanisms and frequency band |
|---|---|---|
| `hao1` (default) | Hao & Greenhalgh nominal Q0, not exact Q(f_ref) | `None` resolves to 5; an explicit count must be 5; explicit `(scale,200*scale)` Hz design band required |
| `band_fit` | Approximate constant-Q fitting target, not exact Q(f_ref) | `None` resolves to 3; integers 2–6 supported; explicit positive finite ordered `(fmin,fmax)` Hz required |
| `sls_compat` | Exact complex bulk-modulus Q at f_ref | Explicit `n_mechanisms=1` required; frequency_band is unnecessary and band-fitting options are not used |

In every built-in mode, `vp` is exact phase velocity at f_ref; it cannot be replaced directly by the paper's v0. Positive-frequency complex moduli use `exp(i*omega*t)`; modulus Q is not the wavenumber-attenuation definition. Q=+inf is the exact lossless limit with zero corresponding Q sensitivity.

Hao1 uses five Table 4 weighting mechanisms and uniform frequency scaling from Eq. (45). “First-order” means inverse-Q expansion order, not one mechanism. With `x_l=2*pi*f_ref*t_l` and `S=sum(k_l*x_l**2/(1+x_l**2))`, passivity requires Q0>S; invalid values raise without clipping. A source working band can be a subset of the design band, for example [1,50] Hz; (1,50) is not a supported Hao1 design band. The paper's second-order coupled model is not implemented.

Hao1 and band_fit check sampled relative inverse-Q error by default: above 0.05 warns, `strict=True` raises, and `fit_tolerance=None` skips diagnostics. The tolerance neither changes the defined Hao1 material nor guarantees fit accuracy for arbitrary Q. band_fit is piecewise differentiable, with nondifferentiable active-set/rank changes. sls_compat does not run a band-fit error check; strict is still validated as bool.

Migration from standalone SLS requires both mode and mechanism count; changing only the function name silently selects Hao1. Also use the GSLS preparation signature, with backend before device.

(visco-gsls-physics)=
### Source, coordinates, sampling and input domain

Models have shape `[nz,nx]`, in `[z,x]` order, with both extents at least two. All three models and sources must share dtype/device. CPU/CUDA float32/64 strided tensors and noncontiguous views are supported; contiguous copies retain autograd chains. Model/source dtype and device are not implicitly changed. Density is fixed, positive and finite, with a representable reciprocal. Only Q may contain +inf.

Geometry must explicitly have shapes `[B,S,2]` and `[B,R,2]`, with integer `[z,x]` cell indices inside the original unpadded model. Omitted source dimensions and out-of-domain points are rejected. Coordinates are copied to the model device as int64. Coincident sources add; repeated receivers repeat samples and accumulate cotangents during backward. B, S, R and T must each be at least one.

Sources are second-order forcing in Pa/m²; the pressure update contains `-dt**2*c_rel2*forcing`. The first source update has half weight for zero initial pressure and pressure time derivative. There is no hidden cell-volume factor; a Pa/s pressure-rate source cannot be substituted without conversion.

Records sample pre-update `p[n]` at `t=n*dt`, for T samples. The first sample is exactly zero. The last source sample only affects unreturned `p[T]`, so its records VJP is zero. For T=1, records and model/source gradients are zero. Review sign, amplitude units and time origin when migrating observations or wavelets.

(visco-gsls-gradients)=
### Gradients, padding and PML

- Vp, Q, source and provider closure parameters may be trained independently or together; only first-order propagation derivatives are supported. Density, geometry, dt, spacing, FD and CPML settings are fixed.
- Replicate padding produces `[nz+top+bottom,nx+left+right]`; a positive `(z,x)` pair uses symmetric opposite faces, while top-zero has no top PML extension. Its complete transpose accumulates padded sensitivities into original edges and corners rather than using crop-backward. Model dependence of the source c_rel2 factor remains in the gradient chain.
- The top-zero list `[0,bottom,left,right]` enforces `p(z=0)=0`: pressure is extended oddly and vertical face fields evenly, with divergence the exact negative transpose of the gradient. The same pair applies to relaxation and CPML states. Sources must have z>=1 and surface pressure receivers are zero; every material mode remains available with fixed rho. Other zero-face/no-PML requests are rejected. CPML damping, alpha and the velocity envelope remain fixed numerical setup; no dissipative boundary reconstruction is introduced.
- With gradients enabled and nonzero PML, trainable Vp/Q or any trainable material output (including provider closure parameters) requires explicit fixed max_vel covering `sqrt(c_rel2*(1+strength.sum(0)))`, not merely Vp. Keep the same envelope across optimization steps and paired finite differences.
- Forward-only or source-only gradients can use max_vel=None; the inferred envelope is detached from autograd. Too-small explicit envelopes raise.
- Every forward has independent state/history. Loss, normalization, masks, batching and optimizers belong to the caller. There is no implicit clipping, shot reduction, dt repair or NaN replacement; invalid coefficients and nonfinite inputs/receiver cotangents raise.

(visco-gsls-stability)=
### Time-step guard

Let D be the sum of absolute staggered derivative weights and v_bound cover the unrelaxed speed. The conservative guard is:

```text
dt <= 0.8 / (v_bound * sqrt(max(rho)/min(rho))
             * D * sqrt(1/dz**2 + 1/dx**2))
```

Exceeding the bound raises with the allowed maximum dt. Choose a new dt and regenerate source sampling; there is no implicit resampling or substepping. dt² and inverse spacing must be representable as finite positive normal values in the model dtype. This is not proof of stability for arbitrary strong contrast, extreme Q, heterogeneous CPML or long runs; inspect finite values and boundary reflections in actual experiments. accuracy=2/4/6/8 is spatial order; pressure time discretization is second-order.

(visco-gsls-memory)=
### Full and checkpoint memory

Native CPU/CUDA support full and checkpoint; Torch supports full/checkpoint. boundary is unsupported. With full, checkpoint_interval must be None; an explicit checkpoint interval is a positive number of time steps. Native None uses input shape and requested histories; Torch None uses 32 internal steps. Neither uses GPU-specific tuning.

Native full histories depend on requested primitive gradients: c_rel2 needs force; strength/B need the spatial drive; A needs all M old material memories, for example with trainable relaxation times. Source-only gradients and no_grad observation require none of these volume histories. Fixed-Q Hao1 Vp gradients store T*V entries, joint Vp/Q stores 2*T*V, and a generic provider requesting every primitive gradient stores (M+2)*T*V. All five physical memories still evolve.

The following formulas apply only to native histories, not Torch autograd peak memory. Let N count padded cells, V=B*N, Fx=B*nz_padded*(nx_padded+1), Fz=B*(nz_padded+1)*nx_padded, and M count mechanisms. A complete checkpoint restart includes p, previous p, all material memories, and node/face CPML: S=(4+M)*V+Fx+Fz entries. Segment length K uses approximately `ceil(T/K)*S + K*H*V` tape/replay entries; H counts requested scalar histories, with old material memories contributing M. The default is `K=ceil(sqrt(S*T/(H*V)))` when histories are needed, bounded to [1,T]. These counts, even multiplied by dtype size, are not total peak memory: inputs, outputs, graphs, per-shot gradients, optimizer and workspaces are additional.

Native checkpoint restores complete states, replays each segment forward, and applies its discrete adjoint in reverse order; it does not reverse attenuation. Startup half-weight, global source clock and final partial segments are preserved, with adjoint state carried across segments. When checkpoint replay histories are needed, each time step is recomputed once, exchanging computation for history memory. There is no boundary inverse reconstruction or compression. Examples and memory estimates are not GPU benchmarks.

Torch checkpoint saves complete segment-start states including PML/material memories, replays forward with non-reentrant checkpointing, and differentiates with autograd. It does not invert attenuation or use the native adjoint history above. Short records/small models may not reduce peak memory.

(visco-gsls-example)=
### Minimal call and complete CPU example

This minimal call assumes a matching prepared CUDA library, fixed models, source and geometry on the same CUDA device, and a valid dt. Although frequency_band defaults to None in the signature, default Hao1 requires an explicit band.

<!-- api-doc:visco_gsls:minimal:start -->
```python
records, = starwave.visco_gsls(vp, q, rho, 10.0, 0.0005,
    source_amplitudes=source, source_locations=src, receiver_locations=rec,
    f_ref=18.0, frequency_band=(1.0, 200.0))
```
<!-- api-doc:visco_gsls:minimal:end -->

The complete CPU Torch full example below needs no native library and explicitly supplies every optional parameter. The 1000 amplitude is an example Pa/m² calibration; an energy loss checks gradient wiring, not FWI convergence. Hao1 at Q0=40 may trigger the default 5% fit warning. Retain and diagnose it rather than relaxing tolerance to claim improved material accuracy.

```python
import torch
import starwave

# V17 source; explicit CPU Torch reference, no native build required.
dtype = torch.float64
nz, nx, nt = 12, 16, 80
vp = torch.full((nz, nx), 1800.0, dtype=dtype, requires_grad=True)
q = torch.full((nz, nx), 40.0, dtype=dtype, requires_grad=True)
depth = torch.linspace(0.0, 1.0, nz, dtype=dtype)
rho = (1800.0 + 200.0 * depth[:, None]).expand(nz, nx).contiguous()
dt = 0.0005
t = torch.arange(nt, dtype=dtype) * dt
a = torch.pi * 18.0 * (t - 0.02)
source = (1000.0 * (1.0 - 2.0 * a.square()) * torch.exp(-a.square()))
source = source.reshape(1, 1, nt).requires_grad_()
src = torch.tensor([[[3, 8]]], dtype=torch.int64)
rec = torch.tensor([[[3, 4], [3, 8], [3, 12]]], dtype=torch.int64)

records, = starwave.visco_gsls(
    vp, q, rho, (10.0, 10.0), dt,
    source_amplitudes=source, source_locations=src,
    receiver_locations=rec, f_ref=18.0,
    accuracy=4, pml_width=(4, 4), memory="full",
    max_vel=4000.0, backend="torch", mode="hao1",
    frequency_band=(1.0, 200.0), n_mechanisms=5,
    fit_tolerance=0.05, strict=False,
    material_model=None, checkpoint_interval=None,
    execution="eager", compile_steps=1,
)
assert records.shape == (1, 3, nt)
assert torch.isfinite(records).all()
assert torch.count_nonzero(records[..., 0]) == 0
records.square().mean().backward()
for grad in (vp.grad, q.grad, source.grad):
    assert grad is not None and torch.isfinite(grad).all()
assert torch.count_nonzero(source.grad[..., -1]) == 0
```

Complete native checkpoint call (first build the matching CUDA library and move models/source/geometry to one CUDA device; V17 import initializes the library, or explicitly use native_cpu with CPU tensors):

<!-- api-doc:visco_gsls:full:start -->
```python
records, = starwave.visco_gsls(vp, q, rho, (10.0, 12.0), 0.0005,
    source_amplitudes=source, source_locations=src, receiver_locations=rec,
    f_ref=18.0, accuracy=4, pml_width=(8, 10), memory="checkpoint",
    max_vel=4000.0, backend="cuda", mode="hao1",
    frequency_band=(1.0, 200.0), n_mechanisms=5, fit_tolerance=0.05,
    strict=False, material_model=None, checkpoint_interval=32,
    execution="eager", compile_steps=1)
```
<!-- api-doc:visco_gsls:full:end -->

The example max_vel is configuration-specific; verify the actual unrelaxed speed and CFL. The Torch example can retain backend="torch" and select memory="checkpoint", checkpoint_interval=32. execution="compile" and compile_steps=4 control compilation separately.

(visco-gsls-coefficients)=
### Material and spectrum helpers

Import helpers from `starwave.visco_gsls_coefficients`, not a standalone starwave_gsls namespace. Signature inventory (not executable calls):

```text
from starwave.visco_gsls_coefficients import (
    coefficients, hao1_weighting, hao1_relaxation_times,
    coefficients_from_relaxation, band_relaxation_times,
    quality_spectrum, normalized_modulus, fit_diagnostics,
)

coefficients(vp, q, dt, f_ref, *, mode="hao1", frequency_band=None,
             n_mechanisms=None, fit_tolerance=0.05, strict=False)
hao1_weighting(frequency_band=(1., 200.), *, dtype=torch.float64, device=None)
hao1_relaxation_times(frequency_band=(1., 200.), *, dtype=torch.float64, device=None)
coefficients_from_relaxation(vp, strength, relaxation_times, dt, f_ref)
band_relaxation_times(frequency_band, n_mechanisms=3, *,
                      dtype=torch.float64, device=None)
quality_spectrum(strength, relaxation_times, frequencies)
normalized_modulus(strength, relaxation_times, frequencies)
fit_diagnostics(q, strength, relaxation_times, frequency_band, *,
                sample_count=1001, chunk_size=1024,
                target_convention="band_fit_target")
```


- coefficients returns `(c_rel2,strength,A,B)`: c_rel2 is `[z,x]` in m²/s²; the others are dimensionless `[M,z,x]`. Complete Vp/Q dependencies remain differentiable. This helper accepts scalar Q; propagation still requires model-shaped Q.
- hao1_weighting returns `[5]` times in seconds and dimensionless weights `(times,k)`; hao1_relaxation_times returns only times. The helper default design band (1,200) does not imply an automatic propagation band.
- coefficients_from_relaxation accepts nonnegative strengths `[M,z,x]` and positive finite times `[M]` or `[M,z,x]`, preserves Vp/strength/time gradients and calibrates reference phase speed. band_relaxation_times is for band_fit only; do not combine its times with Hao1 weights.
- quality_spectrum returns actual modulus Q of shape `[frequency,*model_shape]`, with +inf for zero attenuation; normalized_modulus returns complex F=M/M_R with the same shape. Frequencies are Hz and times are seconds.
- fit_diagnostics returns a detached dictionary containing max_relative_inverse_q_error, max_relative_q_error, minimum_strength, all_nonnegative, worst_frequency_hz, worst_model_flat_index, sample_count, frequency_band and target_convention. sample_count≥2 and chunk_size≥1; logarithmic samples including band edges check every model cell, not the entire frequency continuum. For Hao1 use hao1_nominal_Q0; the diagnostic band can be a working subset of its design band.

Run this after the complete CPU example to inspect the [1,50] Hz working sub-band:

```python
from starwave.visco_gsls_coefficients import (
    coefficients, hao1_relaxation_times, quality_spectrum, fit_diagnostics,
)
c_rel2, strength, A, B = coefficients(
    vp, q, dt, 18.0, mode="hao1", frequency_band=(1.0, 200.0),
)
times = hao1_relaxation_times((1.0, 200.0), dtype=vp.dtype, device=vp.device)
frequencies = torch.logspace(0, torch.log10(torch.tensor(50.0)).item(),
                            101, dtype=vp.dtype, device=vp.device)
actual_q = quality_spectrum(strength, times, frequencies)
report = fit_diagnostics(q, strength, times, (1.0, 50.0),
                        target_convention="hao1_nominal_Q0")
print(actual_q.shape, report["max_relative_inverse_q_error"])
```

(visco-gsls-provider)=
### Advanced material_model protocol

`provider(vp_padded,q_padded,dt,f_ref)` runs once per forward after replicate padding. It overrides built-in mode, frequency_band, n_mechanisms, fit_tolerance and strict; those settings neither configure nor diagnose the provider. Original input, coefficient and CFL checks still apply. Capture bands and extra parameters in a closure; the provider owns Q meaning, phase-speed normalization and spectrum accuracy.

Return four strided Tensors matching model dtype/device: c_rel2 `[zp,xp]` and strength/A/B `[M,zp,xp]`, M≥1. All must be finite, with c_rel2>0, strength≥0, B≥0, −1<A≤1 and B≈strength*(1−A), using 32 times dtype epsilon as the consistency tolerance. This is a material interface for the passive trapezoidal GSLS recurrence; arbitrary coefficients cannot replace the wave equation.

The native backward returns VJPs for all four material outputs, which Torch chains through provider parameters. Avoid detach, NumPy, item on learned values or rebuilding them with torch.tensor. Original-grid spatial tensors captured in the closure are not automatically padded; explicitly extend them to the exact same domain and preserve gradients. All trainable material outputs with nonzero PML still require fixed explicit max_vel.

The following extension to the CPU example defines learnable per-mechanism weights and times. Once changed, it is experimental material, no longer the fixed published Hao1 spectrum. The optimizer must preserve Q0>S, and the actual spectrum needs new diagnostics. An external provider change alone needs no recompilation; package files tracked by the fingerprint require rebuilding and restarting.

```python
from starwave.visco_gsls_coefficients import (
    coefficients_from_relaxation, hao1_weighting,
)

base_times, base_k = hao1_weighting(
    (1.0, 200.0), dtype=vp.dtype, device=vp.device,
)
log_weight = torch.nn.Parameter(torch.zeros_like(base_k))
log_time = torch.nn.Parameter(torch.zeros_like(base_times))

def provider(vp_padded, q_padded, dt, f_ref):
    times = base_times * log_time.exp()
    k = base_k * log_weight.exp()
    x = 2 * torch.pi * f_ref * times
    S = (k * x.square() / (1 + x.square())).sum()
    if not bool(torch.all(q_padded > S)):
        raise ValueError("Learned material requires Q0 > S")
    invq = q_padded.reciprocal()
    strength = k[:, None, None] * invq[None] / (1 - S * invq)[None]
    return coefficients_from_relaxation(vp_padded, strength, times, dt, f_ref)

# Pass material_model=provider and a covering fixed max_vel to visco_gsls.
# Add log_weight and log_time to the caller's optimizer when learning them.
```

(visco-gsls-runtime)=
### Native preparation, build and restart

```{py:function} starwave.prepare_visco_gsls(backend: str='native_cpu', device: str | torch.device | None=None) -> dict

Optionally check an explicitly selected device and obtain the existing GSLS library. V17 import already loads and validates available libraries; subsequent calls reuse the ready handle without re-hashing loaded files. Does not compile or propagate.

:param backend: **Default `'native_cpu'`; positional-or-keyword.** Only 'native_cpu' or 'cuda'; 'torch' is rejected.
:type backend: `str`
:param device: **Default `None`; positional-or-keyword.** CUDA requires a valid explicit logical device such as 'cuda:0' or torch.device('cuda:0'), not bare 'cuda'. Ignored for native_cpu.
:type device: `str | torch.device | None`
:returns: Status dictionary; success establishes loading and compatibility only.
:rtype: `dict`
```

```{py:function} starwave.visco_gsls_native_status(backend: str='native_cpu') -> dict

Query library/metadata presence and loaded state without loading or compiling.

:param backend: **Default `'native_cpu'`; positional-or-keyword.** Only 'native_cpu' or 'cuda'; 'torch' is rejected.
:type backend: `str`
:returns: Status dictionary with backend, library_path, library_exists, metadata_exists, library_loaded, cuda_available, gpu_numerically_verified, supported_memory and gradient_order. gpu_numerically_verified is always False; status does not perform GPU acceptance.
:rtype: `dict`
```

Preparation/status default to native_cpu; propagation defaults to cuda. Both prepare parameters are positional-or-keyword. Explicit keywords avoid confusion with the removed standalone SLS argument order.

```python
import starwave

print(starwave.visco_gsls_native_status(backend="native_cpu"))
starwave.prepare_visco_gsls(backend="native_cpu", device=None)
# After a matching CUDA build, with an available logical CUDA device:
starwave.prepare_visco_gsls(backend="cuda", device="cuda:0")
```

Build explicitly from a complete authorized V17 source checkout using an installed Torch-compatible toolchain:

```bash
# Current dev: core/elastic/GSLS/Born; replace 80 with the actual target SM.
python compile_all.py --arch 80
# GSLS CUDA only:
python compile_visco_gsls.py --arch 80
# Separate native CPU library:
python compile_visco_gsls.py --backend cpu
```


In current dev commit `3f2b24daba0fbe6956389baf95dc910405a7ed4a`, compile_all builds/verifies four independent libraries: core, elastic, GSLS and Born. Frozen V17 builds the first three and requires separate `compile_born.py --arch 80`; its immutable archive does not include the later orchestration update. combined_build covers core/elastic only. VRR is separately installed and built. Source-tree GSLS output defaults to native/build/gsls; installed builds use the StarWave cache. An optional STARWAVE_BUILD_DIR uses its gsls subdirectory and must agree between build/runtime; standalone STARWAVE_GSLS_BUILD_DIR is not used. Complete-source execution requires neither pip registration nor export and downloads no compiler; optional editable installation is separate from compilation.

GSLS and core are separate native libraries. GSLS source/library/metadata fingerprints and ABI are checked when the existing library is first loaded during import, then its ready handle is reused without per-propagation source hashing or file polling. Never borrow another tree's library or bypass identity checks. Rebuild and restart Python/Jupyter after updating loaded inputs. Ordinary first propagation or DataParallel needs no manual GSLS preparation; prepare_native and prepare_elastic remain separate optional helpers. No runtime compilation or backend fallback occurs. Root compile_all does not rebuild an optional retained GSLS CPU library; rebuild that separately when stale.

CUDA numerical correctness, actual multi-GPU execution, long FWI and performance require separate target-device validation. Library presence, compilation or successful preparation do not establish those results. This section supplies no new GPU or benchmark acceptance evidence.

(visco-gsls-references)=
### Scope and references

Unsupported: 3D, rho gradients, trainable PML/dt, boundary reconstruction, initial-state inputs, final-state returns, arbitrary-face free surfaces/topography, compression, illumination, AMP, CUDA graphs and supported higher-order propagation derivatives. Incidental higher-order Torch-graph behavior is not a contract.

- Hao & Greenhalgh (2021), “Nearly constant Q models of the generalized standard linear solid type and the corresponding wave equations”, GEOPHYSICS 86(4), T239–T260. [DOI](https://doi.org/10.1190/geo2020-0548.1).

The paper describes the physical model. This API follows the source sign, reference-phase normalization, discrete sampling, PML, padding and adjoint contracts above; bitwise agreement with other implementations is not claimed. See [inversion](inversion/fwi.md) for general objective organization. Historical SLS experiments are not current GSLS execution evidence.

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

Optionally validates explicit logical CUDA devices and returns the existing core native library status. V17 import already initializes available core libraries before returning; this helper is not required before ordinary first propagation or DataParallel. Does not compile.

:param device_ids: Required. Nonempty, duplicate-free collection of nonnegative visible logical CUDA device IDs; Booleans are rejected. Use the device-visibility mapping of the current process.
:type device_ids: `list[int] | tuple[int, ...]`
:returns: Dictionary containing native status, `selected_device_ids`, and preparation messages. Successful initialization is not a GPU numerical test.
:rtype: `dict`
```

(native-notes)=
## Native runtime notes

After explicit compilation, ordinary V17 `import starwave` initializes available core, elastic, Born and GSLS libraries plus an importable independent VRR extension. Shared dispatch is ready before the first parallel call; no serial warm-up or mandatory `prepare_*` step is needed. Import does not compile, initialize CUDA, select devices, allocate wavefields or propagate. Missing unbuilt libraries are skipped for supported pure-Torch use; an existing invalid build fails instead of silently falling back. Configure build paths before import and restart Python/Jupyter after rebuilding loaded code. Ready handles are reused without per-propagation source hashing or library polling.

Use `native_status()` for diagnostics and optional `prepare_native()` to check explicit logical CUDA devices. Neither function compiles or replaces GPU numerical acceptance; device `0` in the example must be visible. Core, elastic, GSLS and VRR retain their initial-load identity checks. Born performs full hashes/symbol auditing only during explicit builds/development audits, then uses ordinary one-time runtime loading.

(native-examples)=
## Native runtime examples

With the reviewed V17 source, a matching explicit native build and compatible PyTorch, query status after import. The optional preparation call requires an available CUDA device. The pending 8.0.0 wheel must be independently verified; published 7.1.0 retains its older lifecycle.

```python
import starwave

status = starwave.native_status()
print(status)
prepared = starwave.prepare_native([0])  # Optional device validation.
```

(prepare-elastic)=
## starwave.prepare_elastic

```{py:function} starwave.prepare_elastic(device_ids: list[int] | tuple[int, ...] | None=None) -> dict

Optionally obtain the separate elastic library and validate explicit logical devices. V17 import already initializes available elastic bindings; this is not a mandatory CUDA/DataParallel preload step. Does not compile or execute propagation tests.

:param device_ids: **Default `None`.** None loads the elastic library without selecting or preparing a CUDA device list, suitable for CPU use; a nonempty list/tuple selects distinct, nonnegative visible logical CUDA integer IDs (bool is rejected). CUDA requests validate the devices and build capability.
:type device_ids: `list[int] | tuple[int, ...] | None`
:returns: Dictionary with elastic-library status, selected_device_ids and a preparation message; successful loading is not numerical or DataParallel acceptance.
:rtype: `dict`
```

`starwave.elastic_native_status()` reads elastic-library status. The existing `prepare_native()` serves scalar/VRZ/VTI; elastic-only use does not require preparing those propagators.

(other-exports)=
## Other exported names and scope

`ScalarIllumination`, `IlluminationFields`, and `precondition_gradient` are verified exported names related to scalar illumination. This edition does not yet provide a complete lifecycle tutorial for them; the propagator pages explain the usage boundaries of the `illumination` parameter.

StarWave V17 has no public `starwave.Scalar` wrapper class; `starwave.ScalarBorn` is the distinct Born module documented above. The tutorials define their own Module wrapper. Do not assume another library’s classes exist in StarWave, or add elastic state, `nt`, or storage options to scalar/VRZ/VTI calls.

{ref}`genindex`
