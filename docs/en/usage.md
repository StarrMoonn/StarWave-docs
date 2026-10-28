# Usage

New: [V21 elastic VRR API and runnable CPU example](examples/elastic-vrr.md). This addition requires matching authorized V21 source; older version/package statements below remain historical baseline information and do not establish current PyPI availability or include elastic VRR. Full reference: [Elastic VRR Function](#elastic-vrr-api) and [parameter conversion](#elastic-vrr-parameters).

New: [Maxwell / GPR API](#gpr) covers voxel-based 2D TMz/TEz and 3D electromagnetic propagation with full/checkpoint storage. Use the matching current source, not the frozen V19 package baseline.

GSLS: [visco_gsls](#visco-gsls). VRR: [starwave.vrr](#vrr). Both are included in the V19 root package.

```{container} sw-page-toc

**On this page**

- [Scalar Function](#scalar)
- [Born Function](#born)
- [VRZ Function](#vrz)
- [VRR Function](#vrr)
- [VTI Function](#vti)
- [qP VTI / TTI Function](#tti)
- [Elastic Function](#elastic)
- [GSLS Function](#visco-gsls)
- [Elastic VRR Function](#elastic-vrr-api)
- [Maxwell / GPR Function](#gpr)
- [Material conversions](#elastic-conversions)
- [starwave.native_status](#native-status)
- [starwave.prepare_native](#prepare-native)
- [starwave.prepare_elastic](#prepare-elastic)
- [Other exported names and scope](#other-exports)
```

This reference targets frozen V19 source `0.1.0.dev19`, commit `46d32877e3336d531520a696d5f2e074e6f4ef74`, published as [Release V19](https://github.com/StarrMoonn/StarWave/releases/tag/V19). Except for the separately labelled V21 elastic VRR, V22 qP TTI and current Maxwell/GPR additions, signatures below are checked against that source, including the 23-parameter Scalar API and integrated `starwave.vrr`. These frozen V19 API contracts are a historical compatibility baseline, not a statement of current package availability. Use the matching complete source checkout for the separately labelled additions on this page; do not infer current PyPI availability from historical wheel records. Source, distribution and device-validation evidence remain separate. See [Status](status.md) and [installation](installation.md).

See [Release Notes](release-notes.md) for 6.0.0 internal storage and PML-transpose maintenance, and [Scalar](modeling/scalar.md) for 2D storage changes. These introduce no new public parameters; source upgrades require rebuilt matching native libraries.

The page organization follows the Sphinx Python API style of the [official Deepwave Usage documentation](https://ausargeo.com/deepwave/usage). All descriptions are newly written from StarWave’s actual contracts. The libraries’ parameter sets, source units, return structures, and differentiability scopes are not interchangeable.

(propagators)=
## Propagators at a glance

- {py:func}`starwave.scalar`: `2D/3D` scalar acoustics; velocity model `v`; default records tuple, optional terminal physical wavefield.

- {py:func}`starwave.scalar_born`: `1D/2D/3D` constant-density acoustic Born; velocity perturbation δv, background/scattered records and terminal states. Included in V19.

- {py:func}`starwave.vrz`: `2D` variable-density acoustics; `v` plus exactly one `impedance` / `density` parameterization.

- {py:func}`starwave.vrr`: `2D` acoustic vector reflectivity; independent `v, rx, rz` fields. Integrated in the root package.

- {py:func}`starwave.vti`: `2D/3D` acoustic VTI; `vp, epsilon, delta, rho`; returns recordings in the selected component order.

- {py:func}`starwave.tti`: `2D` local approximate pseudo-acoustic qP VTI/TTI; `vp, epsilon, delta, theta`; scalar forcing, full/checkpoint. Included in the [V22 complete-source release](https://github.com/StarrMoonn/StarWave/releases/tag/V22).

- {py:func}`starwave.elastic`: `2D/3D` isotropic elasticity; `lamb, mu, buoyancy`; complete final states plus p/velocity records.

- {py:func}`starwave.visco_gsls`: 2D generalized-standard-linear-solid viscoacoustics, fixed variable density and native full/checkpoint; see the [GSLS Function](#visco-gsls) section below.

- {py:func}`starwave.gpr` / {py:func}`starwave.maxwell`: voxel-based `2D TMz/TEz` and six-component `3D` Maxwell; epsilon_r, sigma and prescribed mu_r; complete restart state plus one E/H receiver component.

Notation: `B` is the number of shots, `S` the sources per shot, `R` the receivers per shot, `T` the number of user time samples, and `D` the number of spatial dimensions. Sources and receivers use integer grid indices in the physical model, not coordinates in meters, and do not include PML offsets.

(scalar)=
## Scalar Function

V19 retains the 23-parameter Scalar contract introduced in V18: arbitrary zero-face subsets (equal positive widths), trainable 2D/3D sources, optional differentiable terminal fields, and read-only diagnostic callbacks. V18 / 9.0.0 introduced this API; V17 / unpublished 8.0.0 and V16 / 7.1.0 retain the historical 20-parameter contract. Native execution requires matching rebuilt libraries.

```{py:function} starwave.scalar(v: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, accuracy: int=8, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float | int=0.0, time_pad_frac: float | int=0.0, time_taper: bool=False, illumination: ScalarIllumination | None=None, backend: str | None=None, execution: str='eager', checkpoint_interval: int | None=None, compile_steps: int=1, return_wavefield: bool=False, forward_callback: Callable | None=None, callback_frequency: int=1) -> tuple[torch.Tensor, ...]

Propagates scalar acoustic waves on a 2D or 3D grid selected by `v.ndim`, with independent shots. Both dimensions support first-order velocity and source-waveform gradients. Coordinates and numerical settings are not differentiable. Each call starts with fresh state; optionally return the differentiable terminal physical wavefield, not a complete restart state.

:param v: **Required; positional or keyword.** Two-dimensional `[N0,N1]` or three-dimensional `[N0,N1,N2]` model, with at least 2 cells per dimension; float32 (native CUDA; Torch CPU/CUDA), typically in m/s. `v.ndim` selects the dimension. Coordinates `[i,j]` / `[i,j,k]` directly index `v[i,j]` / `v[i,j,k]`; the 3D examples use `[nx,ny,nz]`, with no automatic physical-axis permutation. All values must be finite. Signed and zero values are accepted without guaranteeing physical validity. Model-axis transformations preserve autograd; `requires_grad=True` requests a velocity gradient.
:type v: `torch.Tensor`
:param grid_spacing: **Required; positional or keyword.** Positive, finite spacing, typically in m. In 2D, accepts one value or two equal values. In 3D, accepts one value or three values in model-axis order (for example `[dx,dy,dz]`); unequal spacings are supported. Booleans, zero/negative values, and mismatched sequence lengths are rejected.
:type grid_spacing: `float | int | list[float] | tuple[float, ...]`
:param dt: **Required; positional or keyword.** Positive, finite sampling interval of the user input/output, typically in s. Internally, the CFL condition may require a smaller time step; the returned sampling interval remains this value. Tensor and Boolean inputs are not accepted, and this parameter is not differentiated.
:type dt: `float | int`
:param source_amplitudes: **Required; keyword-only.** Real tensor `[B,S,T]` with nonempty dimensions. Values must be finite and representable as float32; conversion to the model device and float32 preserves the supported gradient chain. Both 2D and 3D support first-order source gradients, alone or together with velocity gradients. This is normalized forcing `f`, not a per-step pressure increment; its unit is Pa/m² if the wavefield is in Pa and length in m. `[T]` and `[B,T]` are not accepted: explicitly expand shot/source dimensions for a shared waveform.
:type source_amplitudes: `torch.Tensor`
:param source_locations: **Required; keyword-only.** Fixed integer tensor `[B,S,D]`, where `D=v.ndim`; `[B,D]` is also accepted for a single source per shot. `torch.long` is recommended. Coordinates directly index the physical grid in input-model axis order; 3D `[i,j,k]` means `v[i,j,k]`. No PML offsets or out-of-bounds values are allowed. Each source has a waveform, and coincident sources within one shot add their increments.
:type source_locations: `torch.Tensor`
:param receiver_locations: **Required; keyword-only.** Fixed integer tensor `[B,R,D]`, where `D=v.ndim` and `R>=0`; `torch.long` is recommended. The shot count matches the sources. Coordinates index the physical grid in input-model axis order. Duplicate positions are allowed. This keyword remains required; an empty integer tensor `[B,0,D]` requests no receivers and produces `[B,0,T]` records. No dummy receiver is needed for a terminal-only loss.
:type receiver_locations: `torch.Tensor`
:param accuracy: **Default `8`; keyword-only.** Spatial finite-difference order, one of `2,4,6,8`, not an error tolerance. Affects stencil extent, computational cost, and the minimum boundary buffer. Booleans and other orders are rejected.
:type accuracy: `int`
:param pml_freq: **Default `25.0`; keyword-only.** Positive, finite frequency used to configure the PML, typically in Hz; also used for spatial-sampling diagnostics. It does not generate a waveform or automatically measure the source frequency. Choose it for the actual experiment.
:type pml_freq: `float | int`
:param pml_width: **Default `20`; keyword-only.** Nonnegative integer thickness in cells or four/six nonnegative face widths for 2D/3D. Any subset of faces may be zero, including scalar 0/all-zero; all positive widths must still equal one A. Order is `[x-low,x-high,z-low,z-high]` / `[x-low,x-high,y-low,y-high,z-low,z-high]`. Zero faces use fixed-zero FD ghosts without PML/buffer; physical boundary samples remain active. Negative/noninteger/bool values and unequal positive widths are rejected. This scalar-only domain does not broaden VRZ/VTI.
:type pml_width: `int | list[int] | tuple[int, ...]`
:param boundary_buffer: **Default `5`; keyword-only.** Nonnegative buffer in spatial grid cells, not a number of time steps or a checkpoint interval. Boundary mode requires at least `accuracy // 2 + 1`, which is 5 at eighth order; full mode allows 0. All absorbing sides use total padding `pml_width + boundary_buffer + accuracy // 2`; each zero-width face uses only `accuracy // 2` halo cells, without PML or buffer on that face.
:type boundary_buffer: `int`
:param memory: **Default `'boundary'`; keyword-only.** Backpropagation-history strategy. `"boundary"` stores pressure strips and two terminal pressure fields for reconstruction; the six 3D faces each have width `M=accuracy//2` (Radius-M). In 3D, `"full"` stores unscaled `Lap(u)` history over the whole padded volume at every internal step and is intended for small comparisons. Source-only gradients in either dimension also retain the selected history; a forward with neither velocity nor source gradients retains none. Full mode can exhaust GPU memory. Native has no automatic fallback, CPU/disk offload or checkpoint. Torch separately supports full/checkpoint (no boundary), with autograd graphs or complete-state checkpoints rather than this native Lap(u) layout.
:type memory: `str`
:param max_vel: **Default `None`; keyword-only.** CFL/PML velocity envelope, typically in m/s. With `None`, each call replans from the current model’s `max(abs(v))`. An explicit value must be positive, finite, and cover that maximum; it is not a clipping limit. An all-zero model requires an explicit positive value. Extrema, time substeps, and PML configuration are not differentiated.
:type max_vel: `float | int | None`
:param freq_taper_frac: **Default `0.0`; keyword-only.** Finite fraction in `[0,1]` controlling the cosine taper of high-frequency FFT bins during temporal resampling; the bin count is obtained by truncating the proportional count to an integer. During resampling, the last positive rFFT bin is suppressed even when this fraction is 0, including for odd lengths; preservation of every frequency is not guaranteed. This setting does not change the signal when the internal resampling ratio is 1.
:type freq_taper_frac: `float | int`
:param time_pad_frac: **Default `0.0`; keyword-only.** Finite fraction in `[0,1]`. Appends `int(time_pad_frac*T)` user samples of zeros before resampling and removes them afterwards. Does not increase the public recording length or the actual propagation duration. It does not change the signal when the internal resampling ratio is 1.
:type time_pad_frac: `float | int`
:param time_taper: **Default `False`; keyword-only.** Whether to apply a nonperiodic Hann window during temporal resampling: after upsampling and before downsampling. Changes both the signal and its backpropagation transpose. Accepts only Booleans, not the integers 0/1. It does not change the signal when the internal resampling ratio is 1.
:type time_taper: `bool`
:param illumination: **Default `None`; keyword-only.** Optional fresh `ScalarIllumination` collector in 2D, requiring enabled gradient tracking and a trainable model. Collects detached source, receiver, and geometric-product statistics; it is sealed after forward and reduced after one backward. In 3D only `None` is accepted; any other value raises `NotImplementedError`. `None` allocates no illumination buffers. These statistics are not an exact Hessian and do not automatically precondition gradients. Must be None for Torch or any scalar zero-width face.
:type illumination: `ScalarIllumination | None`
:param backend: **Default `None`; keyword-only.** None preserves native behavior and never automatically selects Torch. `'cuda'` explicitly requires CUDA models; `'torch'` selects CPU/CUDA tensor propagation and autograd. Device follows inputs without implicit model transfer. `'cpu'` is not a backend selector. MPS is supported only for Scalar2D accuracy=4 float32; Scalar3D and other scalar MPS combinations are rejected. Explicitly select full/checkpoint for scalar/VRZ/VTI Torch calls.
:type backend: `str | None`
:param execution: **Default `'eager'`; keyword-only.** Torch accepts `'eager'` or `'compile'`. Compile uses PyTorch Inductor/AOTAutograd and raises on failure without eager fallback. Native propagation only accepts eager.
:type execution: `str`
:param checkpoint_interval: **Default `None`; keyword-only.** Torch checkpoint segment length in internal time steps: a positive integer, excluding bool; None resolves to 32. Keep None for full or native propagation. Longer segments increase the local replay graph and need not reduce peak memory.
:type checkpoint_interval: `int | None`
:param compile_steps: **Default `1`; keyword-only.** Positive integer, excluding bool; consecutive internal steps per Torch execution block. Try 4 for compile; the final partial block is retained. Does not change dt, nt, shots or loss. Native propagation accepts only 1.
:type compile_steps: `int`
:param return_wavefield: **Default `False`; keyword-only.** Boolean. False preserves `(records,)`; True returns `(terminal_physical_wavefield, records)`, keeping `[-1]` as records. The differentiable field is `[B,*v.shape]`, in public model-axis order, excluding PML/buffer/FD ghosts. It is after the final internal update and source injection, at `T*dt`; previous time level and CPML memories are not returned.
:type return_wavefield: `bool`
:param forward_callback: **Default `None`; keyword-only.** Callable receiving a ScalarCallbackState after completed forward intervals and the final update. Its physical wavefield is an independent detached read-only snapshot; mutation during the call raises. No callback occurs at initial time or during Torch checkpoint backward replay. Use the returned terminal field, not callback snapshots, for differentiable losses. See the clock below.
:type forward_callback: `Callable | None`
:param callback_frequency: **Default `1`; keyword-only.** Positive integer, excluding bool, counting user samples. Internal cadence is `CFL_ratio * callback_frequency`; the last update always triggers a callback even for a partial interval. Validated even when forward_callback is None.
:type callback_frequency: `int`
:returns: **Default `(receiver_amplitudes,)`; with `return_wavefield=True`, `(terminal_physical_wavefield, receiver_amplitudes)`.** Records are contiguous float32 `[B,R,T]` on the model device, at nominal times `0, dt, ..., (T-1)*dt`; `[-1]` always selects them. The optional differentiable terminal field is `[B,*v.shape]` at `internal_nt*internal_dt = T*dt`, after the last source update. Its time differs from the last record sample; it excludes PML/buffer/ghosts and cannot restart propagation by itself.
:rtype: `tuple[torch.Tensor, ...]`
```

**Torch path:** set `backend="torch"` on the same main entry point and explicitly choose `memory="full"` or `"checkpoint"` for CPU/CUDA without requiring a native build. Import still initializes available native libraries. Native history-layout/stream restrictions below describe the native path. See the [PyTorch backend](pytorch-backend.md) for all options and the memory table.

(scalar-details)=
### Sources and autograd

The public equation convention is `u_tt = v² (Lap(u) - f)`. The user may train `f` in both 2D and 3D; the internal source increment is `-v_source² * internal_dt² * U(f)`, where `U` is internal upsampling. Do not multiply by `-v² dt²` again externally. The velocity-scaling chain at source locations is retained in the model gradient.

Post-step recordings are aligned to the user clock before downsampling by prepending zero and dropping the final step. The transposes of temporal resampling and shifting remain in autograd; noncausal FFT filtering can cause ringing in the initial user samples.

Model gradients are conditional on a fixed extended model, CFL configuration, and PML. When replicate padding at the physical edges is regenerated, the current returned gradient does not include the transpose accumulation of the complete extension chain. It must not be treated as the total derivative of the entire boundary-rebuilding process; switching to full mode does not remove this limitation.

Each zero-width face has `M=accuracy//2` fixed-zero exterior ghost cells and no PML/buffer. Physical edge samples remain active; keep original coordinates. This is a truncated centered stencil, not odd reflection: at second order the effective zero plane lies one grid interval outside the physical edge. Higher interior order does not establish an odd-image closure or surface convergence order. Native full uses full-grid history when zero faces are present; boundary still saves/reconstructs required face strips without a full fallback. Any zero face rejects illumination. Native CUDA full/boundary and Torch CPU/CUDA full/checkpoint eager/compile support the observation options. MPS eligibility remains Scalar2D float32, accuracy=4; no new MPS device acceptance is claimed.

Callback state exposes `wavefield`, `step` (integer completed user steps), `user_step`, `internal_step`, `internal_dt`, `step_ratio`, `dt` (alias of internal_dt), and `time=internal_step*internal_dt`. Use `state.time` for seconds: `state.step*state.dt` is wrong when the CFL ratio exceeds one. For ratio 2 and frequency 3, callbacks occur at user steps 3,6,... / internal steps 6,12,..., plus the final update. Snapshots are independent clones and do not carry loss gradients. Callback closures used with DataParallel must handle their own concurrent observation bookkeeping.

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


V19 CPU example: terminal-only loss, trainable 2D source, all-zero faces, empty receivers and checkpoint-safe observation. It is a small gradient-wiring check, not target-wavefield convergence or GPU acceptance.

<!-- scalar-dev:example:start -->
```python
import torch
import starwave

v = torch.full((12, 10), 1500.0, dtype=torch.float32, requires_grad=True)
source = torch.zeros((1, 1, 7), dtype=v.dtype)
source[0, 0, 1] = 1.0
source.requires_grad_()
src = torch.tensor([[[6, 5]]], dtype=torch.long)
rec = torch.empty((1, 0, 2), dtype=torch.long)
steps = []

def observe(state):
    assert state.wavefield.shape == (1, 12, 10)
    assert not state.wavefield.requires_grad
    steps.append((state.step, state.internal_step, state.time))

terminal, records = starwave.scalar(
    v, 10.0, 0.004,
    source_amplitudes=source, source_locations=src,
    receiver_locations=rec, accuracy=4, pml_width=0,
    pml_freq=15.0, max_vel=1600.0, backend="torch",
    memory="checkpoint", checkpoint_interval=4,
    return_wavefield=True, forward_callback=observe, callback_frequency=3,
)
assert terminal.shape == (1, 12, 10) and records.shape == (1, 0, 7)
assert [step for step, _, _ in steps] == [3, 6, 7]
assert torch.isfinite(terminal).all()
terminal.square().mean().backward()
assert len(steps) == 3  # No callbacks during checkpoint backward replay.
for gradient in (v.grad, source.grad):
    assert gradient is not None and torch.isfinite(gradient).all()
```
<!-- scalar-dev:example:end -->

The frozen V19 source contains the full user target-wavefield example at `notebook/Scalar/Scalar2D_TargetWavefield_CUDA.ipynb`, with CUDA `.py` export and matching `notebook/Scalar/Scalar2D_TargetWavefield_TorchCompile.ipynb` / `notebook/Scalar/Scalar2D_TargetWavefield_TorchCompile.py` pair. Open it from that source checkout; this public manual does not redistribute the private notebook, report assets or datasets. It optimizes sources against a terminal-field target and keeps callback snapshots diagnostic. User-supplied RTX 4060/A30 results describe the reviewed feature branch; they are not cloud GPU reruns or validation of a released wheel.

(scalar-time-sampling)=
### CFL and internal temporal resampling

Scalar uses the conservative coefficient 0.6, every model-axis spacing, and `max(abs(v))` (or the checked `max_vel`) to compute a time-step bound. The internal ratio `r` is a positive integer satisfying that bound, with `internal_dt=dt/r` and `internal_nt=T*r`. Unequal 3D spacings participate axis by axis. `accuracy` does not enable a different public CFL parameter. Time planning, the velocity envelope, and PML coefficients are fixed setup values outside autograd.

Sources are FFT-upsampled before source-location velocity scaling. Records are time-aligned before downsampling to `[B,R,T]`. Backward uses the actual transposes of these operations, not simple repetition or decimation. At ratio 1, valid taper/padding settings do not alter the signal. Time substeps cannot repair spatial dispersion.

(scalar-memory)=
### Native 3D memory: full and Radius-M boundary

`M=accuracy//2` is the pressure-Laplacian stencil radius, distinct from `boundary_buffer` and `pml_width`. Boundary mode stores six pressure faces of exactly M cells plus two terminal pressure fields, without a full time-volume history or automatic fallback to full. Full mode stores unscaled `Lap(u)` volume history at every internal step. Both modes retain propagation/adjoint workspaces; 3D CPML memory uses directional slabs. Source-only gradients also retain the selected history. Memory grows with internal time steps and shot count, so boundary mode does not guarantee production-scale 3D fits a GPU. See the exact payload formulas in {ref}`Scalar3D reconstruction <reconstruction-scalar3d>`.

(scalar-3d-example)=
### Runnable 3D call: first-order velocity and source gradients

With the V19 scalar source, its matching rebuilt native library, and visible logical CUDA device 0 available, the following is a standalone small example. The model uses `[x,y,z]`, spacing is `[dx,dy,dz]`, and coordinates are grid indices. The loss only checks gradient wiring. Syntax and interface have been checked; this documentation update did not execute the GPU example, so these assertions are not numerical acceptance results.

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
    return_wavefield=False, forward_callback=None, callback_frequency=1,
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

- Native supports CUDA FP32 on the default stream only; each forward permits one first-order backward pass. Source gradients are supported at first order in both 2D and 3D. Higher-order derivatives, AMP, CUDA graphs, custom streams, CPU propagation, and public initial/restart states are unsupported; the optional terminal physical field is not a full state.
- CFL time substeps cannot compensate for inadequate spatial sampling. Spatial-resolution warnings are diagnostics, not certificates of accuracy or stability.
- For DataParallel, place the complete model in a Module and split acquisition inputs only along the shot dimension; see [Multi-GPU introduction](inversion/dataparallel.md).
- Source/API and bounded CPU checks are distinct from the user-supplied feature-branch GPU report. This documentation update does not rerun target-GPU propagation or establish merged-dev/wheel performance and FWI acceptance.

(born)=
## Born Function

V19 includes this main-package API. This constant-density acoustic Born operator is derived from [Deepwave 0.0.27](https://github.com/ar4/deepwave/tree/dc2f8af71e1fb67676c50a91a22bdd059bac8772). `scatter` is velocity perturbation δv, in the units of `v`, not relative squared velocity or squared slowness. Its background operator differs from `starwave.scalar`; use the returned background records for compatible observations. No installed Deepwave package is required.

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
# Complete authorized V19 source; use the actual GPU SM.
python compile_born.py --arch 80
# CPU-only native full, alternatively:
python compile_born.py --cpu-only
```

Born is in the main package. V19 `python compile_all.py --arch 80` builds core, elastic, GSLS, Born and VRR together. `python compile_born.py --cpu-only` remains available for native CPU full. Explicit Torch CPU/CUDA needs no Born native build.

After explicit compilation, V19 `import starwave` loads available core/elastic/Born/GSLS libraries and completes shared-dispatch bindings before returning. Optional Born boundary-entry signatures are bound by their calling path before invocation. Integrated VRR is prepared too. Ordinary first parallel propagation needs no serial warm-up or mandatory manual prepare call. Import does not compile, initialize CUDA, select devices, allocate wavefields or propagate. Missing builds are skipped for supported pure-Torch use; an existing invalid build is not silently replaced. Configure build paths before import, rebuild after updates and restart Python/Jupyter. Runtime does not monitor source/library changes or re-hash sources for each propagation. Optional prepare helpers remain available for explicit device checks.

(born-scope)=
### Version and validation scope

This section describes frozen V19. Source release, compilation, package publication, clean installation and CPU/GPU/multi-GPU numerical acceptance are distinct evidence. Documentation checks do not certify hardware numerics. Native higher-order AD, MPS, AMP and performance guarantees are outside this contract.

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
:param pml_width: **Default `20`; keyword-only.** Positive integer thickness in cells or four equal positive widths, in `[x-low,x-high,z-low,z-high]` order. V19 accepts `[A,A,0,A]` for a flat top, where A is one positive integer width on the remaining faces. Other zero faces, all-zero, unequal remaining widths, negative/noninteger/bool values are rejected. The first physical row remains active; see below.
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

<span id="starwave_vrr.vrr"></span>

```{py:function} starwave.vrr(v: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, rx: torch.Tensor, rz: torch.Tensor, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, accuracy: int=8, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float=0.0, time_pad_frac: float=0.0, time_taper: bool=False, backend: str='cuda', execution: str='eager', checkpoint_interval: int | None=None, compile_steps: int=1) -> tuple[torch.Tensor]


VRR is integrated in V19 as `starwave.vrr` and `starwave.vrr_reference`. Install the root StarWave package; no separate `starwave-vrr` distribution is required. This public manual contains API contracts, not the scientific implementation.

:param v: **Required.** Positive finite velocity [nx,nz], m/s. Each model axis has at least 2 cells.
:type v: `torch.Tensor`
:param grid_spacing: **Required.** Positive uniform spacing: scalar or equal pair, metres.
:type grid_spacing: `float | int | list[float] | tuple[float, ...]`
:param dt: **Required.** Positive user sampling interval, seconds.
:type dt: `float | int`
:param rx: **Required.** Keyword-only. Finite signed x reflectivity-vector field, same shape/device/dtype as v; 1/m.
:type rx: `torch.Tensor`
:param rz: **Required.** Keyword-only. Finite signed z reflectivity-vector field, same shape/device/dtype as v; 1/m.
:type rz: `torch.Tensor`
:param source_amplitudes: **Required.** Keyword-only. Forcing [B,S,T]; explicitly expand shared wavelets across shots/sources. All B/S/T are positive; dtype and device must already match v exactly. No implicit transfer/casting.
:type source_amplitudes: `torch.Tensor`
:param source_locations: **Required.** Keyword-only. Integer [B,S,2], or [B,2] for one source per shot. Use int32/int64 already on the model device.
:type source_locations: `torch.Tensor`
:param receiver_locations: **Required.** Keyword-only. Integer [B,R,2]; geometry indexes unpadded physical models. R must be at least 1; int32/int64 on the model device. Empty receiver geometry is rejected.
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
:param backend: **Default `"cuda"`.** Keyword-only. cuda or torch; no automatic backend fallback. None and native_cpu are rejected.
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

(vrr-reference)=
### starwave.vrr_reference

```{py:function} starwave.vrr_reference(v: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, rx: torch.Tensor, rz: torch.Tensor, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, accuracy: int=8, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='full', max_vel: float | int | None=None, freq_taper_frac: float=0.0, time_pad_frac: float=0.0, time_taper: bool=False) -> tuple[torch.Tensor]

Explicit tensor reference for CPU/CUDA float32/float64. All four differentiable inputs, geometry, PML, source and return conventions match `vrr`; only full memory is supported. There are no backend, execution, checkpoint or compilation selectors. This function never acts as an automatic fallback.

:param v: **Required.** Positive finite velocity [nx,nz], m/s. Each model axis has at least 2 cells.
:type v: `torch.Tensor`
:param grid_spacing: **Required.** Positive uniform spacing: scalar or equal pair, metres.
:type grid_spacing: `float | int | list[float] | tuple[float, ...]`
:param dt: **Required.** Positive user sampling interval, seconds.
:type dt: `float | int`
:param rx: **Required.** Keyword-only. Finite signed x reflectivity-vector field, same shape/device/dtype as v; 1/m.
:type rx: `torch.Tensor`
:param rz: **Required.** Keyword-only. Finite signed z reflectivity-vector field, same shape/device/dtype as v; 1/m.
:type rz: `torch.Tensor`
:param source_amplitudes: **Required.** Keyword-only. Forcing [B,S,T]; explicitly expand shared wavelets across shots/sources. All B/S/T are positive; dtype and device must already match v exactly. No implicit transfer/casting.
:type source_amplitudes: `torch.Tensor`
:param source_locations: **Required.** Keyword-only. Integer [B,S,2], or [B,2] for one source per shot. Use int32/int64 already on the model device.
:type source_locations: `torch.Tensor`
:param receiver_locations: **Required.** Keyword-only. Integer [B,R,2]; geometry indexes unpadded physical models. R must be at least 1; int32/int64 on the model device. Empty receiver geometry is rejected.
:type receiver_locations: `torch.Tensor`
:param accuracy: **Default `8`.** Keyword-only. Interior spatial finite-difference order: 2, 4, 6 or 8.
:type accuracy: `int`
:param pml_freq: **Default `25.0`.** Keyword-only. Positive CPML reference frequency, Hz.
:type pml_freq: `float | int`
:param pml_width: **Default `20`.** Keyword-only. Positive scalar, equal positive four-face widths, or [w,w,0,w] with positive integer w.
:type pml_width: `int | list[int] | tuple[int, ...]`
:param boundary_buffer: **Default `5`.** Keyword-only. Spatial cells; at least accuracy//2+1, including full mode.
:type boundary_buffer: `int`
:param memory: **Default `'full'`.** Keyword-only. Only full history is accepted.
:type memory: `str`
:param max_vel: **Default `None`.** Keyword-only. Detached current velocity maximum when omitted; explicit positive bound must cover v.
:type max_vel: `float | int | None`
:param freq_taper_frac: **Default `0.0`.** Keyword-only. FFT resampling frequency-taper fraction.
:type freq_taper_frac: `float`
:param time_pad_frac: **Default `0.0`.** Keyword-only. FFT resampling time-padding fraction.
:type time_pad_frac: `float`
:param time_taper: **Default `False`.** Keyword-only. Optional time taper during resampling.
:type time_taper: `bool`
:returns: Single-element `(record,)` tuple; record shape `[B,R,T]`. Use `result[-1]`; no final states or illumination.
:rtype: tuple[torch.Tensor]
```

(vrr-install)=
### Install and build

From complete V19 source, install the root package and explicitly build for the actual GPU SM:

```bash
python -m pip install --no-deps --no-build-isolation -e .
python compile_all.py --arch 80
```

The unified builder includes core, elastic, Born, GSLS and VRR. `backend="torch"` requires no native build. Import and propagation never compile implicitly. Rebuild matching source and restart Python/Jupyter after native changes. See [installation](installation.md) for the shared build configuration.

(vrr-backends)=
### Backends and gradients

| Backend | Device / dtype | Memory | Execution |
|---|---|---|---|
| `cuda` (default) | CUDA float32 | `full`, `boundary` (default) | `eager` selector; native execution |
| `torch` | CPU/CUDA float32/float64 | `full`, `checkpoint` | `eager`, `compile` |

Selecting Torch alone does not change the boundary default: also pass `memory="full"` or `"checkpoint"`. Invalid combinations raise rather than silently falling back. Torch compile explicitly uses installed Inductor and propagates compilation errors. Native CUDA requires `execution="eager"`, `compile_steps=1`, `checkpoint_interval=None`.

CUDA full stores the six-state trajectory (two pressure levels and four CPML memories). Boundary stores shell history and final states and reconstructs the interior; history still grows with time and shell size. Torch checkpoint stores complete six-state segment boundaries and recomputes forward segments during backward. It is not native boundary reconstruction.

One first-order backward supports independent `v`, `rx`, `rz` and source amplitudes, including source velocity scaling and the full replicated-padding transpose. Geometry and planning inputs are not trainable. MPS, AMP/autocast, CUDA graphs, native checkpoint, Torch boundary, VRR3D and higher-order AD are unsupported. `starwave.vrr_reference` is an explicitly selected CPU/CUDA float32/64 full-history reference; it has no backend/execution/checkpoint/compile selectors and is not a fallback.

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
from starwave import vrr

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

This reference targets frozen V19 source `0.1.0.dev19`, commit `46d32877e3336d531520a696d5f2e074e6f4ef74`, published as [Release V19](https://github.com/StarrMoonn/StarWave/releases/tag/V19). Except for the separately labelled V21 elastic VRR and V22 qP TTI additions, signatures below are checked against that source, including the 23-parameter Scalar API and integrated `starwave.vrr`. These frozen V19 API contracts are a historical compatibility baseline, not a statement of current package availability. Use the matching complete source checkout for the separately labelled additions on this page; do not infer current PyPI availability from historical wheel records. Source, distribution and device-validation evidence remain separate. See [Status](status.md) and [installation](installation.md).

The physical parameterization follows [Whitmore et al., EAGE 2020](https://doi.org/10.3997/2214-4609.202010332). The tensor top convention follows [SWEEP dev](https://github.com/DeepWave-KAUST/sweep/tree/32235f796ffa5b272ca81b9016f8cc85e25ecdf1) commit `32235f796ffa5b272ca81b9016f8cc85e25ecdf1`; the native VRR CUDA step/VJP and boundary reconstruction are independent implementations, not an upstream CUDA port.

The reviewed source records user-supplied RTX 4060 Laptop and A30 numerical reports, including native full/boundary and input gradients; CPU/Inductor, host loading, actual GPU, multi-GPU, long FWI and performance are separate evidence categories. This documentation build does not rerun or certify those GPU results, and no result is transferred to untested wheel/device/physical combinations. A first-order backward is not a Born/LSRTM or Hessian API.

(vti)=
## VTI Function

This section follows the [current VTI source contract](https://github.com/StarrMoonn/StarWave/blob/7145b5306af11615eb50c28a9ef64bb991a0bd97/starwave/vti.py), including native CPU/CUDA selective history and optional material-gradient sampling. It does not change the frozen V19/V16 snapshots or establish availability in an older PyPI wheel. Use a matching source checkout and explicitly rebuild native libraries.

```{py:function} starwave.vti(vp: torch.Tensor, epsilon: torch.Tensor, delta: torch.Tensor, rho: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, ...], dt: float | int, *, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, source_fields: str | list[str] | tuple[str, ...]=('sH', 'sV'), receiver_fields: str | list[str] | tuple[str, ...]=('vz',), accuracy: int=4, pml_freq: float | int=25.0, pml_width: int | list[int] | tuple[int, ...]=20, boundary_buffer: int=5, memory: str='boundary', max_vel: float | int | None=None, freq_taper_frac: float | int=0.0, time_pad_frac: float | int=0.0, time_taper: bool=False, backend: str | None=None, execution: str='eager', checkpoint_interval: int | None=None, compile_steps: int=1, model_gradient_sampling_interval: int=1) -> tuple[torch.Tensor, ...]

Propagates a Duveneck-type first-order acoustic VTI system, selecting 2D or 3D from `vp.ndim`. Outputs follow the selected receiver-component order. Each of the four models can independently request a first-order gradient in one backward pass. Fixed models still participate in all physical calculations; this interface is not full elastic VTI.

:param vp: **Required; positional or keyword.** Vertical P-wave velocity in m/s. float32 (native CPU/CUDA; Torch CPU/CUDA), positive and finite, with shape `[nx,nz]` or `[nx,ny,nz]` and at least 2 cells per dimension. The last axis is always vertical. All four models must have matching shape, dtype, and device. Physical axis order is neither inferred nor converted automatically.
:type vp: `torch.Tensor`
:param epsilon: **Required; positional or keyword.** Dimensionless Thomsen epsilon, float32 (native CPU/CUDA; Torch CPU/CUDA), with the same shape/device as `vp`. Must be finite and strictly greater than -0.5, and the coefficients and derivatives must be numerically representable. `epsilon<delta` is allowed, without implicit clipping; growing modes are a known limitation.
:type epsilon: `torch.Tensor`
:param delta: **Required; positional or keyword.** Dimensionless Thomsen delta, float32 (native CPU/CUDA; Torch CPU/CUDA), with the same shape/device as `vp`. Must be finite and strictly greater than -0.5, and the coefficients and derivatives must be numerically representable. The input is not constrained to `delta<=epsilon`; the model is not modified implicitly.
:type delta: `torch.Tensor`
:param rho: **Required; positional or keyword.** Density in kg/m³. float32 (native CPU/CUDA; Torch CPU/CUDA), with the same shape/device as `vp`, positive and finite; reciprocals, stiffnesses, and derivatives must be representable. There is no default density or implicit unit conversion. If the original data are confirmed to be in g/cm³, explicitly multiply them by 1000 before calling.
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
:param pml_width: **Default `20`; keyword-only.** Positive integer thickness in cells or `2*D` equal positive widths in low/high axis pairs. V19 accepts `[A,A,0,A]` in 2D `[x,z]` or `[A,A,A,A,0,A]` in 3D `[x,y,z]`, with equal positive remaining widths A. Only the top face may be zero; other zero faces, all-zero, unequal remaining widths, negative/noninteger/bool values are rejected. This selects the projected normal-traction closure below.
:type pml_width: `int | list[int] | tuple[int, ...]`
:param boundary_buffer: **Default `5`; keyword-only.** Nonnegative buffer in spatial grid cells, not a number of time steps or a checkpoint interval. Boundary mode requires at least `accuracy // 2 + 1`, which is 5 at eighth order; full mode allows 0. All absorbing sides use total padding `pml_width + boundary_buffer + accuracy // 2`; a top-zero face uses only `accuracy // 2` halo cells, without top PML or buffer.
:type boundary_buffer: `int`
:param memory: **Default `'boundary'`; keyword-only.** Model-gradient history strategy. Native CPU/CUDA supports full/boundary, not checkpoint; full stores selected effective material operands, while boundary retains complete reconstruction data. Torch supports full/checkpoint, not boundary. No automatic fallback. Without a requested model gradient, no model-backpropagation history is retained. See selective history and sampling below.
:type memory: `str`
:param max_vel: **Default `None`; keyword-only.** Anisotropic CFL/PML velocity envelope in m/s. With `None`, it is computed dynamically. An explicit value must be positive, finite, and cover `max(vp*sqrt(max(1,1+2*epsilon,sqrt(1+2*delta))))`. Covering only `max(vp)` is insufficient. Density contrast is still recomputed. Planning also depends on dimensionality, spacing, and finite-difference coefficients, and is not differentiated.
:type max_vel: `float | int | None`
:param freq_taper_frac: **Default `0.0`; keyword-only.** Finite fraction in `[0,1]` controlling the cosine taper of high-frequency FFT bins during temporal resampling; the bin count is obtained by truncating the proportional count to an integer. During resampling, the last positive rFFT bin is suppressed even when this fraction is 0, including for odd lengths; preservation of every frequency is not guaranteed. This setting does not change the signal when the internal resampling ratio is 1.
:type freq_taper_frac: `float | int`
:param time_pad_frac: **Default `0.0`; keyword-only.** Finite fraction in `[0,1]`. Appends `int(time_pad_frac*T)` user samples of zeros before resampling and removes them afterwards. Does not increase the public recording length or the actual propagation duration. It does not change the signal when the internal resampling ratio is 1.
:type time_pad_frac: `float | int`
:param time_taper: **Default `False`; keyword-only.** Whether to apply a nonperiodic Hann window during temporal resampling: after upsampling and before downsampling. Changes both the signal and its backpropagation transpose. Accepts only Booleans, not the integers 0/1. It does not change the signal when the internal resampling ratio is 1.
:type time_taper: `bool`
:param backend: **Default `None`; keyword-only.** None selects the native backend from CPU/CUDA model placement, never Torch. `'native_cpu'` explicitly selects built CPU kernels; `'cuda'` requires CUDA models; `'torch'` selects CPU/CUDA tensor propagation and autograd. No implicit device transfer or fallback. `'cpu'` and MPS are rejected. Torch requires explicit full/checkpoint.
:type backend: `str | None`
:param execution: **Default `'eager'`; keyword-only.** Torch accepts `'eager'` or `'compile'`. Compile uses PyTorch Inductor/AOTAutograd and raises on failure without eager fallback. Native propagation only accepts eager.
:type execution: `str`
:param checkpoint_interval: **Default `None`; keyword-only.** Torch checkpoint segment length in internal time steps: a positive integer, excluding bool; None resolves to 32. Keep None for full or native propagation. Longer segments increase the local replay graph and need not reduce peak memory.
:type checkpoint_interval: `int | None`
:param compile_steps: **Default `1`; keyword-only.** Positive integer, excluding bool; consecutive internal steps per Torch execution block. Try 4 for compile; the final partial block is retained. Does not change dt, nt, shots or loss. Native propagation accepts only 1.
:type compile_steps: `int`
:param model_gradient_sampling_interval: **Default `1`; keyword-only.** Integer in `1..2^31-1`, excluding bool, measured in **internal CFL steps**, not user samples. One retains the dense discrete material VJP; s>1 samples material products at call-global `t % s == 0`, weighted by `min(s, internal_nt - t)`, including a partial tail. This is approximate material-gradient quadrature, not the exact derivative of the unchanged forward. Propagation and the coupled adjoint still execute every internal step; no automatic multiplication by the CFL ratio.
:type model_gradient_sampling_interval: `int`
:returns: **`(records_0, ..., records_n)`**, with one contiguous float32 (native CPU/CUDA; Torch CPU/CUDA) tensor `[B,R,T]` per `receiver_fields` entry, in exactly the requested component order. The default returns only `(vz_records,)`, the vertical particle velocity in m/s, not pressure; `sH/sV` are stresses in Pa. Tensors are on the same device as the model, at nominal user times `0, dt, ..., (T-1)*dt`, without final states.
:rtype: `tuple[torch.Tensor, ...]`
```

**Torch path:** set `backend="torch"` on the same main entry point and explicitly choose `memory="full"` or `"checkpoint"` for CPU/CUDA without requiring a native build. Import still initializes available native libraries. Native history-layout/stream restrictions below describe the native path. See the [PyTorch backend](pytorch-backend.md) for all options and the memory table.

(vti-details)=
### Components, timing, and gradients

A positive stress rate adds positive stress to every selected stress component; internally, `internal_dt*U(rate)` is applied. There is no scalar-style `-vp² dt²` factor or model-dependent source scaling. Legacy per-step stress increments cannot be used directly as public rate inputs.

Stress recordings are time-aligned before downsampling by prepending zero and removing the final entry. Velocity recordings instead average the preceding and following half-time-step samples onto integer times, taking the initial preceding half step as zero. Both outputs are labeled with nominal user times, but their physical quantities and spatial locations differ. Resampling may produce endpoint ringing.

Each of `vp`, `epsilon`, `delta`, and `rho` requests a gradient only when its own `requires_grad=True`; fixed fields still participate in forward and adjoint propagation. No rule automatically freezes fields based on their values. `nn.Parameter` enables gradients by default; fixed fields can be registered as buffers.

`epsilon<delta` is not automatically repaired. A smaller dt does not guarantee removal of negative-stiffness growth or arbitrary PML instability. Covering the velocity envelope is only a necessary input condition.

The current VTI material VJP includes effective CPML operands and the complete replicate-padding transpose: sensitivities of copied buffer/PML cells sum back to their physical edges, intersections and corners. Numerical setup (CFL ratio, extrema/`max_vel` planning and CPML profiles) remains fixed, not differentiated. The former frozen-extension/cropped-gradient limitation is historical for VTI. At sampling interval 1 this preserves the dense discrete first-order material derivative with that fixed setup; intervals greater than 1 explicitly approximate its time integral.

Top-zero `pml_width` imposes `sV=0` at user `z=0`. Eliminating normal strain gives the surface constitutive coefficient `diag(2*rho*vp**2*(epsilon-delta),0)`; odd normal-stress and even face-velocity images form an exact negative-transpose derivative pair. This is a local pseudo-acoustic closure, not full elastic VTI or a claim of the interior high order at the surface. Sources including `sV` at `z=0` are rejected, including the default two-component source; surface `sV` receivers are zero. `sH` sources and velocity receivers retain native staggering. Native full/boundary keep the top strip; Torch full/checkpoint use the same closure. The projected material chain and complete padding transpose are differentiated; the `epsilon<delta` instability remains.

**Selective history and sampling.** Fixed parameters still participate in forward and coupled-adjoint physics; omission from an optimizer does not freeze a tensor. Native CPU/CUDA full saves two effective-strain histories when rho needs no gradient, adding 2/3 effective-force histories in 2D/3D when rho needs a VJP. A differentiable `rho = f(vp)` retains its density chain, even if rho is not an optimizer leaf. Torch preserves each input’s independent derivative path. Do not wrap the whole propagation in `no_grad` to freeze just one field.

For `model_gradient_sampling_interval=s`, native full retains `ceil(internal_nt/s)` material frames. Boundary keeps every reconstruction step, shell/state and coupled field; its peak memory need not decrease with s. Torch full/checkpoint eager/compile uses the same global material-VJP weights without resetting phase at checkpoint or compile-chunk boundaries; its autograd storage is not the native full history. Forward records, CFL subdivision, source/receiver clocks, batch and optimizer semantics are unchanged. Compare s>1 against s=1 for the actual bandwidth and objective; neither a universal error bound nor 1/s total-memory/time scaling is promised.

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

- Native supports CPU/CUDA FP32 full/boundary; native CUDA requires the default stream and one first-order backward per forward. Native CPU supports retained first-order backward. Source gradients, higher-order derivatives, AMP, CUDA graphs, custom streams, and public initial/final states are unsupported.
- CFL time substeps cannot compensate for inadequate spatial sampling. Spatial-resolution warnings are diagnostics, not certificates of accuracy or stability.
- For DataParallel, place the complete model in a Module and split acquisition inputs only along the shot dimension; see [Multi-GPU introduction](inversion/dataparallel.md).
- Interface checks do not rerun numerical or performance acceptance. Backend-specific evidence and untested hardware limits are recorded in the [source validation report](https://github.com/StarrMoonn/StarWave/blob/7145b5306af11615eb50c28a9ef64bb991a0bd97/docs/development/VTI_CPU_TORCH_SAMPLING_20261028.md); no blanket GPU or long-run convergence guarantee is implied.


(tti)=
## qP VTI / TTI Function

This API is included in the [V22 complete-source release](https://github.com/StarrMoonn/StarWave/releases/tag/V22), with its signature verified against tagged `starwave/tti.py`. This does not change frozen V19/V21 packages or establish PyPI availability. Use a matching complete checkout and explicitly rebuild native libraries. This single-field Liang local-direction approximate qP equation is separate from Duveneck `starwave.vti`: no rho, velocity/stress source fields or component selectors are accepted.

```{py:function} starwave.tti(vp, epsilon, delta, theta, grid_spacing, dt, *, source_amplitudes, source_locations, receiver_locations, accuracy=4, pml_freq=25.0, pml_width=20, boundary_buffer=5, memory='full', max_vel=None, freq_taper_frac=0.0, time_pad_frac=0.0, time_taper=False, backend=None, execution='eager', checkpoint_interval=None, compile_steps=1, return_wavefield=False, forward_callback=None, callback_frequency=1, free_surface=False)

:param vp: Required. Velocity [x,z], finite and positive; float32/float64 CPU/CUDA.
:param epsilon: Required. Same-shaped finite Thomsen epsilon > -0.5.
:param delta: Required. Same-shaped finite Thomsen delta > -0.5; epsilon < delta is allowed if the ellipticity guard passes.
:param theta: Required. Same-shaped finite tilt in radians. Zero selects Liang qP VTI, not the Duveneck starwave.vti equation. All four models share shape, dtype and device.
:param grid_spacing: Required. Positive scalar or (dx,dz), in model-axis order.
:param dt: Required. Positive user-sample interval; internal CFL substeps are automatic.
:param source_amplitudes: Required. Finite nonempty [shot,source,time], with model dtype/device. Scalar forcing: each internal increment is -vp(source)^2 * internal_dt^2 * upsampled_source; not a VTI stress-rate source.
:param source_locations: Required. Integer [shot,source,2], or [shot,2] for one source; physical (x,z) indices without PML offset. Duplicate sources add.
:param receiver_locations: Required. Integer [shot,receiver,2]; empty [shot,0,2] is valid. Duplicate receivers retain separate records.
:param accuracy: Default `4`. Spatial order 2, 4, 6 or 8.
:param pml_freq: Default `25.0`. Positive CPML reference frequency.
:param pml_width: Default `20`. Positive integer A, [A,A,A,A], or [A,A,0,A] in [x-low,x-high,z-low,z-high] order; only z-low may be zero. Other zero-face patterns and unequal positive widths are not supported.
:param boundary_buffer: Default `5`. Nonnegative spatial buffer, not a checkpoint interval.
:param memory: Default `'full'`. full or checkpoint only, on native and Torch; boundary is rejected.
:param max_vel: Default `None`. Optional positive bound covering the current anisotropic qP speed envelope, not merely max(vp). Too-small values are rejected.
:param freq_taper_frac: Default `0.0`. Temporal resampling frequency taper fraction.
:param time_pad_frac: Default `0.0`. Temporal resampling padding fraction.
:param time_taper: Default `False`. Enable the time taper during resampling.
:param backend: Default `None`. None selects native for the model device; native_cpu requires CPU, cuda requires CUDA, torch selects the tensor implementation. No implicit fallback or device transfer.
:param execution: Default `'eager'`. eager for native; eager or compile for Torch. Compilation must be requested explicitly.
:param checkpoint_interval: Default `None`. Positive integer INTERNAL steps for checkpoint; None resolves to 32. Must remain None with full.
:param compile_steps: Default `1`. Positive integer compiled time-chunk size for Torch; native requires 1.
:param return_wavefield: Default `False`. Return differentiable terminal [shot,x,z] before records when True.
:param forward_callback: Default `None`. Callable(state), with ScalarCallbackState detached, cloned, read-only physical wavefield; no callbacks during backward replay.
:param callback_frequency: Default `1`. Positive integer USER samples; callback after each interval and at the final internal update, not at time zero.
:param free_surface: Default `False`. Scalar pressure release at z=0; also implied by top-zero PML. Sources on the wall are rejected.
:returns: (records,) with [shot,receiver,time]; or (terminal_wavefield,records) with return_wavefield=True.
```

**Backends and gradients.** Native CPU/CUDA use explicit discrete adjoints, not time-loop autograd. Native full saves six pre-step derivative/history planes; checkpoint stores all six recurrence-state planes every K internal steps and replays each segment before reverse traversal. Snapshot plus replay storage scales as O(ceil(nt_internal/K)+K), excluding models, records and fixed workspace. Torch supports eager/compile with full/checkpoint. First-order gradients reach vp, epsilon, delta, theta and source amplitudes, including source-location vp² scaling and the full replicate-padding transpose at model edges. CUDA requires the default stream and one backward per forward. No boundary reconstruction, AMP, MPS, double backward, 3D or initial-state API is supported.

**Physics, time and surface.** The local-direction equation is nonlinear; its fixed 1e-26 denominator regularizer breaks source-amplitude homogeneity near zero gradients. It is neither an exact spectral qP projector nor elastic qSV/P–S conversion. Positive frozen-direction ellipticity is required. CFL screening includes tilt, anisotropy, unequal spacing and mixed stencils; it is not a universal nonlinear/CPML stability theorem. Setup extrema, integer resampling ratios and CPML profiles are frozen for differentiation. Records use user times 0,dt,... after shifting raw post-step samples to [0,raw[:-1]]; the terminal field follows the last internal source update. Callbacks are diagnostic, not differentiable loss inputs. Scalar pressure release projects u(x,z=0)=0 and removes top PML; it is not elastic traction-free conversion. Normal derivatives use second-order centered closure in the first accuracy//2 interior rows and requested order elsewhere, not odd-ghost closure.

**Minimal CPU example.** This small Torch example checks first-order gradient wiring and callback timing; it is not a physics or GPU acceptance test.

<!-- tti:example:start -->
```python
import torch
import starwave

vp = torch.full((12, 10), 1500.0, requires_grad=True)
epsilon = torch.full_like(vp, 0.1, requires_grad=True)
delta = torch.full_like(vp, 0.05, requires_grad=True)
theta = torch.full_like(vp, 0.2, requires_grad=True)
source = torch.zeros((1, 1, 8), dtype=vp.dtype)
source[..., 1] = 1.0
source.requires_grad_()
src = torch.tensor([[[6, 5]]], dtype=torch.long)
rec = torch.empty((1, 0, 2), dtype=torch.long)
steps = []
terminal, records = starwave.tti(
    vp, epsilon, delta, theta, (10.0, 12.0), 0.0001,
    source_amplitudes=source, source_locations=src,
    receiver_locations=rec, accuracy=2, pml_width=2,
    backend="torch", execution="eager", memory="checkpoint",
    checkpoint_interval=4, return_wavefield=True,
    forward_callback=lambda state: steps.append(state.step),
    callback_frequency=3,
)
terminal.square().sum().backward()
assert records.shape == (1, 0, 8)
assert terminal.shape == (1, 12, 10)
assert steps == [3, 6, 8]
for value in (vp, epsilon, delta, theta, source):
    assert value.grad is not None and torch.isfinite(value.grad).all()
```
<!-- tti:example:end -->

[Full source API and acceptance limits](https://github.com/StarrMoonn/StarWave/blob/dev/docs/reference/TTI_API.md) · [CUDA notebook](https://github.com/StarrMoonn/StarWave/blob/dev/notebook/TTI/CUDA/TTI2D_ThreeLayer_CUDA.ipynb) · [Torch compile notebook](https://github.com/StarrMoonn/StarWave/blob/dev/notebook/TTI/TorchCompile/TTI2D_ThreeLayer_TorchCompile.ipynb)

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
:param pml_width: **Default `20`.** Positive integer thickness in cells or length-2D positive face widths in low/high axis order; positive faces may differ. V19 accepts top-zero `[0,A,B,C]` for 2D `[y,x]` and `[0,A,B,C,D,E]` for 3D `[z,y,x]`, with every remaining width positive. Other zero faces and all-zero are rejected. This selects improved-vacuum preparation on the first model axis: row 0 is fixed vacuum and row 1 is the first solid stress row. Top-zero rejects survey_pad, origin and supplied initial fields; see below.
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
- V19 import initializes the available elastic library before DataParallel; {py:func}`starwave.prepare_elastic` remains an optional explicit device check. Keep complete materials in a Module and split acquisition along shots only. Successful initialization is not multi-GPU numerical acceptance.
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

(elastic-vrr-api)=
## Elastic VRR Function

V21 addition, checked against authorized dev source `ecec2876d450a2327a26114a7bb7035e38d81acd`. This is a 2D independent momentum/stress operator, not acoustic VRR or ordinary elastic. Requires matching source; no current PyPI availability or new numerical acceptance is implied.

```{py:function} starwave.elastic_vrr(vp, vs, rp_x, rp_z, rs_x, rs_z, grid_spacing, dt, *, source_amplitudes, source_locations, receiver_locations, source_component='pz', receiver_components=('pz',), source_type='momentum_rate', density=None, accuracy=4, pml_width=20, pml_freq=25.0, max_vel=None, backend='torch', execution='eager', memory='full', checkpoint_interval=20, forward_callback=None, callback_frequency=1, return_state=False, free_surface=False, reflectivity_layout='edge', cfl_resample=True, return_time_sampling=False, compile_steps=1)

:param vp: **Required.** Positive P-wave speed [nx,nz], typically m/s; dimensions >=3. All six finite models share float32/float64 dtype and CPU/CUDA device. Require vp**2 > 4*vs**2/3; autocast and MPS are unsupported.
:type vp: `torch.Tensor`
:param vs: **Required.** Positive S-wave speed matching vp; zero-Vs fluids are unsupported.
:type vs: `torch.Tensor`
:param rp_x: **Required.** P reflectivity [nx,nz], typically 1/m, on positive x edges (i+1/2,j). The last x edge is ignored and has zero gradient.
:type rp_x: `torch.Tensor`
:param rp_z: **Required.** P reflectivity matching vp on positive z edges (i,j+1/2); the last z edge is ignored and has zero gradient.
:type rp_z: `torch.Tensor`
:param rs_x: **Required.** S reflectivity matching vp on positive x edges; same exterior-edge convention as rp_x.
:type rs_x: `torch.Tensor`
:param rs_z: **Required.** S reflectivity matching vp on positive z edges; same exterior-edge convention as rp_z.
:type rs_z: `torch.Tensor`
:param grid_spacing: **Required.** Finite positive scalar or pair [dx,dz], typically metres.
:type grid_spacing: `float | int | list[float] | tuple[float, float]`
:param dt: **Required.** Finite positive user sampling interval, typically seconds; internal substeps may be smaller.
:type dt: `float | int`
:param source_amplitudes: **Required.** Finite [B,S,T] tensor, all dimensions nonempty, matching model dtype/device. Rates are integrated by internal dt after each complete leapfrog step; do not preintegrate.
:type source_amplitudes: `torch.Tensor`
:param source_locations: **Required.** int32/int64 [B,S,2] on the model device, physical [x,z] grid indices without padding offsets.
:type source_locations: `torch.Tensor`
:param receiver_locations: **Required.** int32/int64 [B,R,2] on the model device; physical grid indices, R may be zero.
:type receiver_locations: `torch.Tensor`
:param source_component: **Default `'pz'`.** One of px, pz, sxx, szz, sxz. Momentum lives on x/z half-grid faces; normal stresses at centres and sxz at corners.
:type source_component: `str`
:param receiver_components: **Default `('pz',)`.** Nonempty component sequence (or single string), choosing px,pz,sxx,szz,sxz,vx,vz. Output order follows this sequence; momentum is not silently relabelled velocity.
:type receiver_components: `str | list[str] | tuple[str, ...]`
:param source_type: **Default `'momentum_rate'`.** momentum_rate or velocity_rate for px/pz; stress_rate for sxx/szz/sxz. velocity_rate requires density.
:type source_type: `str`
:param density: **Default `None`.** Optional finite positive [nx,nz] tensor matching models. Required for velocity-rate sources or vx/vz receivers; differentiable face-density conversion calibrates units, not the independent-R propagation operator.
:type density: `torch.Tensor | None`
:param accuracy: **Default `4`.** Spatial order 2,4,6,8, excluding bool.
:type accuracy: `int`
:param pml_width: **Default `20`.** Nonnegative integer or four integers [x-low,x-high,z-low,z-high]. free_surface overrides z-low to zero.
:type pml_width: `int | list[int] | tuple[int, int, int, int]`
:param pml_freq: **Default `25.0`.** Finite positive CPML design frequency, typically Hz.
:type pml_freq: `float | int`
:param max_vel: **Default `None`.** CPML design bound >= max(vp); None uses the detached current maximum. Fix a covering bound for differentiation; CPML design constants are not differentiated.
:type max_vel: `float | int | None`
:param backend: **Default `'torch'`.** torch on CPU/CUDA, or native cpu/cuda on the matching device with a matching built library. No implicit fallback.
:type backend: `str`
:param execution: **Default `'eager'`.** Torch eager or compile; native backends require eager.
:type execution: `str`
:param memory: **Default `'full'`.** Torch accepts full/checkpoint; native CPU/CUDA accept full/boundary. Unsupported combinations raise; full history can be large.
:type memory: `str`
:param checkpoint_interval: **Default `20`.** Positive integer length in internal time steps for Torch checkpoint segments; independent of compile_steps.
:type checkpoint_interval: `int`
:param forward_callback: **Default `None`.** Optional callable receiving ElasticVRRCallbackState with detached cloned physical fields px,pz,sxx,szz,sxz, each [B,nx,nz]. Not a differentiable loss or restart state.
:type forward_callback: `Callable | None`
:param callback_frequency: **Default `1`.** Positive integer cadence in completed user steps; called at divisible steps only, not automatically at an unmatched final step. Callback step is one-based, dt is user_dt, with internal_step, internal_dt and step_ratio.
:type callback_frequency: `int`
:param return_state: **Default `False`.** Append final tuple of 13 padded staggered arrays: px,pz,sxx,szz,sxz followed by eight CPML histories. This API accepts no initial restart state.
:type return_state: `bool`
:param free_surface: **Default `False`.** Flat improved-vacuum top: physical z=0 is vacuum, z=1 first solid row. Only pz sources and pz/vz receivers may occupy z=0. Not arbitrary topography or simply disabling top PML.
:type free_surface: `bool`
:param reflectivity_layout: **Default `'edge'`.** Only edge is accepted; old cell-centred reflectivities must be recomputed.
:type reflectivity_layout: `str`
:param cfl_resample: **Default `True`.** Automatically choose coefficient-dependent internal substeps and differentiably resample sources/records; False rejects an excessive dt. Screening is not a stability guarantee for arbitrary R or CPML.
:type cfl_resample: `bool`
:param return_time_sampling: **Default `False`.** Append actual time-sampling plan after the optional state tuple, including user/internal dt and nt and step_ratio.
:type return_time_sampling: `bool`
:param compile_steps: **Default `1`.** Positive integer number of internal steps per Torch execution graph; native requires 1. Does not change the physical time sampling.
:type compile_steps: `int`
:returns: One [B,R,T] tensor per receiver component in order; first record follows the first completed user step. Optional state follows records; optional plan is last: (*records, state, plan) when both flags are True. State fields are [B,nx+pad_xlow+pad_xhigh,nz+pad_zlow+pad_zhigh], with each pad = effective face PML width + 2*(accuracy//2). Stresses precede momenta by internal_dt/2.
:rtype: tuple
```

(elastic-vrr-parameters)=
### Physical-model conversion

```{py:function} starwave.elastic_vrr_parameters(vp, vs, rho, grid_spacing, *, accuracy=4)

:param vp: **Required.** Positive P-wave speed [nx,nz], typically m/s; dimensions >=3. All six finite models share float32/float64 dtype and CPU/CUDA device. Require vp**2 > 4*vs**2/3; autocast and MPS are unsupported.
:type vp: `torch.Tensor`
:param vs: **Required.** Positive S-wave speed matching vp; zero-Vs fluids are unsupported.
:type vs: `torch.Tensor`
:param rho: **Required.** Finite positive density [nx,nz] matching vp/vs dtype/device, typically kg/m³; remains in the autograd chain.
:type rho: `torch.Tensor`
:param grid_spacing: **Required.** Finite positive scalar or pair [dx,dz], typically metres.
:type grid_spacing: `float | int | list[float] | tuple[float, float]`
:param accuracy: **Default `4`.** Spatial order 2,4,6,8, excluding bool.
:type accuracy: `int`
:returns: (vp, vs, rp_x, rp_z, rs_x, rs_z), each [nx,nz]. R = +0.5 D_edge log(rho V), nearest-neighbour differences divided by directional spacing for every accuracy, with zero exterior edges. Keep outputs connected for physical Vp/Vs/rho gradients. This local-expanded discretization does not guarantee exact ordinary-elastic equivalence at sharp interfaces.
:rtype: tuple
```



See the [runnable CPU forward/backward example and backend/build limits](examples/elastic-vrr.md). Native second derivatives are unsupported; full/checkpoint/boundary storage and callback snapshots are not interchangeable. This documentation update did not run new solver/GPU benchmarks.

(visco-gsls)=
## GSLS Function

This section describes frozen V19 GSLS, including its top-zero pressure-release closure and import lifecycle. `starwave.visco_gsls` is the viscoacoustic entry point; single-mechanism compatibility uses `mode="sls_compat", n_mechanisms=1`. Standalone SLS APIs remain removed. See [Status](status.md) and [installation](installation.md) for publication channels.

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

The first five parameters are positional-or-keyword; all others are keyword-only. `source_amplitudes`, `source_locations`, `receiver_locations` and `f_ref` are required. The four-entry V19 top-zero form may be a list or tuple, under the restrictions below. The table lists every parameter and signature default.

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

# V19 source; explicit CPU Torch reference, no native build required.
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

Complete native checkpoint call (first build the matching CUDA library and move models/source/geometry to one CUDA device; V19 import initializes the library, or explicitly use native_cpu with CPU tensors):

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

The following extension to the CPU example defines learnable per-mechanism weights and times. Once changed, it is experimental material, no longer the fixed published Hao1 spectrum. The optimizer must preserve Q0>S, and the actual spectrum needs new diagnostics. An external provider change alone needs no recompilation; native source changes require explicit rebuilding and restarting.

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

Optionally check an explicitly selected device and obtain the existing GSLS library. V19 import already loads and validates available libraries; subsequent calls reuse the ready handle without re-hashing loaded files. Does not compile or propagate.

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

Build explicitly from a complete authorized V19 source checkout using an installed Torch-compatible toolchain:

```bash
# Current dev: core/elastic/GSLS/Born; replace 80 with the actual target SM.
python compile_all.py --arch 80
# GSLS CUDA only:
python compile_visco_gsls.py --arch 80
# Separate native CPU library:
python compile_visco_gsls.py --backend cpu
```


V19 compile_all builds core, elastic, GSLS, Born and VRR. Each uses a matching native library. Configure build paths before import and restart Python after rebuilding. Editable registration and compilation are separate operations.

GSLS import loads existing native libraries and binds their entry points; subsequent calls reuse the handle. There is no runtime source/library hash or ABI-version gate, implicit compilation, repair or backend fallback. Rebuild after source changes and restart Python. Rebuild any retained GSLS CPU library explicitly as well. Optional preparation helpers check their own libraries/devices.

CUDA numerical correctness, actual multi-GPU execution, long FWI and performance require separate target-device validation. Library presence, compilation or successful preparation do not establish those results. This section supplies no new GPU or benchmark acceptance evidence.

(visco-gsls-references)=
### Scope and references

Unsupported: 3D, rho gradients, trainable PML/dt, boundary reconstruction, initial-state inputs, final-state returns, arbitrary-face free surfaces/topography, compression, illumination, AMP, CUDA graphs and supported higher-order propagation derivatives. Incidental higher-order Torch-graph behavior is not a contract.

- Hao & Greenhalgh (2021), “Nearly constant Q models of the generalized standard linear solid type and the corresponding wave equations”, GEOPHYSICS 86(4), T239–T260. [DOI](https://doi.org/10.1190/geo2020-0548.1).

The paper describes the physical model. This API follows the source sign, reference-phase normalization, discrete sampling, PML, padding and adjoint contracts above; bitwise agreement with other implementations is not claimed. See [inversion](inversion/fwi.md) for general objective organization. Historical SLS experiments are not current GSLS execution evidence.

(gpr)=
## Maxwell / GPR Function

Current-source addition, separate from frozen V19. Both entry points share the same validation and discrete propagation contract; the full 24-parameter signature is shown without changing the source defaults. `C0` denotes the vacuum wave speed.

```{py:function} starwave.gpr(epsilon_r, sigma, grid_spacing, dt, source_amplitudes=None, source_locations=None, receiver_locations=None, *, nt=None, accuracy=2, pml_width=10, backend='torch', storage_mode='full', checkpoint_interval=20, initial_state=None, source_component='ez', receiver_component='ez', compile_steps=1, forward_callback=None, callback_frequency=1, callback_requires_grad=False, mu_r=1.0, pml_max_velocity=C0, model_gradient_sampling_interval=1, polarization=None)

Propagates isotropic, nondispersive Maxwell fields with voxel materials and Yee staggering. This is a tensor/autograd API, not the gprMax input-file language. The GPR contract does not change any seismic operator’s axes or defaults.

:param epsilon_r: **Required; positional or keyword.** Finite, strictly positive relative permittivity on voxel cells: float32/float64 `[Ny,Nx]` or `[Nz,Ny,Nx]`, with at least one cell per axis. Dimensionless; physical permittivity is epsilon0 times epsilon_r. The rank selects the default polarization. Gradients are supported.
:type epsilon_r: `torch.Tensor`
:param sigma: **Required; positional or keyword.** Finite nonnegative electric conductivity in S/m, with exactly the model shape, dtype and device. It is Ohmic conductivity, not acoustic Q or a PML damping profile. Gradients are supported.
:type sigma: `torch.Tensor`
:param grid_spacing: **Required; positional or keyword.** Positive finite spacing in metres: scalar or `(dy,dx)` / `(dz,dy,dx)`, following model-axis order. Unequal spacings between axes are supported; no automatic transpose occurs.
:type grid_spacing: `float | int | list[float] | tuple[float, ...]`
:param dt: **Required; positional or keyword.** Positive finite propagation step in seconds. Material/order/spacing-aware CFL is enforced with safety factor 0.95. There are no hidden substeps, temporal resampling or automatic dt adjustment.
:type dt: `float | int`
:param source_amplitudes: **Default `None`; positional or keyword.** Finite electric current density J in A/m², shape `[B,S,T]`, matching model dtype/device. Supply together with source_locations or omit both. Singleton shot/source waveform axes may expand to the explicit geometry; gradients sum over those uses. The electric update subtracts J; do not pre-scale by dt or cell volume.
:type source_amplitudes: `torch.Tensor | None`
:param source_locations: **Default `None`; positional or keyword.** int64 `[B,S,D]` on the model device. Indices select the source component’s physical Yee anchors, not voxel centres; no PML offset is supplied. Integer-aligned axes allow 0..N and half-offset axes 0..N-1. Geometry never broadcasts; coincident sources add.
:type source_locations: `torch.Tensor | None`
:param receiver_locations: **Default `None`; positional or keyword.** int64 `[B,R,D]` on the model device, using the receiver component’s own anchor bounds. Shot counts match explicit source geometry and initial state. Duplicates retain separate traces. None or `[B,0,D]` requests zero receiver traces; the complete final state is still returned.
:type receiver_locations: `torch.Tensor | None`
:param nt: **Default `None`; keyword-only.** Nonnegative number of propagation steps. Inferred from the source time axis when supplied; an explicit nt must match it. Without sources, nt is required. With no geometry or initial state, the shot count defaults to one.
:type nt: `int | None`
:param accuracy: **Default `2`; keyword-only.** Staggered spatial finite-difference order: 2, 4 or 8. Order 6 is not part of this API.
:type accuracy: `int`
:param pml_width: **Default `10`; keyword-only.** Nonnegative cells per face, either one integer or `[y-low,y-high,x-low,x-high]` / `[z-low,z-high,y-low,y-high,x-low,x-high]`. Unequal widths and zero faces are accepted. External CPML material is edge-replicated differentiably; replicated gradients accumulate back to physical cells. Zero width is reflecting zero-ghost truncation, not open space or a general PEC/PMC boundary.
:type pml_width: `int | list[int] | tuple[int, ...]`
:param backend: **Default `'torch'`; keyword-only.** `torch`, `torch_compile`, `native_cpu` (`cpu` alias), or `cuda`. Inputs stay on their device. Native CPU/CUDA require explicitly built matching libraries and an appropriate device; unavailable implementations fail without fallback or automatic compilation. Torch routes require no StarWave native build.
:type backend: `str`
:param storage_mode: **Default `'full'`; keyword-only.** `full` or `checkpoint` on the supported backends. Full retains the selected differentiable history; checkpoint recomputes complete E/H/CPML states. `boundary` is rejected. This is storage_mode, not the seismic memory keyword.
:type storage_mode: `str`
:param checkpoint_interval: **Default `20`; keyword-only.** Positive segment length in propagation steps for checkpoint replay. Longer segments trade fewer boundary states for larger replay history; they need not reduce peak memory. Validated even in full mode.
:type checkpoint_interval: `int`
:param initial_state: **Default `None`; keyword-only.** Complete fields and every CPML memory, matching model dtype/device, shot count and padded `[B,*(N+1+PML_low+PML_high)]` shape. None starts from zero. All nonphysical high-side half-offset guard slots and associated psi must be zero. Restart state is differentiable, never mutated, and must use the same grid, spacing, PML and material convention.
:type initial_state: `MaxwellState | None`
:param source_component: **Default `'ez'`; keyword-only.** TMz: `ez`; TEz: `ex` or `ey`; 3D: `ex`, `ey` or `ez`. Only electric-current sources are supported. For TEz, explicitly select ex or ey because the default remains ez.
:type source_component: `str`
:param receiver_component: **Default `'ez'`; keyword-only.** TMz: `ez/hx/hy`; TEz: `ex/ey/hz`; 3D: any E/H component. One component is recorded per call at its native staggered site. TEz needs an explicit valid selector; hz is a receiver component, not an electric source.
:type receiver_component: `str`
:param compile_steps: **Default `1`; keyword-only.** Positive integer, excluding bool, limiting pure-tensor chunks for explicit backend='torch_compile'. A final partial chunk is retained. Full-graph compilation has no eager fallback and does not alter dt, nt, shots or the objective.
:type compile_steps: `int`
:param forward_callback: **Default `None`; keyword-only.** Callable receiving a GprCallbackState after completed intervals and the final update. Fields are independent, read-only physical N+1-anchor snapshots, without CPML memories. Half-offset endpoint guards are zero. No initial-time callback or checkpoint-replay callback is made; mutation is rejected.
:type forward_callback: `Callable | None`
:param callback_frequency: **Default `1`; keyword-only.** Positive integer, excluding bool, counting completed propagation steps. The final step is included even for a partial interval; validated even when no callback is supplied.
:type callback_frequency: `int`
:param callback_requires_grad: **Default `False`; keyword-only.** False detaches callback snapshots. True retains their graph for user-collected differentiable objectives. Retaining snapshots or losses also retains graph memory and may reduce checkpoint/sampling savings.
:type callback_requires_grad: `bool`
:param mu_r: **Default `1.0`; keyword-only.** Prescribed finite positive relative permeability: scalar, matching 0D tensor, or voxel-model-shaped tensor. Physical permeability is mu0 times mu_r. Tensor values use a separate harmonic H-face map, reducing to the local voxel for TEz Hz. Any requires_grad=True tensor is rejected; magnetic inversion is not supported.
:type mu_r: `float | int | torch.Tensor`
:param pml_max_velocity: **Default `C0`; keyword-only.** Prescribed positive finite CPML reference speed in m/s, about 299792458 m/s. It sets CPML rates only, not physical wave speed, and does not replace the material-aware CFL bound.
:type pml_max_velocity: `float | int`
:param model_gradient_sampling_interval: **Default `1`; keyword-only.** Positive integer, excluding bool. One gives the exact discrete first-order material VJP. Values s>1 approximate only epsilon_r/sigma gradient accumulation: sample call-global zero-based t divisible by s, weighted by min(s,nt-t). Propagation still executes every step; source and complete initial-state VJPs remain exact. See the sampling contract below.
:type model_gradient_sampling_interval: `int`
:param polarization: **Default `None`; keyword-only.** None selects 2D `tmz` or 3D `full` from model rank. Explicit 2D modes are `tmz` and `tez`; explicit 3D mode is `full` only. Incompatible rank/mode pairs are rejected. Switching polarization changes components and geometry, not just a label.
:type polarization: `str | None`
:returns: MaxwellResult(state, receiver_amplitudes): complete padded restart state and `[B,R,T]` records on the model dtype/device. E records are at `(n+1)*dt`; H records at `(n+1/2)*dt`. `result[-1]` is the receiver tensor.
:rtype: `MaxwellResult`
```

```{py:function} starwave.maxwell(epsilon_r, sigma, grid_spacing, dt, source_amplitudes=None, source_locations=None, receiver_locations=None, *, nt=None, accuracy=2, pml_width=10, backend='torch', storage_mode='full', checkpoint_interval=20, initial_state=None, source_component='ez', receiver_component='ez', compile_steps=1, forward_callback=None, callback_frequency=1, callback_requires_grad=False, mu_r=1.0, pml_max_velocity=C0, model_gradient_sampling_interval=1, polarization=None)

The electromagnetic entry point has exactly the same signature, defaults, units, returns and gradient scope as {py:func}`starwave.gpr`. GPR delegates to this implementation; use either name, not both for one propagation.

:returns: The same `MaxwellResult(state, receiver_amplitudes)` described above.
:rtype: `MaxwellResult`
```

(gpr-materials)=
### Voxel materials, axes and Yee anchors

The physical model contains N cells spanning N*spacing metres, not N nodal values. In 2D, axes are `[y,x]`, spacing `(dy,dx)` and coordinates `[iy,ix]`; in 3D they are `[z,y,x]`, `(dz,dy,dx)` and `[iz,iy,ix]`. Do not copy a seismic example’s axis order without checking it. Old nodal models, surveys and states are not compatible by shape alone; no legacy-nodal selector exists.

At each electric site, epsilon_r and sigma are arithmetic-averaged separately before forming coefficients: four transverse voxels for TMz Ez and each 3D E edge; two y-transverse voxels for TEz Ex and two x-transverse voxels for TEz Ey. With the resulting physical epsilon_e and sigma_e, `A=(epsilon_e-sigma_e*dt/2)/(epsilon_e+sigma_e*dt/2)` and `B=dt/(epsilon_e+sigma_e*dt/2)` give `E_new=A*E_old+B*(curl(H)-J)`, including CPML curl corrections. Do not average A/B or impedance instead. Fixed tensor mu_r uses the harmonic mean of two voxels normal to each H face; TEz Hz uses the local voxel value. Conduction belongs to the E update.

- TMz `(Ez,Hx,Hy)`: Ez has integer y/x anchors; Hx has y+1/2, Hy has x+1/2.
- TEz `(Ex,Ey,Hz)`: Ex has x+1/2, Ey has y+1/2, Hz has both offsets. A magnetic Hz receiver does not permit an Hz current source.
- 3D `(Ex,Ey,Ez,Hx,Hy,Hz)`: E_i has a half-cell offset along i; H_i along the other two axes. Identical integer indices do not collocate different components.

E is in V/m and H in A/m. Air/ground is an explicit epsilon_r/sigma material interface, not an acoustic pressure-release surface. Keep air thickness, physical geometry, CPML and grid resolution distinct. The 2D current is an invariant-line idealization, not a calibrated finite antenna.

(gpr-state)=
### Return state, clocks and restart

`starwave.MaxwellResult` is a named tuple with `state` and `receiver_amplitudes`. `starwave.MaxwellState` contains `fields` and `psi` tuples. Field order is the polarization-specific order above; there are four CPML memories in 2D and twelve in 3D. Every state tensor is `[B,*(N+1+PML_low+PML_high)]`; high-side half-offset slots are allocated zero guards, not physical samples. Their initial-state cotangents are zero. Initial fields and all memory tensors must be finite and complete.

At call start E is at 0 and H at -dt/2. Source sample n is J at `(n+1/2)*dt`; after step n+1, E is at `(n+1)*dt` and H at `(n+1/2)*dt`. The final state is therefore not collocated in time. Pass `initial_state=result.state` only with compatible geometry, grid/PML and material convention; a new public call starts a new local clock. Callback snapshots omit CPML memories and cannot replace this restart state. Detach all fields and psi explicitly if a restart must truncate the gradient chain.

(gpr-storage)=
### Storage, autograd and gradient sampling

First-order VJPs support epsilon_r, sigma, source waveforms and the complete initial state, including CPML memories. Replicated external-cell gradients accumulate back to physical model edges. The operator does not choose a residual, normalize the loss or perform an optimizer step. Geometry, numerical settings, fixed mu_r and CPML design are not inversion parameters. Native higher-order AD is rejected; tensor higher-order operations need independent validation.

`full` and complete-state `checkpoint` are supported; `boundary` is not. Native material history stores compact pre-step E/curl data, not every field/psi at every time. Checkpoint saves complete segment-boundary states, replays a segment and retains its needed compact material history. Torch retains its propagation AD graph or recomputes checkpoints. Source-only/state-only gradients do not need material correlation history. No CPU/disk offload or FP16 history compression is implied.

For `model_gradient_sampling_interval=s>1`, forward and adjoint/state propagation still execute every step. Only material correlations are sampled at zero-based t=0,s,2s,..., with weight min(s,nt-t), so the final partial interval is weighted correctly. Callback, checkpoint and compiled chunk boundaries do not reset this call-global clock. A separate public restart call starts a new clock and horizon. Source/initial-state VJPs remain exact; material gradients are approximate and higher-order model derivatives are not exact. Compare s>1 with s=1 for the actual objective before using it in inversion.

Native full/checkpoint retains only sampled E/curl frames. Torch detaches material coefficients on unsampled steps to avoid applicable local material-gradient intermediates, but retains propagation AD. Neither whole-process memory nor time is promised to scale as 1/s. `starwave.maxwell_utils.estimate_compute_memory(..., model_gradient_sampling_interval=s)` reports selected CPU array payloads, not measured peak RSS or CUDA memory. `recommended_sampling_interval` is advisory, not an error bound, and never changes the solver setting automatically.

(gpr-callbacks)=
### Callback snapshots and differentiable objectives

`starwave.GprCallbackState` exposes `fields`, `component_names`, `wavefields` (a name-to-field mapping), `receiver_component` and `wavefield` (the selected receiver component). Snapshots have `[B,*(N+1)]` physical shape, with half-offset endpoint guards masked to zero even when the corresponding runtime point lies in PML. They are independent read-only clones; mutation raises.

`step=user_step=internal_step` counts completed steps; `dt=internal_dt` and `step_ratio=1` because GPR does not resample. `time=electric_time=step*dt`; `magnetic_time=(step-0.5)*dt`. Use the magnetic clock for H observations. Callbacks occur every callback_frequency steps plus the final step, never at time zero or during backward checkpoint replay. `callback_requires_grad=True` permits losses built from snapshots; False is diagnostic-only. DataParallel callback closures must handle their own concurrent bookkeeping.

(gpr-example)=
### Runnable TMz, TEz and 3D CPU example

Use the matching installed StarWave package. The example checks API/gradient wiring for three polarization choices using Torch CPU checkpoint storage; it is not a recovery, performance or GPU-acceptance experiment. The default exact material-sampling interval is retained. A real data objective can replace the synthetic record/state/snapshot loss.

<!-- gpr:example:start -->
```python
import starwave
import torch

for polarization, shape, source_component, receiver_component in (
    ("tmz", (8, 10), "ez", "ez"),
    ("tez", (8, 10), "ex", "hz"),
    ("full", (5, 6, 7), "ez", "hy"),
):
    epsilon_r = torch.full(shape, 6.0, dtype=torch.float64, requires_grad=True)
    sigma = torch.full(shape, 0.002, dtype=torch.float64, requires_grad=True)
    mu_r = torch.ones(shape, dtype=torch.float64)  # fixed, not trainable
    source = torch.zeros((1, 1, 9), dtype=torch.float64)
    source[..., 1] = 1.0  # electric current density, A/m²
    source.requires_grad_()
    src = torch.tensor([[[2] * len(shape)]], dtype=torch.long)
    rec = torch.tensor([[[3] * len(shape)]], dtype=torch.long)
    callbacks = []

    def observe(state):
        callbacks.append((state.step, state.wavefield.square().mean()))

    result = starwave.gpr(
        epsilon_r, sigma, grid_spacing=0.02, dt=1e-11,
        source_amplitudes=source, source_locations=src,
        receiver_locations=rec, accuracy=2, pml_width=1,
        backend="torch", storage_mode="checkpoint", checkpoint_interval=4,
        polarization=polarization, source_component=source_component,
        receiver_component=receiver_component, mu_r=mu_r,
        model_gradient_sampling_interval=1,
        forward_callback=observe, callback_frequency=4,
        callback_requires_grad=True,
    )
    loss = result.receiver_amplitudes.square().mean()
    loss = loss + sum(value.square().mean() for value in result.state.fields)
    loss = loss + sum(value for _, value in callbacks)
    loss.backward()
    assert result.receiver_amplitudes.shape == (1, 1, 9)
    assert [step for step, _ in callbacks] == [4, 8, 9]
    for value in (epsilon_r, sigma, source):
        assert value.grad is not None and torch.isfinite(value.grad).all()
```
<!-- gpr:example:end -->

(gpr-build)=
### Explicit build and installed-package import

Run from the complete StarWave source checkout, using the Python environment that will execute the solver:

```bash
python -m pip install -e .
python compile_all.py --backend cpu --equation maxwell
# Choose the actual CUDA architecture(s) for your machine:
python compile_all.py --backend cuda --equation maxwell --arch 80,89
```

`--equation gpr` is an alias for maxwell; the default `--equation all` includes Maxwell alongside the seismic components. Then start a new Python/Jupyter process and use ordinary `import starwave`, including from outside the checkout. Import binds available compiled libraries once; no manual preload, sys.path injection, source-hash gate or automatic compile/fallback is needed. Pure `torch` / `torch_compile` needs no StarWave native build; `backend='torch_compile'` and compile_steps explicitly select tensor compilation. It is not the seismic execution='compile' API. Rebuild after source changes and restart the process before native use.

(gpr-scope)=
### Scope and source references

This addition follows Maxwell/GPR feature source `ab82ca51aedf745a00f26692361cf25156c3e13f` and its selective dev integration. It is separate from the frozen V19 baseline above and is not a claim about PyPI wheel availability. Fixed heterogeneous permeability is supported, but permeability inversion, dispersion, anisotropy, magnetic sources, geometry gradients, physical PEC/PMC/periodic walls and general higher-order AD are outside this contract. CUDA compilation alone is not numerical acceptance; device/compiler validation is limited to the explicitly recorded cases.

[Current source API](https://github.com/StarrMoonn/StarWave/blob/dc3eee1807370b97f2327b0f929faadfdd156cce/docs/reference/API_REFERENCE.md#gpr-voxel-api) · [Discrete contract](https://github.com/StarrMoonn/StarWave/blob/dc3eee1807370b97f2327b0f929faadfdd156cce/docs/maxwell_discrete_contract.md) · [Voxel migration](https://github.com/StarrMoonn/StarWave/blob/dc3eee1807370b97f2327b0f929faadfdd156cce/docs/gpr_voxel_migration.md) · [TEz geometry](https://github.com/StarrMoonn/StarWave/blob/dc3eee1807370b97f2327b0f929faadfdd156cce/docs/gpr_tez.md) · [Quickstart](https://github.com/StarrMoonn/StarWave/blob/dc3eee1807370b97f2327b0f929faadfdd156cce/docs/gpr_quickstart.md) · [Acceptance scope](https://github.com/StarrMoonn/StarWave/blob/dc3eee1807370b97f2327b0f929faadfdd156cce/docs/development/MAXWELL_ACCEPTANCE_20261025.md)

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

Optionally validates explicit logical CUDA devices and returns the existing core native library status. V19 import already initializes available core libraries before returning; this helper is not required before ordinary first propagation or DataParallel. Does not compile.

:param device_ids: Required. Nonempty, duplicate-free collection of nonnegative visible logical CUDA device IDs; Booleans are rejected. Use the device-visibility mapping of the current process.
:type device_ids: `list[int] | tuple[int, ...]`
:returns: Dictionary containing native status, `selected_device_ids`, and preparation messages. Successful initialization is not a GPU numerical test.
:rtype: `dict`
```

(native-notes)=
## Native runtime notes

After explicit compilation, ordinary V19 `import starwave` initializes available core, elastic, Born and GSLS libraries and integrated VRR. Shared dispatch is ready before the first parallel call; no serial warm-up or mandatory `prepare_*` step is needed. Import does not compile, initialize CUDA, select devices, allocate wavefields or propagate. Missing unbuilt libraries are skipped for supported pure-Torch use; an existing invalid build fails instead of silently falling back. Configure build paths before import and restart Python/Jupyter after rebuilding loaded code. Ready handles are reused without per-propagation source hashing or library polling.

Use `native_status()` for diagnostics and optional `prepare_native()` to check explicit logical CUDA devices. Neither function compiles or replaces GPU numerical acceptance; device `0` in the example must be visible. Core, elastic, GSLS and VRR retain their initial-load identity checks. Born performs full hashes/symbol auditing only during explicit builds/development audits, then uses ordinary one-time runtime loading.

V19 retains import-time native binding. The core loader now loads/binds once without runtime source/library hash, fingerprint or version probing; loading/binding errors still fail. A successful import is not a source/build identity audit or numerical check. Rebuild matching source and restart Python/Jupyter after changes; prepare_native remains an optional explicit device check.

(native-examples)=
## Native runtime examples

With frozen V19, a matching explicit build and compatible PyTorch, query status after import. Optional prepare calls validate available CUDA devices; they do not compile or establish numerical acceptance.

```python
import starwave

status = starwave.native_status()
print(status)
prepared = starwave.prepare_native([0])  # Optional device validation.
```

(prepare-elastic)=
## starwave.prepare_elastic

```{py:function} starwave.prepare_elastic(device_ids: list[int] | tuple[int, ...] | None=None) -> dict

Optionally obtain the separate elastic library and validate explicit logical devices. V19 import already initializes available elastic bindings; this is not a mandatory CUDA/DataParallel preload step. Does not compile or execute propagation tests.

:param device_ids: **Default `None`.** None loads the elastic library without selecting or preparing a CUDA device list, suitable for CPU use; a nonempty list/tuple selects distinct, nonnegative visible logical CUDA integer IDs (bool is rejected). CUDA requests validate the devices and build capability.
:type device_ids: `list[int] | tuple[int, ...] | None`
:returns: Dictionary with elastic-library status, selected_device_ids and a preparation message; successful loading is not numerical or DataParallel acceptance.
:rtype: `dict`
```

`starwave.elastic_native_status()` reads elastic-library status. The existing `prepare_native()` serves scalar/VRZ/VTI; elastic-only use does not require preparing those propagators.

(other-exports)=
## Other exported names and scope
StarWave V19 has no public `starwave.Scalar` wrapper class; `starwave.ScalarBorn` is the distinct Born module documented above. The tutorials define their own Module wrapper. Do not assume another library’s classes exist in StarWave, or add elastic state, `nt`, or storage options to scalar/VRZ/VTI calls.

(elastic-native-status)=
### starwave.elastic_native_status

```{py:function} starwave.elastic_native_status()

Inspect the existing elastic library status without compiling or propagating; a status report is not a numerical validation.

:returns: Inspect the existing elastic library status without compiling or propagating; a status report is not a numerical validation.
```


(scalar-illumination)=
### starwave.ScalarIllumination

```{py:class} starwave.ScalarIllumination(*, expected_shots)

One-use collector for native CUDA Scalar2D with positive PML widths and trainable velocity. Create on the caller thread for each batch; pass the same collector to replicas, seal after forward, backward once, then reduce. Use a context manager to release pending statistics on failure. Torch, 3D and any zero PML face are rejected.

:param expected_shots: **Required.** Total positive integer shot count before DataParallel scattering; bool rejected.
:type expected_shots: `int`
:returns: One-use collector for native CUDA Scalar2D with positive PML widths and trainable velocity. Create on the caller thread for each batch; pass the same collector to replicas, seal after forward, backward once, then reduce. Use a context manager to release pending statistics on failure. Torch, 3D and any zero PML face are rejected.
```


(illumination-seal)=
### starwave.ScalarIllumination.seal

```{py:method} starwave.ScalarIllumination.seal()

Finish registration after all forward replicas return, before backward. Returns None.

:returns: Finish registration after all forward replicas return, before backward. Returns None.
```


(illumination-reduce)=
### starwave.ScalarIllumination.reduce

```{py:method} starwave.ScalarIllumination.reduce(*, device)

After backward finishes, consume all registered replicas and return detached float64 global shot SUMs as IlluminationFields on the requested device. Incomplete reduction raises without consuming the collector; invalid native data raises. Successful reduction closes it.

:param device: **Required.** Explicit torch.device or device string for the reduction result.
:type device: `torch.device | str`
:returns: After backward finishes, consume all registered replicas and return detached float64 global shot SUMs as IlluminationFields on the requested device. Incomplete reduction raises without consuming the collector; invalid native data raises. Successful reduction closes it.
```


(illumination-close)=
### starwave.ScalarIllumination.close

```{py:method} starwave.ScalarIllumination.close()

Discard pending statistics and close the collector; safe after a failed forward/backward. Returns None.

:returns: Discard pending statistics and close the collector; safe after a failed forward/backward. Returns None.
```


(precondition-gradient)=
### starwave.precondition_gradient

```{py:function} starwave.precondition_gradient(gradient, fields, *, mode, damping=0.01)

Returns a detached new tensor g / (H / max(H) + damping); never changes the supplied gradient or optimizer. An all-zero H returns an unchanged clone. Apply after backward/reduction, before masking/clipping/step. pseudo_hessian is a per-shot geometric statistic, not an exact Hessian diagonal.

:param gradient: **Required.** Finite floating tensor, same shape, public axes and device as fields.
:type gradient: `torch.Tensor`
:param fields: **Required.** IlluminationFields returned by reduce, containing finite nonnegative statistics.
:type fields: `IlluminationFields`
:param mode: **Required.** One of 'source', 'receiver', 'pseudo_hessian'; no default.
:type mode: `str`
:param damping: **Default `0.01`.** Positive finite dimensionless water level; 0.01 is a numerical choice, not an optimal or literature default.
:type damping: `float`
:returns: Returns a detached new tensor g / (H / max(H) + damping); never changes the supplied gradient or optimizer. An all-zero H returns an unchanged clone. Apply after backward/reduction, before masking/clipping/step. pseudo_hessian is a per-shot geometric statistic, not an exact Hessian diagonal.
```


(ricker)=
### starwave.wavelets.ricker

```{py:function} starwave.wavelets.ricker(freq: float, length: int, dt: float, peak_time: float, dtype: Optional[torch.dtype]=None) -> torch.Tensor

Return a one-dimensional CPU Ricker wavelet of shape [length]. Explicitly move/expand it for the propagator device, shots and sources; this helper does not choose source units or scale by velocity/dt. dt=0 raises.

:param freq: **Required.** Central frequency in Hz.
:type freq: `float`
:param length: **Required.** Number of time samples.
:type length: `int`
:param dt: **Required.** Time sample spacing in seconds.
:type dt: `float`
:param peak_time: **Required.** Peak time in seconds.
:type peak_time: `float`
:param dtype: **Default `None`.** Requested torch dtype; None uses the PyTorch default dtype.
:type dtype: `Optional[torch.dtype]`
:returns: Return a one-dimensional CPU Ricker wavelet of shape [length]. Explicitly move/expand it for the propagator device, shots and sources; this helper does not choose source units or scale by velocity/dt. dt=0 raises.
```


(backend-capabilities)=
### starwave.torch_backend.backend_capabilities

```{py:function} starwave.torch_backend.backend_capabilities(equation=None)

Return a tuple of implemented BackendCapability records, optionally filtered by equation. This describes implementation eligibility, not available devices or hardware acceptance. The six-equation registry excludes the separately implemented integrated VRR path; consult the VRR section above. Unknown equation raises.

:param equation: **Default `None`.** None selects all; otherwise scalar, vrz, vti, elastic, visco_gsls or scalar_born.
:type equation: `str | None`
:returns: Return a tuple of implemented BackendCapability records, optionally filtered by equation. This describes implementation eligibility, not available devices or hardware acceptance. The six-equation registry excludes the separately implemented integrated VRR path; consult the VRR section above. Unknown equation raises.
```

IlluminationFields is a frozen result container with source, receiver and pseudo_hessian detached float64 tensors in physical model axes, and shots (int). They are global shot sums, not averages; time sums include no extra dt. Reduction cannot repair overflow already present in native FP32 statistics.

{ref}`genindex`
