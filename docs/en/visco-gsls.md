(visco-gsls)=
# GSLS viscoacoustic API (V16)

This page describes V16 source `0.1.0.dev16`. The two-dimensional `starwave.visco_gsls` is the current viscoacoustic entry point; a single mechanism uses explicit `mode="sls_compat", n_mechanisms=1`. The standalone `visco_sls`, `prepare_visco_sls` and `visco_sls_native_status` APIs are removed. Existing public PyPI `7.0.0` does not include GSLS; this manual update does not announce a new wheel. See [documentation status](status.md) and [installation](installation.md).

Supported: fixed spatially varying density, float32/float64, and first-order Vp/Q/source/provider-parameter gradients. Native CPU/CUDA support full/checkpoint; explicit Torch supports full/checkpoint and eager/compile. All main propagators gain execution options while native defaults remain unchanged; see the [PyTorch backend](pytorch-backend.md).

(visco-gsls-function)=
## visco_gsls

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
:param source_locations: **Required; keyword-only.** Integer `[shot,source,2]`, unpadded `[z,x]` coordinates; duplicates add
:type source_locations: `torch.Tensor`
:param receiver_locations: **Required; keyword-only.** Integer `[shot,receiver,2]`; duplicates repeat samples
:type receiver_locations: `torch.Tensor`
:param f_ref: **Required; keyword-only.** Positive Hz; exact phase-reference frequency, also used for fixed PML setup
:type f_ref: `float | int`
:param accuracy: **Default `4`; keyword-only.** Staggered FD order 2,4,6,8
:type accuracy: `int`
:param pml_width: **Default `20`; keyword-only.** Nonnegative integer or `(z,x)` symmetric widths. Replicate extension has its full accumulating gradient
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

The first five parameters are positional-or-keyword; all others are keyword-only. `source_amplitudes`, `source_locations`, `receiver_locations` and `f_ref` are required. The table lists every parameter and signature default.

<!-- api-doc:visco_gsls:parameters:start -->
| Parameter | Calling kind | Default when omitted | Type, shape and units | Purpose and behavior | Allowed values, rejections and limits |
|---|---|---|---|---|---|
| `vp` | `positional-or-keyword` | `required` | Tensor [nz,nx]; m/s | Reference phase speed| Positive finite float32/64 tensor `[z,x]`; phase speed at f_ref, m/s; optionally trainable |
| `q` | `positional-or-keyword` | `required` | Tensor [nz,nx]; dimensionless | Mode-specific Q| Matching positive tensor `[z,x]`, including +infinity. `hao1` (default): paper nominal Q0, not exact Qref; passivity Q0>S. `band_fit`: spatial fitting target, not exact Qref. `sls_compat`: exact modulus Q at f_ref. Optionally trainable |
| `rho` | `positional-or-keyword` | `required` | Tensor [nz,nx]; kg/m³ | Fixed density| Matching fixed positive finite density tensor, kg/m³; spatial variation supported, `requires_grad=True` rejected |
| `grid_spacing` | `positional-or-keyword` | `required` | real or pair; m | Grid spacing| Positive scalar or `(dz,dx)` in metres |
| `dt` | `positional-or-keyword` | `required` | real; s | Fixed time step| Positive seconds; conservative CFL guard uses unrelaxed speed and density contrast; no resampling |
| `source_amplitudes` | `keyword-only` | `required` | Tensor [B,S,T]; Pa/m² | Second-order forcing| Matching float tensor `[shot,source,time]`; second-order forcing Pa/m²; optional gradients |
| `source_locations` | `keyword-only` | `required` | int32/int64 Tensor [B,S,2]; cells | Unpadded source indices| Integer `[shot,source,2]`, unpadded `[z,x]` coordinates; duplicates add |
| `receiver_locations` | `keyword-only` | `required` | int32/int64 Tensor [B,R,2]; cells | Unpadded receiver indices| Integer `[shot,receiver,2]`; duplicates repeat samples |
| `f_ref` | `keyword-only` | `required` | real; Hz | Phase reference| Positive Hz; exact phase-reference frequency, also used for fixed PML setup |
| `accuracy` | `keyword-only` | `4` | integer; dimensionless | Staggered spatial order| Staggered FD order 2,4,6,8 |
| `pml_width` | `keyword-only` | `20` | integer or pair; cells | Symmetric PML| Nonnegative integer or `(z,x)` symmetric widths. Replicate extension has its full accumulating gradient |
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
## Modes, Q and reference velocity

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
## Source, coordinates, sampling and input domain

Models have shape `[nz,nx]`, in `[z,x]` order, with both extents at least two. All three models and sources must share dtype/device. CPU/CUDA float32/64 strided tensors and noncontiguous views are supported; contiguous copies retain autograd chains. Model/source dtype and device are not implicitly changed. Density is fixed, positive and finite, with a representable reciprocal. Only Q may contain +inf.

Geometry must explicitly have shapes `[B,S,2]` and `[B,R,2]`, with integer `[z,x]` cell indices inside the original unpadded model. Omitted source dimensions and out-of-domain points are rejected. Coordinates are copied to the model device as int64. Coincident sources add; repeated receivers repeat samples and accumulate cotangents during backward. B, S, R and T must each be at least one.

Sources are second-order forcing in Pa/m²; the pressure update contains `-dt**2*c_rel2*forcing`. The first source update has half weight for zero initial pressure and pressure time derivative. There is no hidden cell-volume factor; a Pa/s pressure-rate source cannot be substituted without conversion.

Records sample pre-update `p[n]` at `t=n*dt`, for T samples. The first sample is exactly zero. The last source sample only affects unreturned `p[T]`, so its records VJP is zero. For T=1, records and model/source gradients are zero. Review sign, amplitude units and time origin when migrating observations or wavelets.

(visco-gsls-gradients)=
## Gradients, padding and PML

- Vp, Q, source and provider closure parameters may be trained independently or together; only first-order propagation derivatives are supported. Density, geometry, dt, spacing, FD and CPML settings are fixed.
- Symmetric replicate padding produces `[nz+2*pml_z,nx+2*pml_x]`. Its complete transpose accumulates padded sensitivities into original edges and corners rather than using crop-backward. Model dependence of the source c_rel2 factor remains in the gradient chain.
- A zero PML width disables that axis. Both widths zero give a finite zero-extension domain without an absorber, not a free surface. Node/face CPML memories use the discrete transpose; damping, alpha and velocity envelope are fixed, nondifferentiated numerical setup.
- With gradients enabled and nonzero PML, trainable Vp/Q or any trainable material output (including provider closure parameters) requires explicit fixed max_vel covering `sqrt(c_rel2*(1+strength.sum(0)))`, not merely Vp. Keep the same envelope across optimization steps and paired finite differences.
- Forward-only, source-only gradients or zero PML can use max_vel=None; the inferred envelope is detached from autograd. Too-small explicit envelopes raise.
- Every forward has independent state/history. Loss, normalization, masks, batching and optimizers belong to the caller. There is no implicit clipping, shot reduction, dt repair or NaN replacement; invalid coefficients and nonfinite inputs/receiver cotangents raise.

(visco-gsls-stability)=
## Time-step guard

Let D be the sum of absolute staggered derivative weights and v_bound cover the unrelaxed speed. The conservative guard is:

```text
dt <= 0.8 / (v_bound * sqrt(max(rho)/min(rho))
             * D * sqrt(1/dz**2 + 1/dx**2))
```

Exceeding the bound raises with the allowed maximum dt. Choose a new dt and regenerate source sampling; there is no implicit resampling or substepping. dt² and inverse spacing must be representable as finite positive normal values in the model dtype. This is not proof of stability for arbitrary strong contrast, extreme Q, heterogeneous CPML or long runs; inspect finite values and boundary reflections in actual experiments. accuracy=2/4/6/8 is spatial order; pressure time discretization is second-order.

(visco-gsls-memory)=
## Full and checkpoint memory

Native CPU/CUDA support full and checkpoint; Torch supports full/checkpoint. boundary is unsupported. With full, checkpoint_interval must be None; an explicit checkpoint interval is a positive number of time steps. Native None uses input shape and requested histories; Torch None uses 32 internal steps. Neither uses GPU-specific tuning.

Native full histories depend on requested primitive gradients: c_rel2 needs force; strength/B need the spatial drive; A needs all M old material memories, for example with trainable relaxation times. Source-only gradients and no_grad observation require none of these volume histories. Fixed-Q Hao1 Vp gradients store T*V entries, joint Vp/Q stores 2*T*V, and a generic provider requesting every primitive gradient stores (M+2)*T*V. All five physical memories still evolve.

The following formulas apply only to native histories, not Torch autograd peak memory. Let N count padded cells, V=B*N, Fx=B*nz_padded*(nx_padded+1), Fz=B*(nz_padded+1)*nx_padded, and M count mechanisms. A complete checkpoint restart includes p, previous p, all material memories, and node/face CPML: S=(4+M)*V+Fx+Fz entries. Segment length K uses approximately `ceil(T/K)*S + K*H*V` tape/replay entries; H counts requested scalar histories, with old material memories contributing M. The default is `K=ceil(sqrt(S*T/(H*V)))` when histories are needed, bounded to [1,T]. These counts, even multiplied by dtype size, are not total peak memory: inputs, outputs, graphs, per-shot gradients, optimizer and workspaces are additional.

Native checkpoint restores complete states, replays each segment forward, and applies its discrete adjoint in reverse order; it does not reverse attenuation. Startup half-weight, global source clock and final partial segments are preserved, with adjoint state carried across segments. When checkpoint replay histories are needed, each time step is recomputed once, exchanging computation for history memory. There is no boundary inverse reconstruction or compression. Examples and memory estimates are not GPU benchmarks.

Torch checkpoint saves complete segment-start states including PML/material memories, replays forward with non-reentrant checkpointing, and differentiates with autograd. It does not invert attenuation or use the native adjoint history above. Short records/small models may not reduce peak memory.

(visco-gsls-example)=
## Minimal call and complete CPU example

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

# V16 source; explicit CPU Torch reference, no native build required.
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

Complete native checkpoint call (first move models/source/geometry to one CUDA device and prepare its library, or explicitly use native_cpu with CPU tensors):

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
## Material and spectrum helpers

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
## Advanced material_model protocol

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
## Native preparation, build and restart

```{py:function} starwave.prepare_visco_gsls(backend: str='native_cpu', device: str | torch.device | None=None) -> dict

Load an existing matched library and verify source/library/metadata and ABI, without compiling or propagating.

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

Build explicitly from a complete authorized V16 source checkout using an installed Torch-compatible toolchain:

```bash
# All three independent CUDA libraries; replace 80 with the actual target SM.
python compile_all.py --arch 80
# GSLS CUDA only:
python compile_visco_gsls.py --arch 80
# Separate native CPU library:
python compile_visco_gsls.py --backend cpu
```


compile_all builds/verifies three independent libraries: core, elastic and GSLS. combined_build covers core/elastic only. Source-tree GSLS output defaults to native/build/gsls; installed builds use the StarWave cache. An optional STARWAVE_BUILD_DIR uses its gsls subdirectory and must agree between build/runtime; standalone STARWAVE_GSLS_BUILD_DIR is not used. Complete-source execution requires neither pip registration nor export and downloads no compiler; optional editable installation is separate from compilation.

GSLS ABI=2, semantics=1, capabilities=255; core ABI=2 belongs to a separate library. Full source/library/metadata fingerprints must match; never borrow another tree's library or bypass checks. Rebuild and restart the process/Notebook kernel after changing loaded Python/native inputs. prepare_native and prepare_elastic do not replace GSLS preparation. There is no runtime compilation or backend fallback.

CUDA numerical correctness, actual multi-GPU execution, long FWI and performance require separate target-device validation. Library presence, compilation or successful preparation do not establish those results. This page supplies no new GPU or benchmark acceptance evidence.

(visco-gsls-references)=
## Scope and references

Unsupported: 3D, rho gradients, trainable PML/dt, boundary reconstruction, initial-state inputs, final-state returns, free surface, compression, illumination, AMP, CUDA graphs and supported higher-order propagation derivatives. Incidental higher-order Torch-graph behavior is not a contract.

- Hao & Greenhalgh (2021), “Nearly constant Q models of the generalized standard linear solid type and the corresponding wave equations”, GEOPHYSICS 86(4), T239–T260. [DOI](https://doi.org/10.1190/geo2020-0548.1).

The paper describes the physical model. This API follows the source sign, reference-phase normalization, discrete sampling, PML, padding and adjoint contracts above; bitwise agreement with other implementations is not claimed. See [inversion](inversion/fwi.md) for general objective organization. Historical SLS experiments are not current GSLS execution evidence.
