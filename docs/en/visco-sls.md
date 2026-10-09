# SLS Viscoacoustic API

V14 adds an independent 2D single-standard-linear-solid (single-SLS) viscoacoustic interface. It is first publicly available in [StarWave 7.0.0](https://pypi.org/project/starwave/7.0.0/); see [Status](status.md) for artifact identity and verification scope. Existing scalar, VRZ, VTI, elastic and native-runtime signatures/defaults in [Usage](usage.md) are unchanged.

The interface supports fixed, spatially varying density and independent first-order gradients of Vp, Q and source waveforms. Propagation is limited to 2D and `memory="full"`, with the explicit source, axis and sampling conventions below.

(visco-sls-function)=
## visco_sls

```{py:function} starwave.visco_sls(vp: torch.Tensor, q: torch.Tensor, rho: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, float], dt: float | int, *, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, f_ref: float | int, accuracy: int=4, pml_width: int | list[int] | tuple[int, int]=20, memory: str='full', max_vel: float | int | None=None, backend: str='cuda') -> tuple[torch.Tensor]

:param vp: **Required; positional or keyword.** `[nz,nx]` 2D phase velocity in m/s, defined at `f_ref`. Both extents must be at least 2; positive finite float32/float64 tensor. Independently supports first-order gradients.
:type vp: `torch.Tensor`
:param q: **Required; positional or keyword.** Complex-bulk-modulus quality factor at `f_ref`, with the same shape, dtype and device as `vp`; positive finite values or positive infinity. Independently differentiable; `+inf` gives the lossless limit and zero Q sensitivity there.
:type q: `torch.Tensor`
:param rho: **Required; positional or keyword.** Positive finite density in kg/m³, matching `vp` in shape, dtype and device; its reciprocal must be representable. Spatial variation is supported, but density must be fixed: `requires_grad=True` is rejected.
:type rho: `torch.Tensor`
:param grid_spacing: **Required; positional or keyword.** Positive finite spacing in m: a scalar for equal spacing, or a two-element tuple/list `(dz,dx)` for possibly unequal spacing.
:type grid_spacing: `float | int | list[float] | tuple[float, float]`
:param dt: **Required; positional or keyword.** Positive finite sample interval in s, satisfying the conservative SLS time-step bound. There is no implicit resampling or automatic substepping.
:type dt: `float | int`
:param source_amplitudes: **Required; keyword-only.** Finite `[B,S,T]` tensor with nonempty extents, matching the model dtype and device. Second-order equation forcing in Pa/m²; independently supports first-order gradients. Making noncontiguous strided inputs contiguous preserves their autograd chain.
:type source_amplitudes: `torch.Tensor`
:param source_locations: **Required; keyword-only.** int32/int64 `[B,S,2]` coordinates indexing the original, unpadded model in `[z,x]` order. CPU or model-device coordinates are copied to model-device int64. Coincident sources add. There is no shorthand omitting the source axis.
:type source_locations: `torch.Tensor`
:param receiver_locations: **Required; keyword-only.** int32/int64 `[B,R,2]`, with `R>=1`, following the source coordinate rules. Repeated receivers yield repeated traces and their incoming cotangents add in backward.
:type receiver_locations: `torch.Tensor`
:param f_ref: **Required; keyword-only.** Positive finite reference frequency in Hz, defining Vp, Q and the fixed CPML alpha profile; angular frequency is `2*pi*f_ref`.
:type f_ref: `float | int`
:param accuracy: **Default `4`; keyword-only.** Staggered-grid spatial finite-difference order: `2,4,6,8`. Time discretization is second order.
:type accuracy: `int`
:param pml_width: **Default `20`; keyword-only.** Nonnegative cells per side, as a scalar or two-element `(z,x)` tuple/list. Each axis is symmetric. Zero disables that axis; two zeros leave a finite zero-exterior domain.
:type pml_width: `int | list[int] | tuple[int, int]`
:param memory: **Default `'full'`; keyword-only.** Only full volume histories are supported; `boundary` is rejected. Memory grows with the padded grid, shots and time steps.
:type memory: `str`
:param max_vel: **Default `None`; keyword-only.** Positive finite envelope in m/s covering the unrelaxed speed, which generally exceeds `vp`. An explicit envelope is required when gradients are enabled, Vp/Q are trainable and PML is nonzero; keep it fixed across training and paired finite-difference calls. Otherwise `None` may recompute a detached bound per call.
:type max_vel: `float | int | None`
:param backend: **Default `'cuda'`; keyword-only.** Explicitly choose `cuda`, `native_cpu` or `torch`. Native paths require CUDA/CPU tensors respectively and a matching native library; `torch` is a small-model reference. No automatic fallback or compilation.
:type backend: `str`
:returns: One-element tuple `(pressure_records,)`, recording `[B,R,T]` pressure in Pa on the model dtype/device.
:rtype: `tuple[torch.Tensor]`
```

(visco-sls-physics)=
## Physical parameters and sampling

`vp` is phase velocity at the reference frequency, not relaxed or unrelaxed speed. `q` is the real-to-imaginary ratio of the complex bulk modulus at that frequency under the `exp(i*omega*t)` convention. A single SLS describes attenuation and dispersion with frequency-dependent Q; it is not a broadband constant-Q model, and modulus Q is not silently identified with a wavenumber-attenuation definition.

The source is discrete-grid second-order forcing in **Pa/m²**. The pressure update includes `-dt**2 * c_rel**2 * forcing`, with a factor `1/2` on the first forcing update for zero initial pressure and pressure derivative. There is no hidden cell-volume factor. A Pa/s pressure-rate source cannot be substituted as the same physical quantity. When migrating wavelets or observations, review source calibration, sampling and equation conventions separately.

Records are the pre-update `p[n]` at `t=n*dt`, with exactly `T` samples. The first sample is exactly zero. The final source sample only affects the unreturned `p[T]`, so its record gradient is zero. With `T=1`, records and model/source gradients are zero; post-step records are never relabelled as pre-step times.

(visco-sls-gradients)=
## Gradients, PML and memory

- Vp, Q and sources may be trained separately or together; density, geometry, spacing, time step and PML setup are fixed. Each forward owns its history. The caller defines losses, normalization, masks and optimizers.
- Symmetric replicate-padding accumulates every padded-cell sensitivity into the original edges/corners through its complete transpose. Relaxed-speed source scaling remains in propagation, preserving model sensitivity at source positions.
- Node/face CPML memory recurrences use their discrete transpose. Damping, alpha and the speed envelope are fixed numerical setup, not differentiable material parameters.
- With gradients enabled, trainable Vp/Q and nonzero PML require an explicit `max_vel` covering the **unrelaxed speed**. Keep it fixed across optimizer steps and paired positive/negative finite-difference perturbations; `vp.max()` alone is generally insufficient.
- Forward-only, source-only differentiation or `pml_width=0` may use `max_vel=None`. An undersized explicit envelope is rejected.
- `full` retains volume histories, with memory increasing with grid size, shots and time steps. Boundary reconstruction, checkpoints, compression, illumination, free surfaces, 3D, initial-state inputs, final-state returns, native higher-order gradients, AMP and CUDA graphs are unsupported.
- float32/float64 models and sources are not implicitly cast or moved. Nonfinite inputs and incoming receiver cotangents are rejected without hidden clamping. Native backward supports value-preserving noncontiguous saved-tensor restores, copying to contiguous storage when needed.

(visco-sls-stability)=
## Time-step guard

Let `S` be the sum of absolute staggered FD weights and `v_bound` an envelope covering the actual unrelaxed speed. The public guard uses:

```text
dt <= 0.8 / (v_bound * sqrt(max(rho)/min(rho))
             * S * sqrt(1/dz**2 + 1/dx**2))
```

The density-contrast factor is conservative. An excessive time step raises an error with a suggested maximum dt; the caller must choose a new step and regenerate source samples. No implicit repair occurs. This no-PML energy bound and margin do not prove stability for arbitrary high contrasts, extreme Q, heterogeneous CPML or long runs; inspect finite outputs and boundary reflections for the actual experiment.

(visco-sls-example)=
## Standalone small-model call

This executable CPU Torch-reference wiring uses fixed variable density and explicitly trains Vp/Q/source while checking endpoint sampling conventions. The Ricker forcing is scaled to 1000 Pa/m² for this example. The loss is recording energy for checking autograd wiring, not an inversion-convergence result or a measured experiment.

```python
import torch
import starwave

device = "cpu"
dtype = torch.float64
nz, nx, nt = 12, 16, 100
dt = 0.001
vp = torch.full((nz, nx), 1800.0, dtype=dtype, device=device,
                requires_grad=True)
q = torch.full((nz, nx), 40.0, dtype=dtype, device=device,
               requires_grad=True)
depth = torch.linspace(0.0, 1.0, nz, dtype=dtype, device=device)
rho = (1800.0 + 200.0 * depth[:, None]).expand(nz, nx).contiguous()
t = torch.arange(nt, dtype=dtype, device=device) * dt
a = torch.pi * 18.0 * (t - 0.04)
source = (1000.0 * (1.0 - 2.0 * a.square()) * torch.exp(-a.square()))
source = source.reshape(1, 1, nt).requires_grad_()
src = torch.tensor([[[3, 8]]], dtype=torch.int64, device=device)
rec = torch.tensor([[[3, 4], [3, 8], [3, 12]]],
                   dtype=torch.int64, device=device)

records, = starwave.visco_sls(
    vp, q, rho, (10.0, 10.0), dt,
    source_amplitudes=source, source_locations=src,
    receiver_locations=rec, f_ref=18.0,
    accuracy=4, pml_width=(4, 4), memory="full",
    max_vel=2500.0, backend="torch",
)
assert records.shape == (1, 3, nt)
assert torch.isfinite(records).all()
assert torch.count_nonzero(records[..., 0]) == 0
records.square().mean().backward()
for gradient in (vp.grad, q.grad, source.grad):
    assert gradient is not None and torch.isfinite(gradient).all()
assert torch.count_nonzero(source.grad[..., -1]) == 0
```

For native CPU, first call `starwave.prepare_visco_sls(backend="native_cpu")`, then select `backend="native_cpu"` in propagation. For CUDA, use an indexed logical device such as `device="cuda:0"`, preload on the main thread, construct models/sources on that device and select `backend="cuda"`. A small Torch example provides no CUDA performance conclusion.

For Vp inversion with fixed Q, enable gradients only for `vp`. For joint Vp/Q inversion, enable both and pass them to the caller's optimizer. Density stays fixed. Real FWI also requires appropriate parameter bounds, source calibration, frequency selection and regularization; Vp/Q trade-offs remain. These experiment choices are not hidden inside propagation.

(visco-sls-runtime)=
## Native runtime preparation

```{py:function} starwave.prepare_visco_sls(device: str | torch.device='cuda:0', *, backend: str='cuda') -> dict

Validate and load an existing matching SLS native library, without compiling or propagating. Prepare before training or external multi-device allocation.

:param device: **Default `'cuda:0'`; positional or keyword.** CUDA requires a valid logical index, such as `'cuda:0'` or `torch.device('cuda:0')`; bare `'cuda'` is rejected. `native_cpu` ignores this argument.
:type device: `str | torch.device`
:param backend: **Default `'cuda'`; keyword-only.** `'cuda'` or `'native_cpu'`; `'torch'` is not accepted.
:type backend: `str`
:returns: Status dictionary. Successful loading and compatibility checks are not GPU numerical validation.
:rtype: `dict`
```

```{py:function} starwave.visco_sls_native_status(*, backend: str='cuda') -> dict

Read the selected SLS library's presence/load state without loading or compiling it.

:param backend: **Default `'cuda'`; keyword-only.** `'cuda'` or `'native_cpu'`.
:type backend: `str`
:returns: Status dictionary including diagnostic fields such as backend, library_exists, library_loaded, memory and gradient_order. Library presence and CUDA availability alone do not establish numerical correctness.
:rtype: `dict`
```

```python
import starwave

print(starwave.visco_sls_native_status(backend="cuda"))
status = starwave.prepare_visco_sls("cuda:0", backend="cuda")
print(status["library_exists"], status["library_loaded"])
```

The public 7.0.0 wheel includes prebuilt SLS CPU/CUDA libraries; see [Installation](installation.md) and [Status](status.md) for availability and platform requirements. Authorized source users build explicitly using the supplied instructions. Existing `prepare_native` and `prepare_elastic` do not replace SLS preparation. Restart the interpreter/notebook kernel after updating the Python package or loaded libraries; do not bypass compatibility checks.

(visco-sls-references)=
## Equation references and scope

- Bai, Yingst, Bloor and Leveille (2014), “Viscoacoustic waveform inversion of velocity structures in the time domain”, GEOPHYSICS 79(3), R103–R119. [Paper](https://www.tgs.com/hubfs/ION%20Papers/2014_GEO_JBai_Q_WFI.pdf), DOI: 10.1190/GEO2013-0030.1.
- [Official Devito viscoacoustic tutorial](https://www.devitoproject.org/examples/seismic/tutorials/11_viscoacoustic.html): educational background on SLS and other rheological models.

These references motivate the physical equations. StarWave defines its source sign, reference-phase-velocity parameterization, staggered finite-domain operators, CN material/PML timing and discrete backward under this interface's own contract; it makes no bitwise-equivalence claim to the paper or Devito. See the [SLS Marmousi2 Example](examples/visco-sls.md) for actual fixed-Q and joint Vp/Q inversion results. Existing [Scalar3D Examples](examples/index.md) retain their original versions and evidence; they are not SLS validation results.
