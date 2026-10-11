(vrr)=
# VRR Function

Independent **source extension** integrated in the reviewed dev source, distribution `starwave-vrr` (`0.1.0.dev1`), import `starwave_vrr`. It is not part of the published V16 / StarWave 7.1.0 wheel. There is no `starwave.vrr` alias. Obtain complete authorized source containing `extensions/vrr`; this public manual does not distribute the private implementation. See [installation](installation.md) for the separate main package.

(vrr-install)=
## Install and build

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

(vrr-api)=
## Signature and parameters

```text
starwave_vrr.vrr(v, grid_spacing, dt, *, rx, rz, source_amplitudes, source_locations, receiver_locations, accuracy=8, pml_freq=25.0, pml_width=20, boundary_buffer=5, memory="boundary", max_vel=None, freq_taper_frac=0.0, time_pad_frac=0.0, time_taper=False, backend="cuda", execution="eager", checkpoint_interval=None, compile_steps=1)
```

The first three arguments may be positional; all remaining arguments are keyword-only. Returns `(record,)`, with `record.shape == [B,R,T]`; use `result[-1]`. No final states or illumination are returned.

| Parameter | Default | Contract |
|---|---|---|
| `v` | `required` | Positive finite velocity [nx,nz], m/s. |
| `grid_spacing` | `required` | Positive uniform spacing: scalar or equal pair, metres. |
| `dt` | `required` | Positive user sampling interval, seconds. |
| `rx` | `required` | Finite signed x reflectivity-vector field, same shape/device/dtype as v; 1/m. |
| `rz` | `required` | Finite signed z reflectivity-vector field, same shape/device/dtype as v; 1/m. |
| `source_amplitudes` | `required` | Forcing [B,S,T]; explicitly expand shared wavelets across shots/sources. |
| `source_locations` | `required` | Integer [B,S,2], or [B,2] for one source per shot. |
| `receiver_locations` | `required` | Integer [B,R,2]; geometry indexes unpadded physical models. |
| `accuracy` | `8` | Interior spatial finite-difference order: 2, 4, 6 or 8. |
| `pml_freq` | `25.0` | Positive CPML reference frequency, Hz. |
| `pml_width` | `20` | Positive scalar, equal positive four-face widths, or [w,w,0,w] with positive integer w. |
| `boundary_buffer` | `5` | Spatial cells; at least accuracy//2+1, including full mode. |
| `memory` | `"boundary"` | CUDA: full/boundary. Torch: full/checkpoint; select explicitly. |
| `max_vel` | `None` | Detached current velocity maximum when omitted; explicit positive bound must cover v. |
| `freq_taper_frac` | `0.0` | FFT resampling frequency-taper fraction. |
| `time_pad_frac` | `0.0` | FFT resampling time-padding fraction. |
| `time_taper` | `False` | Optional time taper during resampling. |
| `backend` | `"cuda"` | cuda or torch; no automatic backend fallback. |
| `execution` | `"eager"` | Torch: eager/compile. Native CUDA accepts eager only. |
| `checkpoint_interval` | `None` | Torch checkpoint only; None means 32 internal steps, otherwise a positive integer. |
| `compile_steps` | `1` | Positive internal steps per compiled chunk; native CUDA accepts only 1. |

(vrr-backends)=
## Backends and gradients

| Backend | Device / dtype | Memory | Execution |
|---|---|---|---|
| `cuda` (default) | CUDA float32 | `full`, `boundary` (default) | `eager` selector; native execution |
| `torch` | CPU/CUDA float32/float64 | `full`, `checkpoint` | `eager`, `compile` |

Selecting Torch alone does not change the boundary default: also pass `memory="full"` or `"checkpoint"`. Invalid combinations raise rather than silently falling back. Torch compile explicitly uses installed Inductor and propagates compilation errors. Native CUDA requires `execution="eager"`, `compile_steps=1`, `checkpoint_interval=None`.

CUDA full stores the six-state trajectory (two pressure levels and four CPML memories). Boundary stores shell history and final states and reconstructs the interior; history still grows with time and shell size. Torch checkpoint stores complete six-state segment boundaries and recomputes forward segments during backward. It is not native boundary reconstruction.

One first-order backward supports independent `v`, `rx`, `rz` and source amplitudes, including source velocity scaling and the full replicated-padding transpose. Geometry and planning inputs are not trainable. MPS, AMP/autocast, CUDA graphs, native checkpoint, Torch boundary, VRR3D and higher-order AD are unsupported. `starwave_vrr.vrr_reference` is an explicitly selected CPU/CUDA float32/64 full-history reference; it has no backend/execution/checkpoint/compile selectors and is not a fallback.

(vrr-physics)=
## Models, source and time conventions

```text
p_tt = v² (p_xx + p_zz) + (v v_x - 2 v² rx) p_x + (v v_z - 2 v² rz) p_z
```

The 2D acoustic vector-reflectivity model uses public `[x,z]` arrays, depth last, and integer `[ix,iz]` geometry. Transpose every conventional `[z,x]` model. `rx/rz` are signed inverse-length fields, not density, Q or dimensionless reflection coefficients. For positive smooth impedance `Z=v*rho`, the continuum relation is `r=0.5*gradient(log Z)`; discrete finite differences need not satisfy the chain/product rules, so this is not discrete VRZ equivalence. No automatic density conversion is provided. Independent fields need not represent a physically realizable impedance.

If pressure has unit P, source forcing has unit P/m²: injection is `-v_source² * dt_internal² * U(source)`, where U is time upsampling. Initial states are zero. Public records apply downsampling to the pre-update-aligned sequence `[0, raw[:-1]]` at user times `0,dt,...`; FFT resampling can ring, so the first public sample need not be exactly zero for every substep ratio.

Internal planning uses `max_dt=0.6*h/(max_vel*sqrt(2))` and `ratio=max(1,ceil(dt/max_dt))`. This principal-speed bound does not guarantee stability for arbitrary reflectivity, velocity gradients or CPML. Use a fixed covering `max_vel` for optimization and finite differences, test the actual time horizon, and refine space/time. The propagator does not clamp/smooth models, repair losses or choose an optimizer.

(vrr-top)=
## Flat top boundary

Face order is `[x-low,x-high,z-low,z-high]`. Only `pml_width=[w,w,0,w]` with positive integer w selects a flat top; a positive scalar or equal positive four-face widths retain all-face absorption. Other zero faces, multiple/all-zero widths and unequal positive absorbers are rejected.

The top removes PML/buffer but keeps `M=accuracy//2` zero halo cells. Physical `z=0` remains active. This is a zero-halo closure, not an odd image extension: at second order the pressure-release plane is the first ghost row; higher interior FD order does not establish matching surface convergence order. No new free-surface switch, arbitrary face or topography support is implied.

(vrr-example)=
## Small synthetic example

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
## Source and validation scope

The exact reviewed dev runtime baseline is `904c9106bdb883325df8586b7c3412f7814a9737`, retaining the VRR runtime integrated at `6b9ec41d3bef1a8a3593a63f124c6ee4139955b0`. This dev extension is later than frozen V16 source `5a224dbb1e86154ff034bbedaf67c0e4cf7a1b6a`, which underlies public 7.1.0. Its reviewed VRR feature input is `e8ccf9b0fe70edd023fc4c6d0319abae6c203c65`; the dev integration preserves the existing main-package free-surface work and keeps the extension separately packaged. These source identities do not designate a new public wheel or release.

The physical parameterization follows [Whitmore et al., EAGE 2020](https://doi.org/10.3997/2214-4609.202010332). The tensor top convention follows [SWEEP dev](https://github.com/DeepWave-KAUST/sweep/tree/32235f796ffa5b272ca81b9016f8cc85e25ecdf1) commit `32235f796ffa5b272ca81b9016f8cc85e25ecdf1`; the native VRR CUDA step/VJP and boundary reconstruction are independent implementations, not an upstream CUDA port.

The reviewed source records user-supplied RTX 4060 Laptop and A30 numerical reports, including native full/boundary and input gradients; CPU/Inductor, host loading, actual GPU, multi-GPU, long FWI and performance are separate evidence categories. This documentation build does not rerun or certify those GPU results, and no result is transferred to the 7.1.0 wheel or arbitrary hardware/physical regimes. A first-order backward is not a Born/LSRTM or Hessian API.
