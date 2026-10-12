(pytorch-backend)=
# PyTorch backend

The unified PyTorch backend is selected through `starwave.scalar / scalar_born / vrz / vti / elastic / visco_gsls`. With `backend="torch"`, tensor operations perform forward propagation and autograd generates derivatives; `execution="compile"` compiles that computation. V16 introduced the five non-Born entries supplied by the [7.1.0 Linux wheel](https://pypi.org/project/starwave/7.1.0/). V17 adds Born and the equation-specific flat-top/import behavior documented here; **8.0.0 is not published**. There is no macOS wheel: supported Mac MPS use requires authorized source. See [installation](installation.md) and [release status](status.md).

The separately installed {ref}`VRR source extension <vrr>` uses `starwave_vrr.vrr`, separate from the main-package entry points. It is absent from 7.1.0. Its explicit Torch mode supports CPU/CUDA float32/64 full/checkpoint and eager/compile; native CUDA float32 full/boundary requires its own build. Follow that page for extension installation and limits.

V18 scalar (reviewed at dev `e040aa9b8225ab387f7e4853bfe6162f5351f5d5`) adds arbitrary zero-face subsets (positive widths still equal), trainable 2D/3D sources, optional differentiable terminal physical fields and diagnostic forward callbacks. These source features are absent from frozen V17/its 8.0.0 release baseline and from 7.1.0. See [Scalar Function](usage.md#scalar-function) for the 23-parameter contract; the older 20-parameter release snapshot remains separately pinned.

[Release V18](https://github.com/StarrMoonn/StarWave/releases/tag/V18) is published at `fba1f496b890f2dbaf3fbb3011d39f6d723cfea1`, preserving the reviewed scalar API from dev `e040aa9`. PyPI **9.0.0 is still pending**; 7.1.0 remains the latest verified published wheel. Use matching V18 source for these features. V17 and the 8.0.0 baseline remain unchanged.

(torch-selection)=
## Explicit backend and device selection

- scalar/VRZ/VTI/elastic default to `backend=None`, a compatibility marker retaining native behavior, never an automatic Torch selection. `backend="cuda"` requires CUDA models.
- GSLS still defaults to `backend="cuda"`; `backend="native_cpu"` means the native C++ CPU implementation, distinct from `backend="torch"`.
- Torch supports CPU/CUDA and follows model devices. Arrange inputs on the same device; models are not moved implicitly. `backend="cpu"` is invalid. MPS is supported only for Scalar2D accuracy=4 float32 with backend="torch", full/checkpoint and eager/compile. Other MPS combinations, including Scalar3D, VRZ, VTI, elastic and GSLS, are rejected. No final Mac hardware, 500-epoch FWI or performance acceptance is claimed.
- scalar/VRZ/VTI retain `memory="boundary"` by default. When selecting Torch, also explicitly select `memory="full"` or `"checkpoint"`; otherwise the call fails rather than silently changing storage strategy.
- Torch propagation does not require native libraries or nvcc/`compile_all.py`; eager needs no native build. V17 import still loads/prepares existing native libraries and shared dispatch without compiling, initializing CUDA or propagating. Missing builds permit explicit Torch use; retained invalid builds fail. Manual prepare is optional, and no serial propagation warm-up is required. Compile requires the current PyTorch Inductor toolchain and, for CUDA, its applicable Triton environment. Failure does not fall back to eager. This does not make the historical Linux wheel installable on other platforms.

(torch-memory)=
## Backend, execution and memory table

| Entry / backend | Device | memory | Defaults / limits |
|---|---|---|---|
| Native scalar / VRZ / VTI | CUDA | full, boundary | boundary default; no checkpoint |
| Born native | CPU/CUDA float32/64 | full; boundary only on CUDA | Default full; no native checkpoint |
| Native elastic | CPU/CUDA | full; boundary on CUDA only | full default; existing offload/compression conditions remain separate |
| Native GSLS | CUDA or explicit native_cpu | full, checkpoint | full default; None interval uses a shape/history estimate |
| All six main-package entries, backend="torch" | CPU/CUDA | full, checkpoint | Override scalar/VRZ/VTI boundary default; eager/compile both supported |

| Option | Default | Torch meaning / rejections |
|---|---|---|
| execution | eager | eager or compile; native accepts eager only |
| compile_steps | 1 | Positive integer, excluding bool; consecutive internal steps per block; try 4 for compile; native accepts 1 only |
| checkpoint_interval | None | checkpoint only; positive integer, excluding bool; Torch None=32 internal steps; non-GSLS native requires None |

Torch full retains autograd intermediate tensors; checkpoint saves complete segment-start states including PML/GSLS memories and replays forward as needed with non-reentrant checkpointing. Neither is native boundary reverse reconstruction or inverse damping of attenuating states. Native GSLS selective-history formulas do not describe Torch memory. Longer intervals reduce checkpoint counts but enlarge local backward graphs; small models/short records need not reduce peak memory. Inputs, records, gradients, workspaces and the optimizer also consume memory.

compile_steps changes execution block size, not dt, nt, accuracy, shots, loss or update count; final partial blocks execute completely. checkpoint_interval counts internal time steps, whereas boundary_buffer counts spatial cells. There is no automatic GPU-model, SM, L2, free-memory tuning or shot splitting.

(torch-scope)=
## Scientific contracts and limits

| Torch entry | Dimensions | dtype | First-order gradients |
|---|---|---|---|
| scalar (V18) | 2D/3D | float32 | v and source in both dimensions; optional terminal-field loss |
| scalar_born | 1D/2D/3D | float32/64 | v, scatter (δv), source, initial background/scattered/PML states |
| VRZ | 2D | float32 | v and impedance or density; fixed source |
| VTI | 2D/3D | float32 | vp, epsilon, delta, rho; fixed source |
| elastic | 2D/3D | float32/64 | lamb, mu, buoyancy, sources, initial physical/PML states |
| GSLS | 2D | float32/64 | vp, q, source, material-provider tensor parameters; fixed rho |

All listed CPU/CUDA paths support orders 2/4/6/8 with full/checkpoint; MPS supports only Scalar2D accuracy=4 float32. Each equation retains its own axes, source units/scaling, recording clock, CFL and resampling contract; they are not interchangeable. See [Usage](usage.md) and {ref}`GSLS <visco-gsls>`. GSLS retains hao1, band_fit, sls_compat and material_model. A single mechanism explicitly requires `mode="sls_compat", n_mechanisms=1`; independent SLS APIs are not restored.

scalar/VRZ/VTI replicate edge values but freeze gradients through the extension, retaining the crop-image convention. Finite differences at outer model edges that regenerate padding differentiate a different map; distinguish interior-direction checks. GSLS accumulates the complete replicated-padding gradient back into the physical model. Torch VRZ full/checkpoint requires `boundary_buffer >= accuracy//2`.

Torch elastic differentiates every internal step and requires `model_gradient_sampling_interval=1`. Native material gradients are sampled at `CFL step_ratio × model_gradient_sampling_interval`. Even interval=1 can differ from complete AD when step_ratio>1. Exact native/Torch material-gradient comparisons also require step_ratio=1; this backend semantic distinction is not caused by compilation.

Explicitly rejected: Torch boundary, MPS outside the Scalar2D accuracy=4 float32 scope, AMP/autocast, scalar illumination, elastic callbacks, elastic storage offload/compression, and elastic sampling_interval!=1. Elastic requires forward_callback=None, storage_mode="device", storage_compression=False, and python_backend False or eager only; select compilation with execution="compile". Higher-order propagation derivatives, external CUDA Graph capture, vmap, custom streams and distributed training have not been accepted and are outside this guarantee.

Torch scalar/VRZ/VTI and GSLS validate incoming recording cotangents and reject NaN/Inf. A finite loss does not guarantee a finite recording gradient; inspect the objective and differentiation chain.

V17 Torch Born accepts `python_backend=False`, `model_gradient_sampling_interval=1`, device history and no compression; use `memory="full"` or `"checkpoint"` with eager/compile. Complete-state checkpoints preserve gradients to background/scattered initial states, sources, `v` and `scatter`. Born forward callbacks are supported as observers; backward callbacks are rejected. The first compile can be expensive, and errors do not fall back to eager. Its background operator is not `starwave.scalar`; compare observations against Born's returned background records. See [Born Function](usage.md#born-function) for full signatures, returns, axes and sampled native-gradient semantics.

V17 flat-top restrictions remain equation-specific on Torch: scalar/VRZ use fixed-zero exterior halos, VTI zero normal traction, elastic an improved-vacuum surface, and GSLS odd-pressure/even-vertical-face closure. Born's zero-face ghost convention is separate. Device eligibility is unchanged; MPS remains limited to Scalar2D accuracy=4 float32, with target-device free-surface acceptance separately scoped. See [model conventions](modeling/conventions.md).

Current-dev scalar supports return_wavefield and forward_callback on CPU/CUDA full/checkpoint eager/compile. Callbacks run after user-clock intervals and the final step, outside checkpoint replay. The returned terminal field is differentiable; callback snapshots are detached diagnostics. Existing source scaling and frozen-extension/cropped model gradients remain unchanged. No new MPS acceptance is claimed.

(torch-example)=
## Standalone CPU example

Use a matching source environment or the 7.1.0 wheel for this scalar-only example. This small example checks wiring, not FWI convergence; the manual build statically checks code without executing propagation. For CUDA, change device explicitly and place all tensors there. For compilation, select execution="compile" and optionally try compile_steps=4.

```python
import torch
import starwave

# Explicit CPU Torch propagation; no manual native preparation needed.
device = torch.device("cpu")
v = torch.full((24, 20), 1800.0, device=device,
               dtype=torch.float32, requires_grad=True)
t = torch.arange(48, device=device, dtype=v.dtype) * 0.001
a = (torch.pi * 15.0 * (t - 0.02)).square()
source = ((1 - 2 * a) * torch.exp(-a)).reshape(1, 1, -1)
src = torch.tensor([[[12, 4]]], device=device, dtype=torch.long)
rec = torch.tensor([[[8, 4], [12, 4], [16, 4]]], device=device,
                   dtype=torch.long)
records, = starwave.scalar(
    v, 10.0, 0.001,
    source_amplitudes=source, source_locations=src,
    receiver_locations=rec, accuracy=4, pml_freq=15.0,
    pml_width=8, boundary_buffer=5, max_vel=2000.0,
    memory="checkpoint", backend="torch", execution="eager",
    checkpoint_interval=32, compile_steps=1,
)
records.square().mean().backward()
assert records.shape == (1, 3, 48)
assert v.grad is not None and torch.isfinite(v.grad).all()
```

A separate CPU smoke passed on V16 source with Python 3.12.14 and PyTorch 2.14.1+cpu, using eager + checkpoint: records had shape `(1,3,48)` and the backward gradient was finite. Both language snippets are byte-identical. This validates only this small CPU wiring configuration, not compile, CUDA, MPS, FWI convergence or performance.

(torch-all-options)=
## Calls with every optional parameter

These are wiring fragments assuming valid prepared inputs, not complete experiments. Every optional parameter is explicit for four entries; the complete GSLS call is in the {ref}`GSLS examples <visco-gsls-example>`. None initial states mean zero initialization, not omitted API coverage. Do not reuse source/geometry/model variables between different equations without applying their contracts.

This complete scalar option list requires the V18 source. The simple CPU example above omits new arguments and remains release-compatible.

<!-- torch-api:scalar:full:start -->
```python
outputs = starwave.scalar(v, grid_spacing, dt,
    source_amplitudes=source,
    source_locations=src,
    receiver_locations=rec,
    accuracy=8,
    pml_freq=25.0,
    pml_width=20,
    boundary_buffer=5,
    memory="checkpoint",
    max_vel=None,
    freq_taper_frac=0.0,
    time_pad_frac=0.0,
    time_taper=False,
    illumination=None,
    backend="torch",
    execution="eager",
    checkpoint_interval=32,
    compile_steps=1,
    return_wavefield=False, forward_callback=None, callback_frequency=1,
)
```
<!-- torch-api:scalar:full:end -->

<!-- torch-api:vrz:full:start -->
```python
outputs = starwave.vrz(v, grid_spacing, dt,
    impedance=None,
    density=rho,
    source_amplitudes=source,
    source_locations=src,
    receiver_locations=rec,
    accuracy=8,
    pml_freq=25.0,
    pml_width=20,
    boundary_buffer=5,
    memory="checkpoint",
    max_vel=None,
    freq_taper_frac=0.0,
    time_pad_frac=0.0,
    time_taper=False,
    illumination=None,
    backend="torch",
    execution="eager",
    checkpoint_interval=32,
    compile_steps=1,
)
```
<!-- torch-api:vrz:full:end -->

<!-- torch-api:vti:full:start -->
```python
outputs = starwave.vti(vp, epsilon, delta, rho, grid_spacing, dt,
    source_amplitudes=source,
    source_locations=src,
    receiver_locations=rec,
    source_fields=('sH', 'sV'),
    receiver_fields=('vz',),
    accuracy=4,
    pml_freq=25.0,
    pml_width=20,
    boundary_buffer=5,
    memory="checkpoint",
    max_vel=None,
    freq_taper_frac=0.0,
    time_pad_frac=0.0,
    time_taper=False,
    backend="torch",
    execution="eager",
    checkpoint_interval=32,
    compile_steps=1,
)
```
<!-- torch-api:vti:full:end -->

<!-- torch-api:elastic:full:start -->
```python
outputs = starwave.elastic(lamb, mu, buoyancy, grid_spacing, dt,
    source_amplitudes_z=None,
    source_amplitudes_y=source_y,
    source_amplitudes_x=None,
    source_amplitudes_p=None,
    source_locations_z=None,
    source_locations_y=src_y,
    source_locations_x=None,
    source_locations_p=None,
    receiver_locations_z=None,
    receiver_locations_y=rec_y,
    receiver_locations_x=None,
    receiver_locations_p=None,
    accuracy=4,
    pml_width=20,
    pml_freq=None,
    max_vel=None,
    survey_pad=None,
    vz_0=None,
    vy_0=None,
    vx_0=None,
    sigmazz_0=None,
    sigmayz_0=None,
    sigmaxz_0=None,
    sigmayy_0=None,
    sigmaxy_0=None,
    sigmaxx_0=None,
    m_vzz_0=None,
    m_vzy_0=None,
    m_vzx_0=None,
    m_vyz_0=None,
    m_vxz_0=None,
    m_vyy_0=None,
    m_vyx_0=None,
    m_vxy_0=None,
    m_vxx_0=None,
    m_sigmazzz_0=None,
    m_sigmayzy_0=None,
    m_sigmaxzx_0=None,
    m_sigmayzz_0=None,
    m_sigmaxzz_0=None,
    m_sigmayyy_0=None,
    m_sigmaxyy_0=None,
    m_sigmaxyx_0=None,
    m_sigmaxxx_0=None,
    origin=None,
    nt=None,
    model_gradient_sampling_interval=1,
    freq_taper_frac=0.0,
    time_pad_frac=0.0,
    time_taper=False,
    forward_callback=None,
    callback_frequency=1,
    python_backend=False,
    storage_mode='device',
    storage_path='.',
    storage_compression=False,
    memory="checkpoint",
    backend="torch",
    execution="eager",
    checkpoint_interval=32,
    compile_steps=1,
)
```
<!-- torch-api:elastic:full:end -->

(torch-performance)=
## Compilation, repeated training and evidence

Time first compilation/warmup separately from steady-state forward + loss + backward + optimizer.step, and record allocated/reserved peaks separately. Shape, dtype, shot count, order or gradient-requirement changes may recompile. Only code can be cached, never stale model fields/coefficients/gradients. An eager/compile speedup is not a speedup over native CUDA.

If the same environment also uses native elastic/GSLS, changed Python entries change the source-closure identity. Rebuild from matching complete source and restart; never edit an old manifest by hand. See [installation](installation.md). Documentation and CPU interface checks do not replace target GPU, DataParallel, long FWI or performance acceptance. Existing Scalar3D figures, tutorials and presentation retain their historical evidence scope and are not relabelled as V16 results.

## Scalar2D MPS notebook

The source notebook L2_True_StarWave_Acoustic_MPS.ipynb and matching .py use plain import starwave, explicit PROJECT_ROOT resource loading, and no implicit installation/build. They preserve dt=0.003, nt=2000, 30 shots / 5 batches, 500 epochs, Adam lr=10, checkpoint_interval=32 and compile_steps=4. Spatial accuracy and compile block length are independent. Install the reviewed current source and restart Python; disable PYTORCH_ENABLE_MPS_FALLBACK. Unsupported operators or compilation fail explicitly without CPU/eager fallback. The published 7.1.0 wheel is Linux-only. It cannot be installed for Mac MPS; use the reviewed complete source. Historical 7.0.0 binaries do not provide this update.
