(elastic-vrr)=
# Elastic VRR (V21)

This chapter documents the new `starwave.elastic_vrr` interface in [Release V21](https://github.com/StarrMoonn/StarWave/releases/tag/V21), source `0.1.0.dev21`, commit `fa1eab2ba64771d9df1823b009aad5e3d72a6aa4`. It is a 2D local-expanded momentum/stress solver with six independent material coordinates, not the acoustic `starwave.vrr`, ordinary `starwave.elastic`, or an `elastic_vrz` interface. Other reference pages retain their explicitly labelled V19 baseline; historical tests are not V21 results.

## Install matching source

Use the complete authorized V21 source checkout and its requirements in your PyTorch environment. The source repository is private; access is required. This public manual distributes only interface documentation and synthetic examples, not the solver implementation or model data. No V21 PyPI wheel is asserted here. An unpinned `pip install starwave` or an older wheel is not a substitute for this source.

```bash
# From the authorized V21 source root:
python -m pip install --no-deps -e .
# Optional native CPU build:
python compile_all.py --backend cpu --equation elastic_vrr
# Optional native CUDA build; select your actual GPU architecture:
python compile_all.py --backend cuda --equation elastic_vrr --arch 80,89
```

Restart Python/Jupyter after installation or rebuilding. Torch eager CPU needs no native propagation library. Unified `all` builds include elastic VRR; installed `starwave-build` accepts the same flags. Build/import success is not numerical validation.

## Interface and coordinate convention

```text
starwave.elastic_vrr(
    vp, vs, rp_x, rp_z, rs_x, rs_z, grid_spacing, dt, *,
    source_amplitudes, source_locations, receiver_locations,
    source_component="pz", receiver_components=("pz",),
    source_type="momentum_rate", density=None, accuracy=4,
    pml_width=20, pml_freq=25.0, max_vel=None,
    backend="torch", execution="eager", memory="full",
    checkpoint_interval=20, forward_callback=None, callback_frequency=1,
    return_state=False, free_surface=False, reflectivity_layout="edge",
    cfl_resample=True, return_time_sampling=False, compile_steps=1,
)
starwave.elastic_vrr_parameters(vp, vs, rho, grid_spacing, *, accuracy=4)
```

- All six unbatched models have shape `[nx, nz]`, each dimension at least 3, and matching CPU/CUDA device and float32/float64 dtype. They must be finite; `vp > 0`, `vs > 0` and `vp**2 > 4*vs**2/3`. Zero-Vs fluids, MPS and autocast are unsupported.
- `grid_spacing` is a positive scalar or two positive spacings `[dx, dz]`; `dt` is the positive user sampling interval. `accuracy` is 2, 4, 6 or 8.
- Positive-edge reflectivity uses units 1/m: `rp_x[i,j]` is at `(i+1/2,j)`, `rp_z[i,j]` at `(i,j+1/2)`, likewise S. Only `reflectivity_layout="edge"` is supported. The final positive normal edge crosses into constant extension and is fixed to zero, with no effect or gradient from that input edge.
- `elastic_vrr_parameters` returns `(vp, vs, rp_x, rp_z, rs_x, rs_z)` with `R = +0.5 D_edge log(rho V)`. The difference is always nearest-neighbour, divided by its directional spacing, with a zero exterior edge, regardless of propagation accuracy. Positive matching `rho` is required. This helper preserves the physical-model autograd chain; it is optional when optimizing six independent coordinates.
- Sources have shape `[shot, source, time]`, matching model dtype/device. Source and receiver locations are integer `[shot, point, 2]` tensors on that device, indexing the unpadded physical `[x,z]` model, not metres.
- Momentum-rate sources use `px` or `pz`; stress-rate sources use `sxx`, `szz` or `sxz`. Velocity-rate sources require positive matching `density`. Amplitudes are rates, not preintegrated increments. Injection occurs after each complete stress/momentum step; recording begins after the first injection.
- Returns one `[shot, receiver, time]` tensor per requested component in order. `px,pz` are momentum, not velocity. `vx,vz` receivers require `density`; this calibrates units without imposing a common density on independent R fields.
- `return_state=True` appends the final tuple of 13 padded staggered fields (five momentum/stress fields plus eight CPML histories). `return_time_sampling=True` appends the sampling plan after any state tuple.

## Runnable CPU forward and gradient example

This complete example makes six independent optimization leaves. It does not detach the wave propagation or discard any time steps from its gradient. Download the same {download}`Python script <../../examples/elastic_vrr_cpu.py>`.

```python
import starwave
import torch

# 2D local-expanded momentum/stress equation; model axes [x, z].
vp = torch.full((24, 24), 2500.0, dtype=torch.float64)
vs = torch.full_like(vp, 1200.0)
rho = torch.full_like(vp, 2000.0)
models = starwave.elastic_vrr_parameters(vp, vs, rho, 10.0)
# Independent six-coordinate optimization leaves (not physical-model AD).
models = tuple(x.detach().clone().requires_grad_() for x in models)
source = torch.zeros((1, 1, 32), dtype=vp.dtype)
source[0, 0, 0] = 1.0  # momentum rate, integrated by the solver
src = torch.tensor([[[12, 10]]], dtype=torch.long)
rec = torch.tensor([[[12, 10], [13, 10]]], dtype=torch.long)
(pz,) = starwave.elastic_vrr(
    *models, 10.0, 0.0002,
    source_amplitudes=source, source_locations=src, receiver_locations=rec,
    source_component="pz", receiver_components=("pz",),
    source_type="momentum_rate", reflectivity_layout="edge",
    accuracy=4, pml_width=4, pml_freq=25.0, max_vel=3000.0,
    backend="torch", execution="eager", memory="full",
)
assert pz.shape == (1, 2, 32)
pz.square().sum().backward()
assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in models)
assert torch.isfinite(pz).all()
print("Elastic VRR CPU forward/backward passed:", tuple(pz.shape))
```

For physical Vp/Vs/rho inversion, instead make the physical models trainable and keep the helper outputs connected. Do not detach those outputs: otherwise the density/speed chain rule is lost. Source waveform gradients are supported; density calibration is differentiable when used. Freezing a six-coordinate leaf means disabling its gradients, clearing stale `.grad`, and excluding it from the optimizer and projection, not deleting its value from the coupled propagation.

## Native and Torch execution

| Choice | Device | Memory | Execution |
|---|---|---|---|
| `backend="torch"` | CPU/CUDA | `full`, `checkpoint` | `eager`, `compile` |
| `backend="cpu"` | CPU | `full`, `boundary` | `eager` |
| `backend="cuda"` | CUDA | `full`, `boundary` | `eager` |

To run the example with native CPU, build as above and change only `backend="torch"` to `backend="cpu"`. To use CUDA, also move every model, waveform and location tensor to CUDA before calling the solver. No backend silently substitutes a different memory mode. Native first derivatives cover six model fields, source and density calibration; native second derivatives are unsupported.

Torch checkpoint uses `checkpoint_interval` (default 20). Torch compile uses `execution="compile"`; `compile_steps` groups internal time steps (default 1), independently of checkpoint spacing. Native accepts only `compile_steps=1`. Measure compilation separately from warmed-up execution; no speed claim follows from choosing a larger chunk.

## Boundaries and numerical scope

CPML uses `pml_width`, `pml_freq` and a model-covering `max_vel`; fix that design bound when differentiating. Automatic differentiable CFL resampling is enabled by default, with all internal steps contributing to gradients. `cfl_resample=False` rejects an excessive timestep. This timestep screen is not a proof of energy stability or convergence for arbitrary independent reflectivities.

`free_surface=True` means a flat improved-vacuum top: physical z=0 is vacuum and z=1 is the first solid row. Only pz sources and pz/vz receivers may occupy z=0. This is not arbitrary topography; setting top PML to zero alone is not a traction-free surface. Vacuum material gradients vanish.

The local-expanded stencil is a specific discretization. Conversion from physical models does not guarantee exact finite-grid equivalence to ordinary elastic at sharp interfaces. This small synthetic forward/backward example does not establish long-time stability, field-data accuracy, GPU performance or inversion convergence.

## Formal inversion examples

In the authorized V21 source, `notebook/Elastic_VRR/` contains paired CUDA / TorchCompile notebooks and scripts:

- `Elastic2D_Marmousi2_VpVsRpRs_{CUDA,TorchCompile}`: six independent Vp/Vs/R coordinates, momentum observations.
- `Elastic2D_Marmousi2_VpVsZpZs_VRR_{CUDA,TorchCompile}`: four independent Vp/Vs/Zp/Zs coordinates, forming `Rp=0.5 D_edge log Zp` and `Rs=0.5 D_edge log Zs` in the graph. `rho_aux=0.5*(Zp/Vp+Zs/Vs)` calibrates velocity units, without forcing a shared physical density.
- P-only `VpZp` and `VpRp` controls keep S parameters fixed at truth on the solid crop. They do not add support for water (`vs=0`). Keep each notebook's observation units, initializer and objective; raw objective values across these parameterizations are not interchangeable.

See the authorized source's `notebook/Elastic_VRR/README.md` for exact filenames and experiment settings. These private notebooks and their model data are not mirrored by this public manual.
