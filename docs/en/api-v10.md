# StarWave v10.0.0 API

This page describes the GitHub [v10.0.0 release](https://github.com/StarrMoonn/StarWave/releases/tag/v10.0.0), source commit `5cc7c5aa9d86e50cb366d2e2ea2b09e57404d079`, Python package version `3.3.0`. It focuses on model axes, sampling and backend choices. [Historical Usage (V19 baseline)](usage.md).

## Installation and validation scope

Download `starwave-3.3.0-py3-none-manylinux_2_35_x86_64.whl` from the **private GitHub Release** with repository access and install it in the target PyTorch environment:

```bash
python -m pip install ./starwave-3.3.0-py3-none-manylinux_2_35_x86_64.whl
```

This wheel **was not published to PyPI**. A bare `pip install starwave` is not the v10 installation command. The wheel targets Linux x86-64 with glibc ≥ 2.35; its metadata covers Python 3.10–3.12. Clean installation and CPU forward/backward smoke tests were run on Python 3.12 with PyTorch 2.5.1+cpu and 2.6.0+cpu. All 11 CPU and 11 CUDA libraries were built and host-loaded. CUDA binaries cover sm70/75/80/86/89/90 plus compute80 PTX. No GPU numerical testing was run for this release; GPU propagation still needs a compatible driver, GPU, and CUDA-enabled PyTorch.

## Grid spacing, survey coordinates and PML faces

`grid_spacing` is specified in metres. A scalar is shared by all axes; a sequence follows the **actual model tensor axis order**, not the file's original layout. The last dimension of each source/receiver coordinate lists unpadded integer grid indices in that same order, not distances in metres. For per-face PML widths, list the low and high face of each model axis in turn. Widths are in cells. Free-surface and PML capabilities are operator-specific.

| Operator | Model and spacing supplied | Last dimension of survey locations |
|---|---|---|
| Scalar2D, acoustic VRR/VRZ | `[x,z]`, `(dx,dz)` | `[ix,iz]` |
| Scalar3D | `[x,y,z]`, `(dx,dy,dz)` | `[ix,iy,iz]` |
| VTI-VRR 2D | Default `[x,z]`, `(dx,dz)`; or `[z,x]`, `(dz,dx)` with `depth_axis=0` | `[ix,iz]` or `[iz,ix]`, respectively |
| GSLS 2D | `[z,x]` with `(dz,dx)`, or `[x,z]` with `(dx,dz)`; no `depth_axis` keyword | Model axis order |
| Elastic 2D / 3D | `[y,x]` with `(dy,dx)`, or `[z,y,x]` with `(dz,dy,dx)` | Model axis order |

For example, suppose the file contains `vp_zx[z,x]` with 12.5 m horizontal and 20 m vertical spacing. Pass `vp_zx.T` and `grid_spacing=(12.5,20.0)` to Scalar2D, and reorder any original `[iz,ix]` geometry to `[ix,iz]`. For GSLS with the original `[z,x]` model, use `(20.0,12.5)` and keep geometry already in `[iz,ix]`. The CPML construction method is unchanged; with different spacings, its physical thickness and coefficient values can differ by axis. Finer spacing can change the CFL substep ratio or trigger a fixed-step check.

## Scalar2D: rectangular grids and gradient sampling

`starwave.scalar(v, grid_spacing, dt, ...)` uses 2D velocity in `[x,z]` order. Source amplitudes are `[shot,source,time]`, source and receiver positions are `[shot,source,2]` and `[shot,receiver,2]`. For the ordinary call, `outputs[-1]` is the `[shot,receiver,time]` receiver tensor. This example assumes CPU tensors and positions in `[ix,iz]` order:

```python
import starwave

records = starwave.scalar(
    vp_zx.T, (12.5, 20.0), dt,
    source_amplitudes=source, source_locations=src_xz,
    receiver_locations=rec_xz, backend="native_cpu",
    memory="full", model_gradient_sampling_interval=1,
)[-1]
```

`model_gradient_sampling_interval=1` uses the complete discrete velocity gradient. An integer `s>1` samples only the velocity-gradient sum in **internal CFL steps**, starting at zero and using weight `min(s, internal_nt-t)`. The last group can contain fewer than `s` steps. With CFL ratio `r`, `s=r` samples once per user step; StarWave does not multiply this interval by `r` again. Forward and adjoint propagation, records and source gradients still use all internal steps. Native full storage retains fewer velocity-gradient history frames; boundary mode keeps its full reconstruction data. Sparse gradient sampling is approximate; compare against `s=1`. It is not data downsampling.

## Acoustic VRR / VRZ

`starwave.vrr` takes velocity and independent signed `rx, rz` (1/m) on an `[x,z]` grid. `starwave.vrz` forms `R_i=-Z D_i(1/Z)/2` from impedance `Z` (also available from velocity and density), with a matching centered stencil, then calls VRR. Directly differencing `0.5 log(Z)` is not generally the same discrete transform. Both calls return `(records,)`. Native CPU/CUDA support full/boundary; explicit Torch supports full/checkpoint. For a flat free surface use `pml_width=[w,w,0,w]` with positive integer `w`; this does not enable arbitrary zero faces or unequal absorbing widths.

## VTI-VRR: transposition, components and sample clock

VTI-VRR puts depth last by default. To supply all model tensors as `[z,x]`, set `depth_axis=0` and reorder spacing, geometry, and PML faces accordingly. If using the legacy `dx` and `dz` arguments, these remain **physical directions** with the same names after transposing.

The result is a tuple in `receiver_fields` order. The default is `("sV",)`; unpack it as `sV, = ...`. `sH/sV` are center tensile stresses in Pa; `vx/vz` are staggered-face velocities in m/s with no interpolation to centers. Velocity readout requires physical `density` in kg/m³; it affects readout, not VRR propagation. The output uses completed-step times `(n+1)*dt`, unlike conventional VTI's `n*dt`. The operator can resample for CFL; with `cfl_resample=False` it rejects an oversized time step instead.

## GSLS: model axes, fixed time step and materials

`starwave.visco_gsls(vp, q, rho, grid_spacing, dt, ...)` returns `(records,)`. The flat-top PML list is `[0,bottom,left,right]` on `[z,x]` or `[left,right,0,bottom]` on `[x,z]`, with positive widths on the other faces. Sources cannot occupy `z=0` on a free surface, where pressure receiver values are zero. GSLS validates the fixed `dt` against CFL without automatic resampling. `rho` is a fixed positive density; trainable material parameters must obey the material-provider and fixed-speed-envelope rules. Native CPU/CUDA and Torch support full/checkpoint, not boundary. The built-in default `mode="hao1"` requires an explicit valid `frequency_band=(scale, 200*scale)`; the meaning of `q` depends on the chosen material mode.

## Elastic: gradient sampling uses user steps

Elastic's `model_gradient_sampling_interval=m` counts **user time steps**, unlike Scalar's internal steps. It must be a positive integer. When a material gradient is active, it must divide `user_nt`, and the native material-gradient stride is `r*m` for CFL ratio `r`. With no active material gradients the effective interval is 1, including source-only and initial-state-only gradients. Every forward step still runs.

## Full operator references

These pages are pinned to the exact release source and require private repository access. The old examples retain their original versions and are not the current API contract.

- [Full API reference](https://github.com/StarrMoonn/StarWave/blob/5cc7c5aa9d86e50cb366d2e2ea2b09e57404d079/docs/reference/API_REFERENCE.md)
- [VRR / VRZ](https://github.com/StarrMoonn/StarWave/blob/5cc7c5aa9d86e50cb366d2e2ea2b09e57404d079/docs/reference/VRR_API.md) · [VTI-VRR](https://github.com/StarrMoonn/StarWave/blob/5cc7c5aa9d86e50cb366d2e2ea2b09e57404d079/docs/reference/VTI_VRR_RESEARCH_API.md)
- [GSLS](https://github.com/StarrMoonn/StarWave/blob/5cc7c5aa9d86e50cb366d2e2ea2b09e57404d079/docs/reference/VISCO_GSLS_API.md) · [Elastic](https://github.com/StarrMoonn/StarWave/blob/5cc7c5aa9d86e50cb366d2e2ea2b09e57404d079/docs/reference/DEEPWAVE_ELASTIC_API.md)
