# Scalar Acoustics

The CUDA device, full/boundary history layout and native-build descriptions on this page apply to native defaults. V16 adds explicit Torch CPU/CUDA full/checkpoint; see the [PyTorch backend](../pytorch-backend.md).

V18 scalar (reviewed at dev `e040aa9b8225ab387f7e4853bfe6162f5351f5d5`) adds arbitrary zero-face subsets (positive widths still equal), trainable 2D/3D sources, optional differentiable terminal physical fields and diagnostic forward callbacks. These source features are absent from frozen V17/its 8.0.0 release baseline and from 7.1.0. See [Scalar Function](../usage.md#scalar-function) for the 23-parameter contract; the older 20-parameter release snapshot remains separately pinned. Default output remains `(records,)`; `return_wavefield=True` adds the terminal physical field before records, so use `[-1]` when code must handle either form.

[Release V18](https://github.com/StarrMoonn/StarWave/releases/tag/V18) is published at `fba1f496b890f2dbaf3fbb3011d39f6d723cfea1`, preserving the reviewed scalar API from dev `e040aa9`. [PyPI 9.0.0](https://pypi.org/project/starwave/9.0.0/) is now published with these features for supported Linux x86_64 hosts; see [installation](../installation.md) for requirements and commands. V17 and the unpublished 8.0.0 baseline remain unchanged; 7.1.0 is the historical V16 wheel.

## Models, coordinates, and sampling

Models are CUDA float32 with at least 2 cells in every dimension. Coordinates `[i,j]` / `[i,j,k]` directly index the input model `v[i,j]` / `v[i,j,k]`, without PML offsets. For a model arranged as `[nx,ny,nz]`, 3D spacing is `[dx,dy,dz]`; physical axes are never swapped automatically. 2D requires equal spacing; 3D supports unequal per-axis spacing.

Finite positive, zero, and signed model values are accepted without establishing physical validity; conventional velocity models usually use positive values. CFL planning uses `max(abs(v))` and every axis spacing. An all-zero model requires an explicit positive `max_vel`. Output retains T user samples at interval `dt`.

Supported orders are `accuracy=2,4,6,8`. Current-dev scalar permits any zero-face subset or `pml_width=0`, with all positive face widths equal. Public face order follows model axes low/high. A zero face has only `M=accuracy//2` fixed-zero ghosts, no PML/buffer, and active physical edge samples. At order two the effective zero plane is one cell outside the edge; higher-order stencils are truncated, not odd-mirrored. The boundary-mode buffer minimum `accuracy//2+1` still applies, even with all-zero faces. Frozen V17 only supports the last-axis top-zero pattern; 7.1.0 has neither dev expansion nor the V17 top-zero feature.

## Source and call

The public source input is normalized forcing satisfying `u_tt = v² (Lap(u) - f)`. Current-dev scalar supports first-order source gradients in both 2D and 3D; released 2D scalar requires fixed sources. Do not additionally multiply by `-v² dt²`. If `u` is in Pa and length in m, `f` is in Pa/m².

```python
records = starwave.scalar(
    v, grid_spacing=10.0, dt=0.001,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
)[0]
```

This fragment requires prepared inputs. See [Quickstart](../quickstart.md) for a standalone 2D program, the {ref}`3D call example <scalar-3d-example>` for joint velocity/source differentiation, and the {ref}`scalar API reference <scalar>` for all parameters. This documentation update did not execute these GPU examples.

The optional terminal physical field is differentiable but omits previous-time and CPML state, so it cannot restart propagation alone. Required receiver_locations may be empty `[B,0,D]`. Forward callbacks receive independent detached snapshots after user-step intervals and the last update; use state.time for seconds, never assume step*dt is correct under CFL subdivision. Callback mutation during invocation raises, and checkpoint backward never repeats callbacks. See Usage for the runnable CPU example and `notebook/Scalar2D_Target_Wavefield.ipynb` in the authorized current-dev source.

## Gradients and illumination

Set `v.requires_grad=True`, build a scalar loss from the records, and call `backward()` once. The source-location velocity factor remains part of the model-gradient chain. Current-dev 2D/3D supports velocity or source gradients separately or together, from records, the terminal field or a joint loss. `ScalarIllumination` is supported only for 2D scalar; 3D requires `None`. Illumination is rejected for any zero face and for Torch; its statistics are not an exact Hessian.

The all-positive-PML 2D layout introduced in 6.0.0 uses directional compact PML states. Boundary mode stores pressure strips of width `M=accuracy//2` and two terminal pressure fields. For ordinary objective backpropagation (`illumination=None`), full history excludes PML/finite-difference halos but retains `boundary_buffer`: each axis is the original model size plus `2*boundary_buffer`. Enabling the optional illumination collector retains the full layout. Removing padded-region history does not remove propagation, adjoint, or reconstruction workspaces. Actual peak memory still depends on the model, PML, shot count, internal timesteps, and computational graph. Public full/boundary choices and defaults are unchanged. The PML-transpose correction does not change the model-extension gradient limits below.

For current-dev Scalar2D with any zero face, native full retains full-grid history; native boundary retains the required face strips without a full-history fallback. Both preserve the frozen-extension model-gradient limits.

The default is `memory="boundary"`. 3D stores six pressure faces of width `M=accuracy//2` and two terminal pressure fields for reverse reconstruction. `"full"` stores `Lap(u)` history over the padded volume at each internal step. Source-only gradients also retain history. Directional CPML slabs do not eliminate propagation, adjoint, or other workspaces. There is no automatic fallback or CPU/disk offload. See {ref}`Scalar3D reconstruction <reconstruction-scalar3d>` for memory estimates.

For linearized scattering, see [Born Function](../usage.md#born-function): `scatter` is velocity perturbation δv and its background operator differs from this scalar operator. Use Born background records for compatible observations; its 1D/2D/3D axes, dtype/state/gradient and zero-face contracts are separate.

Smaller time steps cannot fix spatial undersampling. Check waveforms, spatial resolution, gradients, and [boundary-gradient limitations](conventions.md) before inversion. Both full and boundary retain the inherited replicate-forward/crop-backward convention; neither is the complete derivative through model extension.
