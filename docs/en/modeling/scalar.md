# Scalar Acoustics

The CUDA device, full/boundary history layout and native-build descriptions on this page apply to native defaults. V16 adds explicit Torch CPU/CUDA full/checkpoint; see the [PyTorch backend](../pytorch-backend.md).

In the current V17 source contract, `starwave.scalar` accepts a 2D or 3D velocity model `v`, selected by `v.ndim`, and returns a single-element tuple `(receiver_amplitudes,)`. Use `[0]` to obtain pressure-like records of shape `[B,R,T]`.

## Models, coordinates, and sampling

Models are CUDA float32 with at least 2 cells in every dimension. Coordinates `[i,j]` / `[i,j,k]` directly index the input model `v[i,j]` / `v[i,j,k]`, without PML offsets. For a model arranged as `[nx,ny,nz]`, 3D spacing is `[dx,dy,dz]`; physical axes are never swapped automatically. 2D requires equal spacing; 3D supports unequal per-axis spacing.

Finite positive, zero, and signed model values are accepted without establishing physical validity; conventional velocity models usually use positive values. CFL planning uses `max(abs(v))` and every axis spacing. An all-zero model requires an explicit positive `max_vel`. Output retains T user samples at interval `dt`.

Supported orders are `accuracy=2,4,6,8`. Positive scalar `pml_width` or equal positive widths on all four/six faces retain the absorbing case. V17 also accepts `[A,A,0,A]` for `[x,z]` or `[A,A,A,A,0,A]` for `[x,y,z]`, with one positive A on every other face. Other zero faces, all-zero and unequal remaining widths are rejected. The top uses only `M=accuracy//2` fixed-zero halo cells, without top PML/buffer; physical `z=0` stays active. This truncated-stencil closure is not odd reflection: the second-order effective surface lies one cell above that sample, and no high-order surface accuracy is implied. Coordinates retain their physical indices. `boundary_buffer=5` by default and must be at least `accuracy//2+1` in boundary mode. Published 7.1.0 does not include this V17 top-zero behavior.

## Source and call

The public source input is normalized forcing satisfying `u_tt = v² (Lap(u) - f)`. Supply fixed `f` in 2D; 3D also supports first-order source gradients. Do not additionally multiply by `-v² dt²`. If `u` is in Pa and length in m, `f` is in Pa/m².

```python
records = starwave.scalar(
    v, grid_spacing=10.0, dt=0.001,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
)[0]
```

This fragment requires prepared inputs. See [Quickstart](../quickstart.md) for a standalone 2D program, the {ref}`3D call example <scalar-3d-example>` for joint velocity/source differentiation, and the {ref}`scalar API reference <scalar>` for all parameters. This documentation update did not execute these GPU examples.

## Gradients and illumination

Set `v.requires_grad=True`, build a scalar loss from the records, and call `backward()` once. The source-location velocity factor remains part of the model-gradient chain. 2D sources remain fixed; 3D supports velocity or source gradients separately or together. `ScalarIllumination` is supported only for 2D scalar; 3D requires `None`. Illumination is rejected for top-zero scalar and for Torch; its statistics are not an exact Hessian.

The all-positive-PML 2D layout introduced in 6.0.0 uses directional compact PML states. Boundary mode stores pressure strips of width `M=accuracy//2` and two terminal pressure fields. For ordinary objective backpropagation (`illumination=None`), full history excludes PML/finite-difference halos but retains `boundary_buffer`: each axis is the original model size plus `2*boundary_buffer`. Enabling the optional illumination collector retains the full layout. Removing padded-region history does not remove propagation, adjoint, or reconstruction workspaces. Actual peak memory still depends on the model, PML, shot count, internal timesteps, and computational graph. Public full/boundary choices and defaults are unchanged. The PML-transpose correction does not change the model-extension gradient limits below.

V17 top-zero Scalar2D native full instead retains full-grid history; native boundary retains the required top strips without a full-history fallback. Both preserve the frozen-extension model-gradient limits.

The default is `memory="boundary"`. 3D stores six pressure faces of width `M=accuracy//2` and two terminal pressure fields for reverse reconstruction. `"full"` stores `Lap(u)` history over the padded volume at each internal step. Source-only gradients also retain history. Directional CPML slabs do not eliminate propagation, adjoint, or other workspaces. There is no automatic fallback or CPU/disk offload. See {ref}`Scalar3D reconstruction <reconstruction-scalar3d>` for memory estimates.

For linearized scattering, see [Born Function](../usage.md#born-function): `scatter` is velocity perturbation δv and its background operator differs from this scalar operator. Use Born background records for compatible observations; its 1D/2D/3D axes, dtype/state/gradient and zero-face contracts are separate.

Smaller time steps cannot fix spatial undersampling. Check waveforms, spatial resolution, gradients, and [boundary-gradient limitations](conventions.md) before inversion. Both full and boundary retain the inherited replicate-forward/crop-backward convention; neither is the complete derivative through model extension.
