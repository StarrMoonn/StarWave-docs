# Scalar Acoustics

In public **6.0.0**, `starwave.scalar` accepts a 2D or 3D velocity model `v`, selected by `v.ndim`, and returns a single-element tuple `(receiver_amplitudes,)`. Use `[0]` to obtain pressure-like records of shape `[B,R,T]`.

## Models, coordinates, and sampling

Models are CUDA float32 with at least 2 cells in every dimension. Coordinates `[i,j]` / `[i,j,k]` directly index the input model `v[i,j]` / `v[i,j,k]`, without PML offsets. For a model arranged as `[nx,ny,nz]`, 3D spacing is `[dx,dy,dz]`; physical axes are never swapped automatically. 2D requires equal spacing; 3D supports unequal per-axis spacing.

Finite positive, zero, and signed model values are accepted without establishing physical validity; conventional velocity models usually use positive values. CFL planning uses `max(abs(v))` and every axis spacing. An all-zero model requires an explicit positive `max_vel`. Output retains T user samples at interval `dt`.

Supported orders are `accuracy=2,4,6,8`. `pml_width` is a positive cell count, equal on all four 2D or six 3D faces. `boundary_buffer=5` by default and must be at least `accuracy//2+1` in boundary mode. Zero/asymmetric PML widths and free surfaces are outside this API.

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

Set `v.requires_grad=True`, build a scalar loss from the records, and call `backward()` once. The source-location velocity factor remains part of the model-gradient chain. 2D sources remain fixed; 3D supports velocity or source gradients separately or together. `ScalarIllumination` is supported only for 2D scalar; 3D requires `None`. Illumination statistics are not an exact Hessian.

In 6.0.0, 2D uses directional compact PML states. Boundary mode stores pressure strips of width `M=accuracy//2` and two terminal pressure fields. For ordinary objective backpropagation (`illumination=None`), full history excludes PML/finite-difference halos but retains `boundary_buffer`: each axis is the original model size plus `2*boundary_buffer`. Enabling the optional illumination collector retains the full layout. Removing padded-region history does not remove propagation, adjoint, or reconstruction workspaces. Actual peak memory still depends on the model, PML, shot count, internal timesteps, and computational graph. Public full/boundary choices and defaults are unchanged. The PML-transpose correction does not change the model-extension gradient limits below.

The default is `memory="boundary"`. 3D stores six pressure faces of width `M=accuracy//2` and two terminal pressure fields for reverse reconstruction. `"full"` stores `Lap(u)` history over the padded volume at each internal step. Source-only gradients also retain history. Directional CPML slabs do not eliminate propagation, adjoint, or other workspaces. There is no automatic fallback or CPU/disk offload. See {ref}`Scalar3D reconstruction <reconstruction-scalar3d>` for memory estimates.

Smaller time steps cannot fix spatial undersampling. Check waveforms, spatial resolution, gradients, and [boundary-gradient limitations](conventions.md) before inversion. Both full and boundary retain the inherited replicate-forward/crop-backward convention; neither is the complete derivative through model extension.
