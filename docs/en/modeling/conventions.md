# Model, Acquisition, and Time Conventions

Unless stated otherwise, the tensor table and scalar/VRZ/VTI restrictions below describe native defaults. V17 source adds equation-specific flat-top boundaries and Born; published 7.1.0 remains the earlier V16 wheel, and 8.0.0 is not published. Explicit Torch CPU/CUDA full/checkpoint has separate device and feature limits; see the [PyTorch backend](../pytorch-backend.md).

## Tensor shapes

`B` is the number of shots, `S` the number of sources per shot, `R` the number of receivers per shot, `T` the number of user time samples, and `D` the number of spatial dimensions.

| Input/output | Shape | Convention |
|---|---|---|
| scalar model | `[N0,N1]` or `[N0,N1,N2]` | CUDA float32; `v.ndim` selects the dimension |
| VRZ model | `[N0,N1]` | CUDA float32 |
| VTI model | `[nx,nz]` or `[nx,ny,nz]` | The vertical axis is always last |
| `source_amplitudes` | `[B,S,T]` | Current-dev scalar2D/3D supports source gradients; released scalar2D and VRZ/VTI sources remain fixed |
| `source_locations` | `[B,S,D]` | Integer grid indices; torch.long is recommended |
| `receiver_locations` | `[B,R,D]` | Integer grid indices |
| Each recorded tensor | `[B,R,T]` | CUDA float32 |

Coordinates directly index the supplied physical model. They are **not coordinates in meters and do not include a PML offset**. Scalar / VRZ do not infer the user's physical axis order. In 3D scalar, `[i,j,k]` directly indexes `v[i,j,k]`. If the model is stored as `[nz,nx]` but acquisition coordinates use `[x,z]`, transpose the model explicitly first. All four VTI models must share the same axis order, shape, device, and dtype.

## Sampling and absorbing boundaries

`grid_spacing` and `dt` must be positive; typical units are meters and seconds. Internally, smaller time steps are planned according to the CFL condition, but the returned length remains `T`, with nominal user times `0, dt, ..., (T-1)dt`. 2D scalar / VRZ require equal spacing; 3D scalar and VTI allow per-axis spacing in model-axis order.

`pml_freq` configures PML and spatial-sampling diagnostics; it does not generate the source. V17 `pml_width` and surface geometry are equation-specific:

- Current-dev scalar only: any subset of the four/six faces may be zero, including all-zero; all positive widths remain equal. Each zero face keeps fixed-zero exterior ghosts and active physical samples, without PML/buffer or a high-order odd-reflection claim. Both dimensions support source gradients; optional terminal field `[B,*v.shape]` is at `T*dt`, while records end at `(T-1)*dt`. Empty receiver geometry `[B,0,D]` is supported. This does not broaden the released V17 or VRZ/VTI contracts below.
- Frozen V17 Scalar2D/3D and VRZ2D: positive scalar or equal positive face widths; only the low face of the last model axis may be zero, with every other width the same positive integer A. Use `[A,A,0,A]` for `[x,z]` or `[A,A,A,A,0,A]` for `[x,y,z]`. The exterior `accuracy//2` halo cells are fixed zero, the first physical row remains active, and top PML/buffer is removed. This is a truncated stencil, not odd reflection; the second-order effective pressure-release plane lies one cell above the first sample, with no high-order surface-accuracy claim. Scalar illumination is rejected with top-zero.
- Independent VRR2D: `[x,z]`, with positive scalar/equal face widths or `[A,A,0,A]`; see [VRR Function](../usage.md#vrr-function) for its separate extension contract.
- VTI2D/3D: the same last-axis top-zero layouts and equal positive remaining widths as scalar, but a projected zero-normal-traction closure. Sources including `sV` at `z=0` are rejected, including the default source; `sV` receivers there return zero. This is an acoustic closure, not elastic VTI.
- Elastic2D/3D: the first model axis is vertical (`[y,x]` / `[z,y,x]`). `[0,A,B,C]` / `[0,A,B,C,D,E]` selects improved-vacuum preparation, with positive remaining widths allowed to differ. Row 0 is fixed vacuum and row 1 is the first solid stress row. Supplied initial states, `origin` and `survey_pad` are rejected with top-zero. No coordinate shift is added.
- GSLS2D: axes `[z,x]`; a positive scalar or positive `(z,x)` pair gives symmetric absorbing faces. `[0,bottom,left,right]`, with the other widths positive and allowed to differ, selects pressure release: odd pressure/even vertical-face closure, surface pressure zero, pressure sources at `z>=1`. Other zero layouts and all-positive four-face lists are rejected.
- Born1D/2D/3D: axes `[x]`, `[y,x]`, `[z,y,x]`; a nonnegative scalar or `2*ndim` per-face widths is allowed. Zero faces retain inherited fixed-zero FD ghost cells while physical cells remain active. This broader domain does not apply to `starwave.scalar`, and it does not promise odd-image high-order surface accuracy or elastic traction-free physics.

These source contracts are absent from the 7.1.0 wheel. See the equation-specific [Usage](../usage.md) sections for all source, receiver, backend and derivative restrictions.

`boundary_buffer` is measured in spatial grid cells and defaults to 5. Boundary mode requires at least `accuracy // 2 + 1`; full mode allows 0, but still retains the configured boundary padding.

## Memory and automatic differentiation

The scalar, VRZ, and VTI equations on this page support `memory="boundary"` (the default) or `"full"`. Full mode stores the time history and may use more GPU memory. There is no automatic mode switching. The 3D scalar Radius-M face history, terminal fields, and workspaces are described in {ref}`Reconstruction <reconstruction-scalar3d>`; source-only gradients also retain history. Each forward pass supports only one first-order backward pass. For these native scalar/VRZ/VTI paths, CPU propagation, AMP, higher-order automatic differentiation, custom CUDA streams and public restart states are unsupported. Explicit Torch supports CPU full/checkpoint, and Born/elastic have their own state-return contracts; these are not restrictions on every StarWave entry point.

```{warning}
Model gradients are conditional on fixed extended-model, CFL, and PML settings. When replicate padding at the physical edges is regenerated during the forward pass, the returned gradient is not guaranteed to include the full transpose accumulation through the model-extension chain. Switching to full mode does not resolve this limitation. The gradient must not be described as the total derivative through the entire boundary-reconstruction process.
```

See the individual equation pages for more specific source units, recorded components, and stability limitations.
