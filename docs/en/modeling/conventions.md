# Model, Acquisition, and Time Conventions

## Tensor shapes

`B` is the number of shots, `S` the number of sources per shot, `R` the number of receivers per shot, `T` the number of user time samples, and `D` the number of spatial dimensions.

| Input/output | Shape | Convention |
|---|---|---|
| scalar model | `[N0,N1]` or `[N0,N1,N2]` | CUDA float32; `v.ndim` selects the dimension |
| VRZ model | `[N0,N1]` | CUDA float32 |
| VTI model | `[nx,nz]` or `[nx,ny,nz]` | The vertical axis is always last |
| `source_amplitudes` | `[B,S,T]` | 3D scalar supports first-order source gradients; 2D scalar, VRZ, and VTI sources remain fixed |
| `source_locations` | `[B,S,D]` | Integer grid indices; torch.long is recommended |
| `receiver_locations` | `[B,R,D]` | Integer grid indices |
| Each recorded tensor | `[B,R,T]` | CUDA float32 |

Coordinates directly index the supplied physical model. They are **not coordinates in meters and do not include a PML offset**. Scalar / VRZ do not infer the user's physical axis order. In 3D scalar, `[i,j,k]` directly indexes `v[i,j,k]`. If the model is stored as `[nz,nx]` but acquisition coordinates use `[x,z]`, transpose the model explicitly first. All four VTI models must share the same axis order, shape, device, and dtype.

## Sampling and absorbing boundaries

`grid_spacing` and `dt` must be positive; typical units are meters and seconds. Internally, smaller time steps are planned according to the CFL condition, but the returned length remains `T`, with nominal user times `0, dt, ..., (T-1)dt`. 2D scalar / VRZ require equal spacing; 3D scalar and VTI allow per-axis spacing in model-axis order.

`pml_freq` is used for PML configuration and spatial-sampling diagnostics. It is not a function that automatically generates a source waveform. `pml_width` is a positive integer thickness in grid cells, equal on every face. Setting it to zero to request a free surface is not supported.

`boundary_buffer` is measured in spatial grid cells and defaults to 5. Boundary mode requires at least `accuracy // 2 + 1`; full mode allows 0, but still retains the configured boundary padding.

## Memory and automatic differentiation

The scalar, VRZ, and VTI equations on this page support `memory="boundary"` (the default) or `"full"`. Full mode stores the time history and may use more GPU memory. There is no automatic mode switching. The 3D scalar Radius-M face history, terminal fields, and workspaces are described in {ref}`Reconstruction <reconstruction-scalar3d>`; source-only gradients also retain history. Each forward pass supports only one first-order backward pass. CPU propagation, AMP, higher-order automatic differentiation, custom CUDA streams, and a public checkpoint/restart state interface are currently unsupported.

```{warning}
Model gradients are conditional on fixed extended-model, CFL, and PML settings. When replicate padding at the physical edges is regenerated during the forward pass, the returned gradient is not guaranteed to include the full transpose accumulation through the model-extension chain. Switching to full mode does not resolve this limitation. The gradient must not be described as the total derivative through the entire boundary-reconstruction process.
```

See the individual equation pages for more specific source units, recorded components, and stability limitations.
