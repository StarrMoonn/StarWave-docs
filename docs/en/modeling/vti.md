# Acoustic VTI: Two and Three Dimensions

`starwave.vti` is a Duveneck-type acoustic VTI interface, not a full elastic-wave interface. Its four model parameters are `vp`, `epsilon`, `delta`, and `rho`. Each can be independently selected for training through `requires_grad`; fixed parameters still participate in forward propagation.

| Model | Condition | Typical units |
|---|---|---|
| `vp` | Positive and finite | m/s |
| `rho` | Positive and finite | kg/m³ |
| `epsilon` | Finite and > -0.5 | Dimensionless |
| `delta` | Finite and > -0.5 | Dimensionless |

The axis order is `[x,z]` in 2D and `[x,y,z]` in 3D. The last axis must be vertical. Specify grid spacings in the same axis order; they may differ between axes.

## Source and recorded components

The source `source_amplitudes` is a **stress rate in Pa/s**, injected into both `sH` and `sV` by default. Time integration is performed internally. Do not multiply by `dt` again externally, and do not apply scalar's `-vp² dt²` normalization.

The default is `receiver_fields=("vz",)`, returning `vz` in m/s. Available fields in 2D are `vx,vz,sH,sV`; 3D additionally provides `vy`. Stress is measured in Pa. The returned tuple follows the order of `receiver_fields`, with each element shaped `[B,R,T]`.

```python
stress_h, velocity_z = starwave.vti(
    vp, epsilon, delta, rho,
    grid_spacing=[10.0, 10.0], dt=0.001,
    source_amplitudes=stress_rate,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    receiver_fields=("sH", "vz"),
)
```

Valid inputs must be prepared before running this call snippet. Velocities are located at positive half-grid offsets along their corresponding axes, while stresses are located at grid centers. The interface does not perform spatial interpolation for the user. Outputs are aligned to nominal user times, but this does not make stress and velocity the same physical quantity. See the {ref}`VTI API reference <vti>` for individual parameters and return ordering.

## Anisotropy and stability

The interface does not enforce `epsilon >= delta` or implicitly clip parameters. There are known stability limitations when `epsilon < delta`; reducing `dt` does not guarantee that growth will be eliminated. An explicit `max_vel` must cover the anisotropic velocity envelope. Covering only `max(vp)` is insufficient to guarantee validity.

VTI illumination, a free surface, public restart state, and higher-order automatic differentiation are not available. Validation of VTI boundary reconstruction and gradients on target GPUs, along with complete forward-modeling examples, is still pending. The [boundary-gradient limitations](conventions.md) also apply.
