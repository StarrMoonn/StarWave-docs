# VRZ: Velocity and Impedance or Density

`starwave.vrz` provides 2D variable-density acoustic propagation. Its model parameters are positive, finite velocity `v` and **exactly one** of `impedance` (impedance Z) or `density` (density rho). Do not supply both or omit both. In a consistent unit system, `Z = v * rho`.

```python
records = starwave.vrz(
    v, grid_spacing=10.0, dt=0.001,
    density=rho,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
)[0]
```

`rho` and `v` should have the same shape, dtype, and device. For the alternative parameterization, replace `density=rho` with `impedance=Z`. This is a call snippet; inputs and the native library must be prepared beforehand. See the {ref}`VRZ API reference <vrz>` for full parameter descriptions.

## Similarities to and differences from scalar

VRZ has the same acquisition input shapes, equal-spacing restriction, and single-element return tuple as scalar. The source is still a normalized forcing term, with velocity-dependent scaling applied internally. VRZ requires positive model values; scalar's allowance for signed and zero values does not apply.

Gradients are returned for the parameterization selected in the call. Training `v` and `Z` and training `v` and `rho` are different optimization problems. Do not interchange these gradients without applying the chain rule.

## Limitations

Positivity checks, numerical representability checks, and CFL planning do not guarantee spatial stability for arbitrarily strong impedance contrasts. Near-zero values and strong contrasts can cause ill-conditioning. VRZ does not provide illumination. The `illumination` argument in the signature is retained only for compatibility and must be `None`. Numerical, gradient, and FWI validation of VRZ on actual devices is still pending.
