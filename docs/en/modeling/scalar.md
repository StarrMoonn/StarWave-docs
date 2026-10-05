# Scalar Acoustics

`starwave.scalar` accepts a 2D velocity model `v` and returns a single-element tuple `(receiver_amplitudes,)`. Use `[0]` to obtain the pressure-like records of shape `[B,R,T]`.

The model permits finite positive, zero, and negative values. This does not establish that all such values are physically meaningful; conventional velocity models usually use positive values. Automatic CFL planning uses `max(abs(v))`. An all-zero model requires an explicit positive `max_vel`.

## Source and call

The public source input is a normalized forcing term satisfying `u_tt = v² (Lap(u) - f)`. Supply fixed `f`; do not additionally multiply it by `-v² dt²` yourself. If `u` is measured in Pa and length in m, the units of `f` are Pa/m².

```python
records = starwave.scalar(
    v, grid_spacing=10.0, dt=0.001,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
)[0]
```

This snippet requires prepared inputs. For a complete standalone program, see the [Quickstart](../quickstart.md). For individual parameters and defaults, see the [scalar API reference](../api/scalar.md).

## Gradients and illumination

Set `v.requires_grad=True`, construct a scalar loss from the records, and call `backward()` once. The velocity factor at the source location is also part of the model-gradient chain; the source waveform itself is not trainable. The optional `ScalarIllumination` applies only to scalar, and its statistics must not be described as an exact Hessian. A complete usage tutorial is not yet included in this first version.

Reducing the time step does not resolve inadequate spatial sampling. Before inversion, separately check waveforms, spatial resolution, gradients, and the [boundary-gradient limitations](conventions.md).
