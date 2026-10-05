# Acquisition

Acquisition connects sources, waveforms, and receivers to a physical model. This guide builds **two shots, one source per shot, and seven receivers per shot** on a 32×32 grid, then passes the inputs to `starwave.scalar`.

## Shots, sources, and receivers

A shot is an independent propagation experiment. A batch can contain several shots; a single shot can also contain several sources, each with its own waveform, contributing to that shot's propagation.

| Tensor | 2D shape | This example |
|---|---|---|
| `source_locations` | `[B,S,2]` | `[2,1,2]` |
| `source_amplitudes` | `[B,S,T]` | `[2,1,256]` |
| `receiver_locations` | `[B,R,2]` | `[2,7,2]` |
| Returned recording tensor | `[B,R,T]` | `[2,7,256]` |

`B` is the number of shots, `S` the sources per shot, `R` the receivers per shot, and `T` the time samples. For one source per shot, scalar also accepts source locations shaped `[B,2]`; this example keeps the explicit source axis. Coincident sources within a shot have their increments summed, and duplicate receiver positions are allowed.

## Coordinates and the physical grid

Locations are **integer indices into the physical model**, preferably `torch.long`. They are not coordinates in meters and do not include a PML offset. Every index must be within the model bounds.

This example chooses `v.shape == (nx,nz)`, with axis 0 representing x and axis 1 representing z, so `[10,10]` directly indexes `v[10,10]`. This is an example convention; scalar does not infer physical directions. With an origin at `(0 m,0 m)` and 10 m grid spacing, that location is `(100 m,100 m)`. For off-grid positions, choose a discretization explicitly rather than silently truncating with an integer conversion.

```{figure} /_static/diagrams/acquisition.svg
:alt: A 32×32 physical grid with axis 0, x, to the right and axis 1, z, downward. Shot 0 has its source at (10,10); shot 1 has its source at (22,10). Each shot has seven receivers at z=6 and x=4,8,12,16,20,24,28.
:class: sw-science-figure

Both shots use the same receiver spread. The two source positions are shown together for comparison; each shot excites only its own source. All numbers are grid indices, and PML is not shown.
```

See [Model, Acquisition, and Time Conventions](conventions.md) for the general axis, sampling, and boundary rules.

## Construct geometry and waveforms

The following block constructs inputs on the CPU using only PyTorch; it does not propagate waves. Both shots share a receiver spread and an explicitly defined Ricker waveform, with `repeat` creating their own tensor entries.

```python
import math
import torch

nx, nz = 32, 32
B, S, R, T = 2, 1, 7, 256
grid_spacing, dt, frequency = 10.0, 0.001, 15.0

# Chosen axis order: [x, z]. Each shot has one source.
source_locations = torch.tensor(
    [[[10, 10]], [[22, 10]]], dtype=torch.long
)

# The same receiver line is explicitly repeated for both shots.
receiver_x = torch.arange(4, 29, 4, dtype=torch.long)
receiver_line = torch.stack(
    (receiver_x, torch.full_like(receiver_x, 6)), dim=-1
)
receiver_locations = receiver_line[None, :, :].repeat(B, 1, 1)

# Fixed scalar forcing waveform, with time in seconds.
time = torch.arange(T, dtype=torch.float32) * dt
a = (math.pi * frequency * (time - 0.08)) ** 2
wavelet = (1.0 - 2.0 * a) * torch.exp(-a)
source_amplitudes = wavelet[None, None, :].repeat(B, S, 1)

assert source_locations.shape == (B, S, 2)
assert receiver_locations.shape == (B, R, 2)
assert source_amplitudes.shape == (B, S, T)
for locations in (source_locations, receiver_locations):
    assert torch.all((0 <= locations[..., 0]) & (locations[..., 0] < nx))
    assert torch.all((0 <= locations[..., 1]) & (locations[..., 1] < nz))
```

For scalar, `source_amplitudes` is the fixed normalized forcing `f`, not a per-step pressure increment; do not multiply it by `-v² dt²` yourself. The waveform amplitude here is an illustrative choice. See [Usage](../usage.md) for source units, time handling, and differences between propagators.

## Connect to the propagator

Prepare a CUDA device and native libraries using [Installation](../installation.md) and [Quickstart](../quickstart.md), then continue with this block. It reuses the variables above and puts the model, coordinates, and waveform on the current CUDA device.

```python
import starwave

device = torch.device("cuda")
v = torch.full((nx, nz), 1800.0, dtype=torch.float32, device=device)

with torch.no_grad():
    records, = starwave.scalar(
        v, grid_spacing=grid_spacing, dt=dt,
        source_amplitudes=source_amplitudes.to(device),
        source_locations=source_locations.to(device),
        receiver_locations=receiver_locations.to(device),
        pml_freq=frequency,
    )

assert records.shape == (B, R, T)
```

`records[b,r,:]` is the time series for receiver `r` in shot `b`. The expected shape is `(2,7,256)` according to the input contract; this page does not report a GPU run of this two-shot example. See [Usage](../usage.md) for the full parameters, defaults, and runtime limits.
