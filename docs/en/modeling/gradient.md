# Simple Gradient Computation

Start with an analytic velocity model, generate three synthetic observed shots, and differentiate the data objective with respect to velocity. This example follows the complete **model → forward → loss → backward** chain. It needs no external seismic data or outputs from another notebook.

```{admonition} Executed teaching experiment · 2026-10-05
:class: sw-run-note
NVIDIA A30 · Python 3.10.18 · PyTorch 2.5.1 / CUDA 11.8 · StarWave **0.1.0.dev9**. Figures and metrics below come from returned server arrays, not anticipated results. This run does not validate the public 2.0.0 wheel on that device.
```

## Download and run

Download {download}`02_forward_and_gradient.ipynb <../../examples/tutorials/02_forward_and_gradient.ipynb>` or its {download}`paired Python script <../../examples/tutorials/02_forward_and_gradient.py>`. Complete [installation and the small smoke test](../installation.md), select the matching CUDA kernel, and run every cell from the beginning.

Each notebook includes the model, geometry, Ricker wavelet, propagation wrapper, objective, plots, and exports. Public downloads have empty outputs and cleared execution metadata. They retain the original Chinese teaching notes; numerical computation cells and parameters are unchanged. See the {ref}`provenance record <tutorial-evidence>` for privacy-only edits.

## Model and acquisition design

The `[x,z]` model has **96 × 64** cells with **10 m** spacing. The background combines a linear depth gradient with a smooth layer transition:

```{math}
v_0(z)=1600+0.55z+220\,\sigma((z-350)/28).
```

Velocity is in m/s, depth in m, and σ is the sigmoid. The true model adds a positive Gaussian lens centered at **(500, 260) m**, with amplitude parameter 180 m/s and horizontal/vertical standard deviations 110/65 m. A squared-sine window tapers the anomaly smoothly to zero at the active-region edges. The 180 m/s value is the unwindowed parameter, not the final anomaly peak.

The outermost 6 cells and top 8 cells are fixed, leaving **4,200 active cells**. Truth and initial background are identical in the fixed band. There is no water layer or free surface.

```{figure} /_static/tutorials/gradient_models.png
:alt: Analytic true and initial background velocity models on a shared velocity scale with x and depth in metres
:class: sw-science-image
:figclass: sw-science-figure

Analytic truth and starting background on the same velocity scale. Depth increases downward. These plots were redrawn from verified numerical arrays.
```

| Setting | Value in this run |
|---|---|
| Sources | x = 160, 470, 790 m; z = 50 m, one source per shot |
| Receivers | x = 100…840 m every 20 m; z = 50 m, 38 receivers |
| Time | dt = 1.5 ms, 480 samples, last sample at 0.7185 s |
| Source | 12 Hz Ricker, peak at 0.10 s, fixed forcing; arbitrary amplitude units |
| Propagation | float32, accuracy=4, PML=20 cells, boundary_buffer=5 |
| Memory and speed bound | memory="boundary", fixed max_vel=2600 m/s |
| Record shape | `[3,38,480]` = `[shot,receiver,time]` |

```{figure} /_static/tutorials/acquisition_source.png
:alt: Active region, three sources, receivers, and fixed Ricker forcing with metre and second axes
:class: sw-science-image
:figclass: sw-science-figure

Acquisition points lie in the shallow fixed band. The waveform is fixed forcing; do not multiply it again by the propagator's internal −v² dt² injection coefficient. Recorded amplitudes are not calibrated Pa.
```

## Objective and automatic differentiation

Observed data are generated once by the same solver on the same grid using the true model, then held fixed. One global observed RMS, **S = 1.9230548143**, scales both prediction and observation. Do not normalize the two separately or normalize individual traces.

```{math}
r=(d_{\mathrm{pred}}-d_{\mathrm{obs}})/S,
\qquad J_{\mathrm{data}}=\operatorname{mean}\ell_{0.25}(r).
```

```{math}
\ell_\beta(r)=
\begin{cases}
 r^2/(2\beta),& |r|<\beta,\\
 |r|-\beta/2,& |r|\ge\beta.
\end{cases}
```

This is PyTorch `SmoothL1Loss(beta=0.25)`, equivalent to `HuberLoss(delta=0.25) / 0.25`. Velocity itself is the parameter, with the fixed band excluded through `torch.where`. Each shot gets a fresh graph and one backward call; dividing its objective by the shot count accumulates the global mean for the three equally sized shots:

```python
# See the download for full definitions; observed and scale are fixed.
v_parameter = torch.nn.Parameter(v_initial.clone())
for s in range(NSHOTS):
    model = torch.where(active, v_parameter, v_initial)
    predicted = propagate(
        model, source_amplitudes[s:s+1], source_locations[s:s+1],
        receiver_locations[s:s+1], CONFIG,
    )
    term = data_loss(predicted, observed[s:s+1], scale, CONFIG) / NSHOTS
    term.backward()
gradient = v_parameter.grad
```

This example computes a gradient without an optimizer update. Its sign describes the first-order effect of increasing local velocity on the objective; it is not the final velocity correction.

## Measured records and gradient

```{figure} /_static/tutorials/gradient_gathers.png
:alt: Second-shot observation, initial prediction, and predicted-minus-observed residual on one full symmetric amplitude scale
:class: sw-science-image
:figclass: sw-science-figure

Observation, initial prediction, and residual for the second shot (x = 470 m). All three share the full symmetric amplitude range, without independent normalization or clipping. The initial data difference is small, so its residual is faint on this scale.
```

```{figure} /_static/tutorials/gradient_velocity.png
:alt: Signed objective gradient with respect to velocity on a full symmetric scale, zero in the fixed band
:class: sw-science-image
:figclass: sw-science-figure

Signed velocity gradient on its full symmetric scale. Weak local structure has not been separately amplified; the fixed-band gradient is zero.
```

| Metric | Measured value |
|---|---:|
| Normalized SmoothL1 data objective | 2.47999068 × 10⁻⁴ |
| Relative data L2, ‖prediction−observation‖₂ / ‖observation‖₂ | 0.0111355064 |
| Velocity-gradient L2 norm | 2.01557941 × 10⁻⁶ |
| Active-region maximum absolute gradient | 1.27708390 × 10⁻⁷ |
| Fixed-band maximum absolute gradient | 0 |

## Conclusions and limits

The run produced finite, nonzero forward records and a first-order velocity gradient, with zero gradient in the fixed band. **The directional check was disabled; no finite-difference or complete adjoint-correctness evidence is supplied.** Same-solver synthetic data exclude noise, source uncertainty, and modeling error, so this is not evidence of field-data performance.

Known outer-extension-chain and PML limitations still apply; see [model conventions](conventions.md). Fixing a boundary band is an experiment constraint, not a repair of those limitations. Continue to the [Simple FWI Example](../inversion/fwi.md) on the same problem, checking both data fit and model recovery.
