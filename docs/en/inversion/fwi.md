# Simple FWI Example

Complete **25 Adam updates** on three synthetic shots while checking both data fit and model recovery. This page uses the analytic model and acquisition from [Simple Gradient Computation](../modeling/gradient.md) and documents one measured run from start to finish.

```{important}
The data objective fell **91.51%**, but active-region velocity RMSE improved only **0.86%** (30.905 → 30.639 m/s). The central lens remains substantially unresolved. Finishing an optimization loop does not establish FWI convergence or scientific acceptance.
```

```{admonition} Environment covered by this result
:class: sw-run-note
2026-10-05 · NVIDIA A30 · Python 3.10.18 · PyTorch 2.5.1 / CUDA 11.8 · StarWave **0.1.0.dev9**. This is not a validation report for the public 2.0.0 wheel, other GPUs, or multi-GPU execution.
```

## Download and run

Download {download}`03_simple_fwi.ipynb <../../examples/tutorials/03_simple_fwi.ipynb>` or its {download}`paired Python script <../../examples/tutorials/03_simple_fwi.py>`. Begin in a CUDA kernel that passed the [installation checks](../installation.md) and run every cell from the beginning. The example creates all inputs independently; no earlier notebook output is needed.

Public downloads retain the original Chinese teaching notes, with historical outputs and execution metadata cleared and privacy/editorial changes only. Download complete settings and provenance: {download}`results_summary.json <../../examples/tutorials/results_summary.json>` · {download}`source_manifest.json <../../examples/tutorials/source_manifest.json>`.

## Experiment configuration

| Item | Setting in this run |
|---|---|
| Model and active region | 96 × 64, 10 m spacing; 4,200 trainable cells, outer 6 cells and top 8 fixed |
| Observations | Same-solver true-model synthetics; 3 shots × 38 receivers × 480 samples, no added noise |
| Acquisition and source | Sources x=160/470/790 m, receivers x=100…840 m, all at z=50 m; 12 Hz Ricker |
| Time and propagation | dt=1.5 ms; accuracy=4, PML=20, boundary_buffer=5, memory="boundary" |
| Velocity constraints | 1450 < v < 2450 m/s; fixed max_vel=2600 m/s |
| Optimizer | Adam, learning rate 0.025, 25 updates; all three shots accumulated per update |
| Regularization | Squared first grid differences of the update, weight 10⁻⁴, update scale 100 m/s |
| Gradient clipping | Global norm of latent θ, threshold 1.0 |
| Full evaluations | After 0, 5, 10, 15, 20, and 25 updates |

The initial model is a smooth layered background; the truth adds a windowed Gaussian lens. See the [gradient example](../modeling/gradient.md) for the model formula, source units, and geometry. Truth is used only to generate observations and report diagnostic errors, not to guide updates or select a “best” model.

## Key points for the loop

The unconstrained latent θ maps through a sigmoid to physical velocity in active cells. Fixed cells take the initial background directly:

```{math}
v(\theta)=1450+1000\,\sigma(\theta)
\quad\text{(active cells)}.
```

Initialize `theta = logit((v_initial − 1450) / 1000)`. The data term matches the [gradient example](../modeling/gradient.md): compute observed RMS once, use the same scale for prediction and observation, and take mean SmoothL1 with beta=0.25.

```{math}
u=(v-v_0)/(100\ \mathrm{m/s}),
\qquad R=\operatorname{mean}[(\Delta_x u)^2]+
          \operatorname{mean}[(\Delta_z u)^2].
```

```{math}
J(\theta)=J_{\mathrm{data}}(v(\theta))+10^{-4}R.
```

Here Δ denotes adjacent grid differences without division by grid spacing. R is an explicit teaching-example smoothness penalty on the update, not a propagator default.

```python
# The notebook defines the complete classes, functions, and inputs.
optimizer.zero_grad(set_to_none=True)
for s in range(NSHOTS):
    model = velocity()  # A fresh model and propagation graph per shot
    predicted = propagate(
        model, source_amplitudes[s:s+1], source_locations[s:s+1],
        receiver_locations[s:s+1], CONFIG,
    )
    term = data_loss(predicted, observed[s:s+1], scale, CONFIG) / NSHOTS
    term.backward()
penalty = CONFIG["smoothness_weight"] * smoothness(velocity(), v_initial, CONFIG)
penalty.backward()
torch.nn.utils.clip_grad_norm_(velocity.parameters(), 1.0, error_if_nonfinite=True)
optimizer.step()  # One update only after accumulating all three shots
```

Each native forward has one backward call; every shot uses a fresh graph, followed by a separate regularization graph. The full notebook checks finiteness, fixed-band invariance, and velocity bounds before/after updates. It stops on failure rather than replacing NaNs or removing failed points.

## Data fit and model error

```{figure} /_static/tutorials/fwi_history.png
:alt: Six post-update evaluations of data and total objectives and active velocity RMSE, which rises before declining
:class: sw-science-image
:figclass: sw-science-figure

Six full evaluation states against completed optimizer updates. Lines connect measured points; intermediate states are not measured here. Curves are not rescaled by their starting values. Model RMSE rises to about 32.636 m/s at update 10 before falling.
```

| Metric | Initial (0 updates) | Final (25 updates) | Change |
|---|---:|---:|---:|
| Normalized SmoothL1 data term | 2.47999065 × 10⁻⁴ | 2.10452727 × 10⁻⁵ | 91.51% decrease |
| Data + weighted regularization | 2.47999065 × 10⁻⁴ | 2.14042969 × 10⁻⁵ | Data and total objective recorded separately |
| Relative data L2 | 0.0111355064 | 0.0032438610 | 70.87% decrease |
| Active velocity RMSE (m/s) | 30.9053688 | 30.6389942 | Only 0.2663745 lower (0.86%) |
| Maximum fixed-band velocity change (m/s) | 0 | 0 | Remains fixed |

Relative data L2 is ‖prediction−observation‖₂/‖observation‖₂. Model RMSE is computed against synthetic truth over the 4,200 active cells only; field data generally provide no such ground-truth metric. The final records, final model, and update-25 evaluation describe the same state, not an iterate selected using true-model error.

## Recovered model and residuals

```{figure} /_static/tutorials/fwi_models.png
:alt: True, initial, and update-25 velocity models on identical physical velocity scales
:class: sw-science-image
:figclass: sw-science-figure

Truth, initial background, and final model use the same velocity range. A similar large-scale background does not establish lens recovery; read these together with the error maps below.
```

```{figure} /_static/tutorials/fwi_model_errors.png
:alt: Initial-minus-true and final-minus-true model errors on a shared full symmetric scale
:class: sw-science-image
:figclass: sw-science-figure

Model error = current velocity − truth. Both panels share the full symmetric color range, without clipping or independent normalization. Substantial central-lens error remains.
```

```{figure} /_static/tutorials/fwi_residuals.png
:alt: First-shot initial and final data residuals on a residual-only symmetric scale shared by all shots
:class: sw-science-image
:figclass: sw-science-figure

First shot (source x = 160 m). Data residual = prediction − observation. All three shots below and both states share the full residual range. This residual-only scale differs from the full-gather amplitude scale on the gradient page; color intensity cannot be compared directly between them.
```

```{figure} /_static/tutorials/fwi_residuals_shot_1.png
:alt: Second-shot initial and final residuals with the common residual scale
:class: sw-science-image
:figclass: sw-science-figure

The same residual scale is used for every shot; this is the second shot (source x = 470 m).
```

```{figure} /_static/tutorials/fwi_residuals_shot_2.png
:alt: Third-shot initial and final residuals with the common residual scale
:class: sw-science-image
:figclass: sw-science-figure

The third shot (source x = 790 m), still using the same full residual scale.
```

## Validation limits

This run shows that the three-shot synthetic optimization workflow completes in the recorded environment and substantially improves data fit. Model-recovery improvement is weak: **it does not establish a high-quality inversion or convergence**.

- No directional/finite-difference check was run; complete adjoint correctness is unverified.
- Same solver, same grid, and no added noise exclude field data, source error, and modeling error.
- One frequency band, limited acquisition, and 25 updates bound the interpretation; their individual effects were not isolated.
- Fixing outer cells does not repair [known extension-chain/PML limitations](../modeling/conventions.md). Multi-GPU, AMP, and higher derivatives were not tested.
- No GPU timings or speedups are reported. Saved model snapshots omit Adam momentum and are not exact-resume checkpoints.

## Retained one-update check

The original small wiring script remains available, including its old links. It uses a separate 32 × 32 configuration and is not the source of the three-shot results above:

```bash
python scalar_demo.py --mode fwi --device 0
```

Download it from [Quickstart](../quickstart.md). Further experiments should first add interior-perturbation step-size scans and directional checks, then vary frequency, acquisition, or optimization settings separately. Those extensions have not been run on this page.
