# Surface-only anomaly

The numerical consistency checks pass, but the anomaly interior is poorly recovered. This 3D comparison retains the incomplete reconstruction and its measured metrics: a large loss reduction does not establish full recovery of the model.

This page shows saved results from the actual CUDA runs of 2026-10-08. See the [Scalar3D overview](index.md) for shared parameters, the objective and validation scope.

## Experiment setup

| Parameter | Measured configuration / result |
|---|---|
| Grid (X × Y × Z) | 80 × 80 × 50 |
| Shots / receivers per shot | 16 / 441 |
| dx / dt / nt | 10 m / 0.001 s / 640 |
| Ricker / accuracy / PML / buffer | 25 Hz / 4 / 12 / 5 |
| memory / step_ratio | boundary / 1 |
| Adam lr / epochs | 10.0 / 200 |
| Training time | 707.66 s |
| Whole-model RMSE (m/s) | 25.305 → 23.276 |
| Fixed-model objective decrease | 96.4050% |

## Acquisition geometry

The 16 shots form a 4 × 4 grid and the 441 receivers a 21 × 21 grid, all at depth 20 m inside the model. There are no side or bottom sensors. PML surrounds every face; a top-plane survey does not imply a pressure-release free surface.

```{raw} html
<figure class="sw-example-figure" id="figure-acquisition">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Scrollable scientific figure">
    <a href="../_static/scalar3d/surface/acquisition.png" target="_blank" rel="noopener" aria-label="Open the full-resolution figure"><img src="../_static/scalar3d/surface/acquisition.png" alt="3D acquisition geometry and a top-view projection along depth. Red stars mark sources and blue points receivers; coordinates are in metres." loading="lazy"></a>
  </div>
  <figcaption>3D acquisition geometry and a top-view projection along depth. Red stars mark sources and blue points receivers; coordinates are in metres. <a href="../_static/scalar3d/surface/acquisition.png" target="_blank" rel="noopener">Open the full-resolution figure ↗</a></figcaption>
</figure>
```

## True, initial and recovered models

Both background and initial model are 2000 m/s. The anomaly is centred at (400, 400, 180) m, with a 70 m core and a cosine taper out to 100 m. The true central velocity is 2300 m/s, but the recovered centre is 1948.37 m/s. Anomaly-region RMSE only decreases from 227.50 to 207.96 m/s. The result mainly recovers scattering interfaces and fails to recover the positive velocity contrast in the interior.

```{raw} html
<figure class="sw-example-figure" id="figure-model-3d">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Scrollable scientific figure">
    <a href="../_static/scalar3d/surface/model_3d.png" target="_blank" rel="noopener" aria-label="Open the full-resolution figure"><img src="../_static/scalar3d/surface/model_3d.png" alt="Three orthogonal sections through the true, initial and final Vp volumes, with one shared velocity scale (m/s). These are sections of the 3D volume, not a full volume rendering. The final model is not smoothed." loading="lazy"></a>
  </div>
  <figcaption>Three orthogonal sections through the true, initial and final Vp volumes, with one shared velocity scale (m/s). These are sections of the 3D volume, not a full volume rendering. The final model is not smoothed. <a href="../_static/scalar3d/surface/model_3d.png" target="_blank" rel="noopener">Open the full-resolution figure ↗</a></figcaption>
</figure>
```

```{raw} html
<figure class="sw-example-figure" id="figure-model-comparison">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Scrollable scientific figure">
    <a href="../_static/scalar3d/surface/model_comparison.png" target="_blank" rel="noopener" aria-label="Open the full-resolution figure"><img src="../_static/scalar3d/surface/model_comparison.png" alt="Sections in three directions compare truth, initial model and final result column by column. Physical aspect ratios and a shared velocity scale are preserved so geometric stretching does not hide reconstruction errors." loading="lazy"></a>
  </div>
  <figcaption>Sections in three directions compare truth, initial model and final result column by column. Physical aspect ratios and a shared velocity scale are preserved so geometric stretching does not hide reconstruction errors. <a href="../_static/scalar3d/surface/model_comparison.png" target="_blank" rel="noopener">Open the full-resolution figure ↗</a></figcaption>
</figure>
```

## Optimization and convergence

The outer shell of three grid cells of known background are fixed, interior voxels are independently optimized, and Vp is bounded to 1700–2500 m/s. No centre/shape constraint or gradient/model smoothing is applied. Each epoch has four four-shot batches. Limited illumination, the 25 Hz band and the homogeneous initial model are possible contributors, but their separate effects have not been established by controlled tests.

```{raw} html
<figure class="sw-example-figure" id="figure-convergence">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Scrollable scientific figure">
    <a href="../_static/scalar3d/surface/convergence.png" target="_blank" rel="noopener" aria-label="Open the full-resolution figure"><img src="../_static/scalar3d/surface/convergence.png" alt="Training-pass means and fixed-model, all-shot objectives are recorded separately. RMSE uses the synthetic truth for evaluation and is not optimized. The model changes within a training pass, so the two objective curves need not coincide." loading="lazy"></a>
  </div>
  <figcaption>Training-pass means and fixed-model, all-shot objectives are recorded separately. RMSE uses the synthetic truth for evaluation and is not optimized. The model changes within a training pass, so the two objective curves need not coincide. <a href="../_static/scalar3d/surface/convergence.png" target="_blank" rel="noopener">Open the full-resolution figure ↗</a></figcaption>
</figure>
```

## Gradients at saved models

The checkpoints are epochs 0, 20, 60, 100, 200. Each figure comes from an all-shot recomputation at a fixed saved model, rather than a single training batch. Raw and masked gradients are saved separately and recomputed losses agree with the original fixed-model evaluations. This post-processing time is excluded from the training duration.

```{raw} html
<figure class="sw-example-figure" id="figure-gradients-normalized">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Scrollable scientific figure">
    <a href="../_static/scalar3d/surface/gradients_normalized.png" target="_blank" rel="noopener" aria-label="Open the full-resolution figure"><img src="../_static/scalar3d/surface/gradients_normalized.png" alt="All-shot dJ/dVp recomputed at saved models, with the known-region mask and no preconditioning or smoothing. Each volume is divided by its own max|g|, retained in the title. Colours compare shape, not absolute amplitude." loading="lazy"></a>
  </div>
  <figcaption>All-shot dJ/dVp recomputed at saved models, with the known-region mask and no preconditioning or smoothing. Each volume is divided by its own max|g|, retained in the title. Colours compare shape, not absolute amplitude. <a href="../_static/scalar3d/surface/gradients_normalized.png" target="_blank" rel="noopener">Open the full-resolution figure ↗</a></figcaption>
</figure>
```

```{raw} html
<figure class="sw-example-figure" id="figure-gradients-shared-scale">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Scrollable scientific figure">
    <a href="../_static/scalar3d/surface/gradients_shared_scale.png" target="_blank" rel="noopener" aria-label="Open the full-resolution figure"><img src="../_static/scalar3d/surface/gradients_shared_scale.png" alt="The same checkpoint gradients use one shared, unnormalized scale for absolute-amplitude comparison. Later gradients appear faint because their magnitude is smaller, not because arrays were discarded or independently rescaled." loading="lazy"></a>
  </div>
  <figcaption>The same checkpoint gradients use one shared, unnormalized scale for absolute-amplitude comparison. Later gradients appear faint because their magnitude is smaller, not because arrays were discarded or independently rescaled. <a href="../_static/scalar3d/surface/gradients_shared_scale.png" target="_blank" rel="noopener">Open the full-resolution figure ↗</a></figcaption>
</figure>
```

## Shot gathers and residuals

```{raw} html
<figure class="sw-example-figure" id="figure-shot-gather">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Scrollable scientific figure">
    <a href="../_static/scalar3d/surface/shot_gather.png" target="_blank" rel="noopener" aria-label="Open the full-resolution figure"><img src="../_static/scalar3d/surface/shot_gather.png" alt="Shot 0 observations, initial prediction, final prediction and final residual share the same amplitude scale. The horizontal axis is receiver index and the vertical axis is time. Read the gather fit together with model error." loading="lazy"></a>
  </div>
  <figcaption>Shot 0 observations, initial prediction, final prediction and final residual share the same amplitude scale. The horizontal axis is receiver index and the vertical axis is time. Read the gather fit together with model error. <a href="../_static/scalar3d/surface/shot_gather.png" target="_blank" rel="noopener">Open the full-resolution figure ↗</a></figcaption>
</figure>
```

## Numerical checks

These are relative L2 errors recorded in the report. Every listed check is within its preset tolerance. The directional test covers one interior direction, not derivatives of padded edge parameters or arbitrary configurations.

| Check | Spatial order | Record error | Gradient / derivative error |
|---|---:|---:|---:|
| full / boundary | 2 | 0 | 5.542873980e-07 |
| full / boundary | 4 | 0 | 6.898225014e-07 |
| full / boundary | 6 | 0 | 6.311192622e-07 |
| full / boundary | 8 | 0 | 6.078466607e-07 |
| Central finite difference / gradient inner product | 4 | — | 7.054908057e-05 |
| One GPU / four GPUs | 4 | 0 | 5.777887630e-08 |

See {ref}`validation scope <validation-scope>` for definitions and tolerances.

## Downloads and reproduction

Use [Downloads and reproduction](reproduce.md) for the complete report, notebooks, scripts and numerical-result documentation. Each figure opens at its original resolution; the documentation does not redraw or cosmetically modify the inversion result.
