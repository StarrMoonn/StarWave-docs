# Undulating thin layers

This lightweight 3D supplementary case has only 10 depth samples. After 100 epochs the whole-model RMSE falls by about 20%. It illustrates a layered model, a Gaussian-smoothed starting model and gradient evolution, without claiming high-accuracy reconstruction.

This page shows saved results from the actual CUDA runs of 2026-10-08. See the [Scalar3D overview](index.md) for shared parameters, the objective and validation scope.

## Experiment setup

| Parameter | Measured configuration / result |
|---|---|
| Grid (X × Y × Z) | 50 × 50 × 10 |
| Shots / receivers per shot | 8 / 256 |
| dx / dt / nt | 10 m / 0.001 s / 400 |
| Ricker / accuracy / PML / buffer | 25 Hz / 4 / 12 / 5 |
| memory / step_ratio | boundary / 1 |
| Adam lr / epochs | 10.0 / 100 |
| Training time | 77.10 s |
| Whole-model RMSE (m/s) | 84.704 → 67.714 |
| Fixed-model objective decrease | 99.8340% |

## Acquisition geometry

Eight sources and 256 receivers per shot sample this thin 3D region; their positions are shown below. Its vertical extent is much smaller than the horizontal extents. Model views and sections preserve the physical aspect ratio. All boundaries use PML.

```{raw} html
<figure class="sw-example-figure" id="figure-acquisition">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Scrollable scientific figure">
    <a href="../_static/scalar3d/layered/acquisition.png" target="_blank" rel="noopener" aria-label="Open the full-resolution figure"><img src="../_static/scalar3d/layered/acquisition.png" alt="3D acquisition geometry and a top-view projection along depth. Red stars mark sources and blue points receivers; coordinates are in metres." loading="lazy"></a>
  </div>
  <figcaption>3D acquisition geometry and a top-view projection along depth. Red stars mark sources and blue points receivers; coordinates are in metres. <a href="../_static/scalar3d/layered/acquisition.png" target="_blank" rel="noopener">Open the full-resolution figure ↗</a></figcaption>
</figure>
```

## True, initial and recovered models

The three layers have velocities of 1600, 2000 and 2350 m/s, with two interfaces undulating in x and y. The initial model is obtained by Gaussian smoothing of the truth, with sigma = (1.35, 1, 1) in saved-array (Z, Y, X) order. This starting model uses the synthetic truth and is not a general recipe for an unknown field model.

```{raw} html
<figure class="sw-example-figure" id="figure-model-3d">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Scrollable scientific figure">
    <a href="../_static/scalar3d/layered/model_3d.png" target="_blank" rel="noopener" aria-label="Open the full-resolution figure"><img src="../_static/scalar3d/layered/model_3d.png" alt="Three orthogonal sections through the true, initial and final Vp volumes, with one shared velocity scale (m/s). These are sections of the 3D volume, not a full volume rendering. The final model is not smoothed." loading="lazy"></a>
  </div>
  <figcaption>Three orthogonal sections through the true, initial and final Vp volumes, with one shared velocity scale (m/s). These are sections of the 3D volume, not a full volume rendering. The final model is not smoothed. <a href="../_static/scalar3d/layered/model_3d.png" target="_blank" rel="noopener">Open the full-resolution figure ↗</a></figcaption>
</figure>
```

```{raw} html
<figure class="sw-example-figure" id="figure-model-comparison">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Scrollable scientific figure">
    <a href="../_static/scalar3d/layered/model_comparison.png" target="_blank" rel="noopener" aria-label="Open the full-resolution figure"><img src="../_static/scalar3d/layered/model_comparison.png" alt="Sections in three directions compare truth, initial model and final result column by column. Physical aspect ratios and a shared velocity scale are preserved so geometric stretching does not hide reconstruction errors." loading="lazy"></a>
  </div>
  <figcaption>Sections in three directions compare truth, initial model and final result column by column. Physical aspect ratios and a shared velocity scale are preserved so geometric stretching does not hide reconstruction errors. <a href="../_static/scalar3d/layered/model_comparison.png" target="_blank" rel="noopener">Open the full-resolution figure ↗</a></figcaption>
</figure>
```

## Optimization and convergence

The top two depth grid planes (z=0 and z=1) are fixed and the remaining voxels are independently optimized, with Vp bounded to 1400–2600 m/s. Each epoch has two four-shot batches. A small waveform mismatch does not imply that all interfaces and layer velocities are recovered; inspect the thin-layer sections alongside RMSE.

```{raw} html
<figure class="sw-example-figure" id="figure-convergence">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Scrollable scientific figure">
    <a href="../_static/scalar3d/layered/convergence.png" target="_blank" rel="noopener" aria-label="Open the full-resolution figure"><img src="../_static/scalar3d/layered/convergence.png" alt="Training-pass means and fixed-model, all-shot objectives are recorded separately. RMSE uses the synthetic truth for evaluation and is not optimized. The model changes within a training pass, so the two objective curves need not coincide." loading="lazy"></a>
  </div>
  <figcaption>Training-pass means and fixed-model, all-shot objectives are recorded separately. RMSE uses the synthetic truth for evaluation and is not optimized. The model changes within a training pass, so the two objective curves need not coincide. <a href="../_static/scalar3d/layered/convergence.png" target="_blank" rel="noopener">Open the full-resolution figure ↗</a></figcaption>
</figure>
```

## Gradients at saved models

The checkpoints are epochs 0, 10, 30, 60, 100. Each figure comes from an all-shot recomputation at a fixed saved model, rather than a single training batch. Raw and masked gradients are saved separately and recomputed losses agree with the original fixed-model evaluations. This post-processing time is excluded from the training duration.

```{raw} html
<figure class="sw-example-figure" id="figure-gradients-normalized">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Scrollable scientific figure">
    <a href="../_static/scalar3d/layered/gradients_normalized.png" target="_blank" rel="noopener" aria-label="Open the full-resolution figure"><img src="../_static/scalar3d/layered/gradients_normalized.png" alt="All-shot dJ/dVp recomputed at saved models, with the known-region mask and no preconditioning or smoothing. Each volume is divided by its own max|g|, retained in the title. Colours compare shape, not absolute amplitude." loading="lazy"></a>
  </div>
  <figcaption>All-shot dJ/dVp recomputed at saved models, with the known-region mask and no preconditioning or smoothing. Each volume is divided by its own max|g|, retained in the title. Colours compare shape, not absolute amplitude. <a href="../_static/scalar3d/layered/gradients_normalized.png" target="_blank" rel="noopener">Open the full-resolution figure ↗</a></figcaption>
</figure>
```

```{raw} html
<figure class="sw-example-figure" id="figure-gradients-shared-scale">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Scrollable scientific figure">
    <a href="../_static/scalar3d/layered/gradients_shared_scale.png" target="_blank" rel="noopener" aria-label="Open the full-resolution figure"><img src="../_static/scalar3d/layered/gradients_shared_scale.png" alt="The same checkpoint gradients use one shared, unnormalized scale for absolute-amplitude comparison. Later gradients appear faint because their magnitude is smaller, not because arrays were discarded or independently rescaled." loading="lazy"></a>
  </div>
  <figcaption>The same checkpoint gradients use one shared, unnormalized scale for absolute-amplitude comparison. Later gradients appear faint because their magnitude is smaller, not because arrays were discarded or independently rescaled. <a href="../_static/scalar3d/layered/gradients_shared_scale.png" target="_blank" rel="noopener">Open the full-resolution figure ↗</a></figcaption>
</figure>
```

## Shot gathers and residuals

```{raw} html
<figure class="sw-example-figure" id="figure-shot-gather">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Scrollable scientific figure">
    <a href="../_static/scalar3d/layered/shot_gather.png" target="_blank" rel="noopener" aria-label="Open the full-resolution figure"><img src="../_static/scalar3d/layered/shot_gather.png" alt="Shot 0 observations, initial prediction, final prediction and final residual share the same amplitude scale. The horizontal axis is receiver index and the vertical axis is time. Read the gather fit together with model error." loading="lazy"></a>
  </div>
  <figcaption>Shot 0 observations, initial prediction, final prediction and final residual share the same amplitude scale. The horizontal axis is receiver index and the vertical axis is time. Read the gather fit together with model error. <a href="../_static/scalar3d/layered/shot_gather.png" target="_blank" rel="noopener">Open the full-resolution figure ↗</a></figcaption>
</figure>
```

## Numerical checks

These are relative L2 errors recorded in the report. Every listed check is within its preset tolerance. The directional test covers one interior direction, not derivatives of padded edge parameters or arbitrary configurations.

| Check | Spatial order | Record error | Gradient / derivative error |
|---|---:|---:|---:|
| full / boundary | 2 | 0 | 5.534903087e-07 |
| full / boundary | 4 | 0 | 7.268850089e-07 |
| full / boundary | 6 | 0 | 5.628268241e-07 |
| full / boundary | 8 | 0 | 4.354951216e-07 |
| Central finite difference / gradient inner product | 4 | — | 9.924772394e-05 |
| One GPU / four GPUs | 4 | 0 | 4.541373676e-08 |

See {ref}`validation scope <validation-scope>` for definitions and tolerances.

## Downloads and reproduction

Use [Downloads and reproduction](reproduce.md) for the complete report, notebooks, scripts and numerical-result documentation. Each figure opens at its original resolution; the documentation does not redraw or cosmetically modify the inversion result.
