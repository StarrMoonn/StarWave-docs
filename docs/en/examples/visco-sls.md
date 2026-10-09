# SLS: Marmousi2 Inversion

This example compares Vp inversion with fixed true Q against joint Vp/Q inversion, using two **100-epoch** CUDA runs saved by the user on 2026-10-09. Both reduce data mismatch and Vp error. Joint Q error reaches its minimum partway through training and then rises, so decreasing training loss and continued Q recovery must be assessed separately.

The runs use the independent SLS source version preceding its V14 integration; they are not relabelled as tests of the 7.0.0 wheel. Documentation preparation reads existing outputs and figures without rerunning a GPU. The original executed notebook is a **5-epoch** test; the 100-epoch runs were executed in the background by the paired script. Their execution records must not be conflated. See the [SLS API](../visco-sls.md) for interface conventions.

(sls-example-setup)=
## Models, acquisition and optimization

| Parameter | Both 100-epoch experiments |
|---|---|
| Model axes / grid | `[z,x]`, 117 × 567, float32 |
| dz / dx | 30 m / 30 m |
| Equation / density | 2D single-SLS; fixed constant density 2000 kg/m³ |
| Vp / Q meaning | Phase velocity and complex-bulk-modulus Q at 6 Hz |
| Ricker / peak | 6 Hz / 0.45 s; forcing in Pa/m² |
| dt / nt | 0.0015 s / 4000, without internal resampling |
| Spatial order / PML / memory | 8 / 10 cells per side / full |
| Fixed max_vel | 6000 m/s, covering unrelaxed speed |
| Shots / receivers per shot | 30 / 567, both at row z=0 |
| Source horizontal indices | 18, 36, …, 540; receivers span all horizontal cells |
| Devices / external batches | Two NVIDIA A30 GPUs; 15 batches, one shot per GPU per batch |
| Initial model | Gaussian smoothing of truth, sigma=8 cells |
| Fixed region | Top 16 water rows and the outermost physical cells |
| Adam / learning rates | betas=(0.5,0.99); Vp=10, joint Q=0.5 |
| Experiment bounds | Vp 800–5000 m/s; joint Q 5–1200 |
| Epochs / updates | 100 epochs per run; 15 updates per epoch, 1500 Adam updates total |

True rock Q is constructed using this experiment's relation `Q=3.516e-6 * Vp[m/s]**2.2`; water is separately fixed at Q=1000. The relation is documented in [Dou and Zhang (2016), §5.3](https://html.rhhz.net/dqwlxb/2016-11-4212.htm), DOI: 10.6038/cjg20161123. Here it only constructs synthetic rock Q; the paper's generalized constant-Q and irregular-grid modeling claims do not transfer to this single-SLS experiment. It is not a density law or an internal propagation constraint. Joint Vp and Q are optimized independently without enforcing that relation during training. Fixed-Q inversion uses true Q throughout and therefore has stronger prior information than the joint run.

Initial Vp is smoothed true Vp; initial joint Q is smoothed true Q, with true water values restored. Fixed regions do not update after initialization, leaving 56,500 of 66,339 cells trainable. Q smoothing includes high-Q water before restoring water truth, so initial rock Q just below the water can exceed the maximum true rock Q. These are synthetic-experiment assumptions, not a claim that such background models are known for field data.

Placing sources and receivers at row z=0 does not introduce a free surface. Source and receiver spacing are 540 m and 30 m; the last sample is at 5.9985 s. Observations use the same SLS equations as inversion, without an added-noise step. This is an idealized, matched-physics synthetic experiment, not a field-Q recovery or model-mismatch robustness test.

(sls-example-models)=
## True, initial and final models


```{raw} html
<figure class="sw-example-figure" id="figure-joint-models-initial">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Horizontally scrollable scientific figure">
    <a href="../_static/sls/figures/joint_models_initial.png" target="_blank" rel="noopener" aria-label="Open full-resolution figure"><img src="../_static/sls/figures/joint_models_initial.png" alt="True and initial models for joint inversion. Initial models are Gaussian-smoothed truth with true values restored in the top 16 water rows, both experimental priors. Initial/epoch-zero panels are duplicates. This original initialization figure and the later final figure use different color limits, so cross-figure comparisons must use numeric scales and model-error metrics." loading="lazy"></a>
  </div>
  <figcaption>True and initial models for joint inversion. Initial models are Gaussian-smoothed truth with true values restored in the top 16 water rows, both experimental priors. Initial/epoch-zero panels are duplicates. This original initialization figure and the later final figure use different color limits, so cross-figure comparisons must use numeric scales and model-error metrics. <a href="../_static/sls/figures/joint_models_initial.png" target="_blank" rel="noopener">Open full-resolution figure ↗</a></figcaption>
</figure>
```


```{raw} html
<figure class="sw-example-figure" id="figure-marmousi100-models-comparison">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Horizontally scrollable scientific figure">
    <a href="../_static/sls/figures/marmousi100_models_comparison.png" target="_blank" rel="noopener" aria-label="Open full-resolution figure"><img src="../_static/sls/figures/marmousi100_models_comparison.png" alt="Both modes at epoch 100 with shared Vp scales in m/s and shared dimensionless rock-Q scales. Fixed water Q=1000 exceeds the displayed rock-Q range. Q in the fixed-Q mode equals truth by construction; it is an input, not an inversion result." loading="lazy"></a>
  </div>
  <figcaption>Both modes at epoch 100 with shared Vp scales in m/s and shared dimensionless rock-Q scales. Fixed water Q=1000 exceeds the displayed rock-Q range. Q in the fixed-Q mode equals truth by construction; it is an input, not an inversion result. <a href="../_static/sls/figures/marmousi100_models_comparison.png" target="_blank" rel="noopener">Open full-resolution figure ↗</a></figcaption>
</figure>
```

(sls-example-metrics)=
## Full evaluation and training curves

The table evaluates normalized MSE over all 30 shots with the initial or final model held fixed. The denominator is the fixed mean-square value of all observations. The optimizer minimizes this normalized objective multiplied by `1e6`. RMSE below covers the entire model relative to synthetic truth, including fixed regions, rather than only trainable cells.

| Metric | Fixed true Q, invert Vp | Joint Vp/Q |
|---|---:|---:|
| Initial full-evaluation normalized MSE | 0.051367496 | 0.071956527 |
| Final full-evaluation normalized MSE | 0.000436810 | 0.003924164 |
| Initial Vp RMSE (m/s) | 365.097 | 365.097 |
| Final Vp RMSE (m/s) | 191.772 | 206.753 |
| Initial / final Q RMSE | Fixed truth, both 0 | 99.737 → 82.820 |

Trainable-region Vp RMSE is 393.076→202.933 m/s for fixed Q and 393.076→219.526 m/s for joint inversion; trainable-region joint-Q RMSE is 107.707→89.300. These differ from the whole-model metrics in the table.

Each training-curve point averages losses computed before successive batch updates within that epoch, while the model changes. It is not a full-shot reevaluation with one fixed model. In particular, the joint epoch-100 training average is about 0.000950100, whereas the final full evaluation is 0.003924164. They must not be combined into one supposedly identical metric.


```{raw} html
<figure class="sw-example-figure" id="figure-marmousi100-loss-comparison">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Horizontally scrollable scientific figure">
    <a href="../_static/sls/figures/marmousi100_loss_comparison.png" target="_blank" rel="noopener" aria-label="Open full-resolution figure"><img src="../_static/sls/figures/marmousi100_loss_comparison.png" alt="Batch-averaged normalized MSE over 100 training epochs on a logarithmic vertical axis. Each batch is followed by an update. Both reduce the training objective, but these curves are distinct from full evaluations at fixed initial/final models." loading="lazy"></a>
  </div>
  <figcaption>Batch-averaged normalized MSE over 100 training epochs on a logarithmic vertical axis. Each batch is followed by an update. Both reduce the training objective, but these curves are distinct from full evaluations at fixed initial/final models. <a href="../_static/sls/figures/marmousi100_loss_comparison.png" target="_blank" rel="noopener">Open full-resolution figure ↗</a></figcaption>
</figure>
```


```{raw} html
<figure class="sw-example-figure" id="figure-marmousi100-rmse-comparison">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Horizontally scrollable scientific figure">
    <a href="../_static/sls/figures/marmousi100_rmse_comparison.png" target="_blank" rel="noopener" aria-label="Open full-resolution figure"><img src="../_static/sls/figures/marmousi100_rmse_comparison.png" alt="Per-epoch whole-model RMSE relative to synthetic truth. Vp errors improve in both runs; joint Q reaches its minimum near epoch 60 and then rises. Truth-based errors are evaluation metrics, not inputs to the optimized loss." loading="lazy"></a>
  </div>
  <figcaption>Per-epoch whole-model RMSE relative to synthetic truth. Vp errors improve in both runs; joint Q reaches its minimum near epoch 60 and then rises. Truth-based errors are evaluation metrics, not inputs to the optimized loss. <a href="../_static/sls/figures/marmousi100_rmse_comparison.png" target="_blank" rel="noopener">Open full-resolution figure ↗</a></figcaption>
</figure>
```

(sls-example-q)=
## Interpreting late-stage Q degradation

The complete per-epoch history gives a minimum joint-Q RMSE of **70.131958 at epoch 59**, compared with **82.819542 at epoch 100**. Epoch 60 is the nearby saved-model checkpoint used for the diagnostic plot's dashed line. Training loss continues to decrease overall in the later stage; it does not establish the final Q model as the best one.

Truth-based RMSE is available only in this synthetic experiment; an unknown Q model cannot supply an RMSE stopping criterion for field data.

This behavior is consistent with Vp/Q trade-offs and identifiability limits arising from acquisition and frequency content. These two trajectories alone cannot identify a unique cause. The fixed-Q run has the true-Q prior, so the difference cannot be assigned directly to one parameter, learning rate or propagation-code defect. Alternative initial models, bands, regularization or optimization strategies require new controlled experiments; unrun alternatives are not presented as improved results.


```{raw} html
<figure class="sw-example-figure" id="figure-joint-loss-and-model-errors">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Horizontally scrollable scientific figure">
    <a href="../_static/sls/figures/joint_loss_and_model_errors.png" target="_blank" rel="noopener" aria-label="Open full-resolution figure"><img src="../_static/sls/figures/joint_loss_and_model_errors.png" alt="Joint training loss, Vp RMSE and Q RMSE shown together. The dashed line is epoch 60; the exact Q optimum in the per-epoch history is 59. Later data fitting improves while Q model error grows." loading="lazy"></a>
  </div>
  <figcaption>Joint training loss, Vp RMSE and Q RMSE shown together. The dashed line is epoch 60; the exact Q optimum in the per-epoch history is 59. Later data fitting improves while Q model error grows. <a href="../_static/sls/figures/joint_loss_and_model_errors.png" target="_blank" rel="noopener">Open full-resolution figure ↗</a></figcaption>
</figure>
```


```{raw} html
<figure class="sw-example-figure" id="figure-joint-Q-all-epochs">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Horizontally scrollable scientific figure">
    <a href="../_static/sls/figures/joint_Q_all_epochs.png" target="_blank" rel="noopener" aria-label="Open full-resolution figure"><img src="../_static/sls/figures/joint_Q_all_epochs.png" alt="Q evolution under a shared scale: truth, initialization and saved models every 10 epochs, not all 100 model snapshots. Water Q=1000 exceeds the rock-Q scale. Shared limits support comparison of recovery patterns and later changes." loading="lazy"></a>
  </div>
  <figcaption>Q evolution under a shared scale: truth, initialization and saved models every 10 epochs, not all 100 model snapshots. Water Q=1000 exceeds the rock-Q scale. Shared limits support comparison of recovery patterns and later changes. <a href="../_static/sls/figures/joint_Q_all_epochs.png" target="_blank" rel="noopener">Open full-resolution figure ↗</a></figcaption>
</figure>
```


```{raw} html
<figure class="sw-example-figure" id="figure-joint-models-epoch-100">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Horizontally scrollable scientific figure">
    <a href="../_static/sls/figures/joint_models_epoch_100.png" target="_blank" rel="noopener" aria-label="Open full-resolution figure"><img src="../_static/sls/figures/joint_models_epoch_100.png" alt="True, initial and final joint Vp/Q models at epoch 100. The final inversion texture is retained without smoothing or independently rescaling images to conceal Q reconstruction error." loading="lazy"></a>
  </div>
  <figcaption>True, initial and final joint Vp/Q models at epoch 100. The final inversion texture is retained without smoothing or independently rescaling images to conceal Q reconstruction error. <a href="../_static/sls/figures/joint_models_epoch_100.png" target="_blank" rel="noopener">Open full-resolution figure ↗</a></figcaption>
</figure>
```

(sls-example-ring)=
## Ring-acquisition small-model comparison

The supplied outputs also contain a separate 41×41, 25 Hz RTX 4060 ring-acquisition experiment. With Vp fixed and only Q inverted, Q RMSE falls from 3.099631 to 0.114001. Joint inversion reduces Vp RMSE from 41.328411 to 0.911016 m/s and Q RMSE from 3.099631 to 0.993559. Q can improve under this small model's acquisition conditions, while jointly estimating parameters remains harder than holding the others fixed. This does not replace Marmousi2 identifiability analysis.


```{raw} html
<figure class="sw-example-figure" id="figure-ring-Q-only-models">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Horizontally scrollable scientific figure">
    <a href="../_static/sls/figures/ring_Q_only_models.png" target="_blank" rel="noopener" aria-label="Open full-resolution figure"><img src="../_static/sls/figures/ring_Q_only_models.png" alt="Fixed-Vp, Q-only inversion in the ring-acquisition model, comparing true, initial and final Q. Acquisition, scale and background differ from the surface-acquisition Marmousi2 experiment." loading="lazy"></a>
  </div>
  <figcaption>Fixed-Vp, Q-only inversion in the ring-acquisition model, comparing true, initial and final Q. Acquisition, scale and background differ from the surface-acquisition Marmousi2 experiment. <a href="../_static/sls/figures/ring_Q_only_models.png" target="_blank" rel="noopener">Open full-resolution figure ↗</a></figcaption>
</figure>
```


```{raw} html
<figure class="sw-example-figure" id="figure-ring-joint-models">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Horizontally scrollable scientific figure">
    <a href="../_static/sls/figures/ring_joint_models.png" target="_blank" rel="noopener" aria-label="Open full-resolution figure"><img src="../_static/sls/figures/ring_joint_models.png" alt="Joint Vp/Q ring-acquisition results. Both central anomalies improve, but local Q errors remain; data fitting alone does not establish a unique recovery." loading="lazy"></a>
  </div>
  <figcaption>Joint Vp/Q ring-acquisition results. Both central anomalies improve, but local Q errors remain; data fitting alone does not establish a unique recovery. <a href="../_static/sls/figures/ring_joint_models.png" target="_blank" rel="noopener">Open full-resolution figure ↗</a></figcaption>
</figure>
```


```{raw} html
<figure class="sw-example-figure" id="figure-ring-loss">
  <div class="sw-example-viewport" role="region" tabindex="0" aria-label="Horizontally scrollable scientific figure">
    <a href="../_static/sls/figures/ring_loss.png" target="_blank" rel="noopener" aria-label="Open full-resolution figure"><img src="../_static/sls/figures/ring_loss.png" alt="Ring-model losses divided by each mode&#x27;s own initial loss, showing relative reduction. This is not Marmousi2 observation-energy-normalized MSE, and the numeric values cannot be directly compared." loading="lazy"></a>
  </div>
  <figcaption>Ring-model losses divided by each mode&#x27;s own initial loss, showing relative reduction. This is not Marmousi2 observation-energy-normalized MSE, and the numeric values cannot be directly compared. <a href="../_static/sls/figures/ring_loss.png" target="_blank" rel="noopener">Open full-resolution figure ↗</a></figcaption>
</figure>
```

(sls-example-download)=
## Download and reproduce

The complete public package is about 23.1 MiB. It contains both 100-epoch histories, every-10-epoch models, representative gradient snapshots, original model data, ring controls, ten figures, the experiment script, a saved-results notebook and a four-page bilingual PDF. Model/results arrays and the original experiment workflow preserve their values and logic; machine paths and process details are removed from summaries.

```{raw} html
<nav class="sw-example-downloads" aria-label="Complete SLS example download">
<button type="button" class="sw-button sw-button-primary" data-sw-package-download data-manifest="../_static/sls/downloads/package/package.json" data-status="sls-package-status" data-cancel="sls-package-cancel">Download complete SLS ZIP (about 23.1 MiB)</button>
<button type="button" class="sw-button" id="sls-package-cancel" hidden>Pause</button>
</nav>
<p id="sls-package-status" class="sw-example-package-status" role="status" aria-live="polite">The current browser verifies each part and the complete SHA-256 before saving one standard ZIP. Downloads can be paused and resumed.</p>
<noscript><p>JavaScript is disabled. Use the Python downloader below to retrieve and verify the same complete ZIP.</p></noscript>
<script src="../_static/sls/package-download.js" defer></script>
```

- [Python downloader and verifier](../_static/sls/downloads/package/join_package.py), using only Python's standard library
- [Package size, parts and SHA-256](../_static/sls/downloads/package/package.json)
- [SLS_Saved_Results.ipynb](../_static/sls/downloads/SLS_Saved_Results.ipynb): read saved 100-epoch results, recompute RMSE and plot, without propagation or training
- [CPU results-review script](../_static/sls/downloads/review_saved_results.py), requiring the complete package directory
- [Original experiment workflow](../_static/sls/downloads/marmousi2_sls_fwi.py), for explicitly rerunning the experiment; its default is five epochs
- [Full notes](../_static/sls/downloads/README.txt), [provenance and publication scope](../_static/sls/downloads/provenance.json), [per-file hashes](../_static/sls/downloads/PACKAGE_MANIFEST.json)

```{raw} html
<nav class="sw-example-downloads" aria-label="SLS report download">
<a class="sw-button sw-button-primary" href="../_static/sls/downloads/SLS_Example_Report.pdf" download="SLS_Example_Report.pdf">Download bilingual PDF report</a>
<a class="sw-button" href="../_static/sls/downloads/SLS_Example_Report.pdf" target="_blank" rel="noopener">Open PDF in a new window ↗</a>
</nav>
```

Downloading and reading saved results require neither StarWave nor a GPU. The notebook alone does not include all data; preserve relative directories after extraction. The four-page PDF is a newly organized bilingual report based on saved outputs, and the notebook is a new read-only results viewer. Neither is a newly executed 100-epoch GPU notebook.

```bash
python join_package.py
# Extract StarWave-SLS-Marmousi2-Example.zip and enter its directory:
sha256sum -c MANIFEST.sha256
python review_saved_results.py
```

CPU review requires NumPy and Matplotlib; the notebook also requires Jupyter. Complete observed shot gathers are not included, so full-objective values come from the original run summaries. Saved-model RMSE can be independently recomputed, but these model arrays alone cannot verify the all-shot MSE.

Rerunning training requires an installed StarWave SLS package compatible with this API, PyTorch, NumPy, SciPy, Matplotlib, CUDA devices and sufficient full-history memory. The script also uses the matching version's SLS coefficient helper for time-step diagnostics; this is not a new public API, and older wheels need not provide it. To preserve the original workflow, its explicit `--source-root` check is retained. For an installed wheel, set it to the parent of the installed `starwave` directory:

```bash
SOURCE_ROOT="$(python -c 'import pathlib, starwave; print(pathlib.Path(starwave.__file__).resolve().parent.parent)')"
python marmousi2_sls_fwi.py --source-root "$SOURCE_ROOT" \
  --data data/mar_big_117_567.bin --out new_fixed_Q \
  --gpu-ids 0 1 --num-batches 15 --epochs 100 --mode vp
python marmousi2_sls_fwi.py --source-root "$SOURCE_ROOT" \
  --data data/mar_big_117_567.bin --out new_joint \
  --gpu-ids 0 1 --num-batches 15 --epochs 100 --mode vpq
```

Choose actual visible GPU indices and fresh output directories; existing experiments are not automatically overwritten. These commands are optional reproduction instructions, not new training performed for this documentation. Read the model binary as float32 with `reshape(567,117).T`; it was reconstructed from saved Vp truth and matches the original recorded SHA-256. Different GPU counts, environments or floating-point ordering can change optimization trajectories; original wall times and bitwise-identical results are not promised.
