# Downloads and reproduction

The download contains training scripts, three executed result notebooks, all model and observation arrays, models saved every ten epochs, raw and masked gradients, acquisition geometry, metric JSON, execution logs and figures for all three Scalar3D cases. These are the completed CUDA experiments of 2026-10-08. No GPU training or numerical validation was rerun for this publication.

## Complete package and individual files

```{raw} html
<nav class="sw-example-downloads" aria-label="Complete Scalar3D download">
<button type="button" class="sw-button sw-button-primary" data-sw-package-download data-manifest="../_static/scalar3d/downloads/package/package.json" data-status="package-status" data-cancel="package-cancel">Download complete ZIP (about 71 MiB)</button>
<button type="button" class="sw-button" id="package-cancel" hidden>Pause</button>
</nav>
<p id="package-status" class="sw-example-package-status" role="status" aria-live="polite">Downloads and SHA-256 verification run in your browser. The result is one standard ZIP file.</p>
<noscript><p>JavaScript is disabled. Download the Python script below to retrieve and verify the complete ZIP from this site.</p></noscript>
<script src="../_static/scalar3d/package-download.js" defer></script>
```

Pause at any time or click the same button to resume after a network error. Each request reads at most 4 MiB and assembly stays in your browser; no data is sent to another service. If a phone's memory or download restrictions prevent completion, use the script on a computer.

- [Python download and verification script](../_static/scalar3d/downloads/package/join_package.py): no JavaScript, Python standard library only
- [Package manifest and complete SHA-256](../_static/scalar3d/downloads/package/package.json)
- [Enclosed anomaly notebook](../_static/scalar3d/downloads/Scalar3D_enclosed_Results.ipynb)
- [Surface-only anomaly notebook](../_static/scalar3d/downloads/Scalar3D_surface_Results.ipynb)
- [Undulating layers notebook](../_static/scalar3d/downloads/Scalar3D_layered_Results.ipynb)
- [Unified entry-point script](../_static/scalar3d/downloads/run_scalar3d_example.py), for use with the complete package
- [Publication notes](../_static/scalar3d/downloads/PUBLICATION_NOTES.json)

```bash
python join_package.py
```

The public package replaces private server paths, adds explanatory notes and regenerates `PACKAGE_MANIFEST.json`. Numerical arrays, figures, metrics and training loops remain unchanged. An unrelated private source-history inventory is omitted, while relevant source and native-library fingerprints are retained. The older 100-epoch surface script, server authoring script and duplicate previews present only in the second input archive do not replace the final 200-epoch result. This publication grants no new code or data license.

## Complete report

```{raw} html
<nav class="sw-example-downloads" aria-label="Scalar3D report downloads">
<a class="sw-button sw-button-primary" href="../_static/scalar3d/downloads/Scalar3D-CUDA-Report.pdf" download="Scalar3D-CUDA-Report.pdf">Download complete PDF report</a>
<a class="sw-button" href="../_static/scalar3d/downloads/REPORT.html" download="Scalar3D-CUDA-Report.html">Download original HTML report</a>
<a class="sw-button" href="../_static/scalar3d/downloads/Scalar3D-CUDA-Report.pdf" target="_blank" rel="noopener">Open PDF in a new window ↗</a>
</nav>
```

Neither input ZIP contains a PDF. The 25-page PDF repaginates the original HTML text, tables and 21 unchanged figures and adds a publication note. The inversion plots are not redrawn or cosmetically altered. The original Chinese HTML is preserved in full, including its historical “not yet published” statement.

The original closing summary of truth usage needs a qualification: the layered starting model also uses Gaussian-smoothed truth. Its fixed top two layers mean grid planes z=0 and z=1. The current bilingual text and PDF publication note make both points explicit. The case pages fully explain the experiments in English; the historical report itself remains Chinese.

## Read and replot saved results

Extract the complete ZIP and enter `Scalar3D_Examples_20261008`. Verify integrity, then read or replot the saved results on CPU. Hash checks establish file integrity, not fresh numerical validation of the propagator.

```bash
python verify_package.py
python render_scalar3d.py results/enclosed
python render_scalar3d.py results/surface
python render_scalar3d.py results/layered
```

CPU plotting requires NumPy and Matplotlib; notebooks additionally use IPython/Jupyter. It needs neither a GPU nor StarWave nor the training helper. Executed notebook cells load actual saved results and replot them; paired Python scripts performed the CUDA training on A30 GPUs. A notebook or entry point alone is not a complete run package: preserve the relative directory layout.

Optional notebook training defaults to `RUN_FWI = False`. Opening the report does not launch training. Configure a source build, GPUs and a fresh output directory before explicitly enabling it.

## Rerun a CUDA experiment

Training from scratch requires NumPy, SciPy, Matplotlib, PyTorch and a compiled Scalar3D source build matching the recorded implementation. The scripts also depend on `example_support/scalar_examples.py` from that source checkout. This helper is absent from both original result ZIPs and is not distributed in the public result package; use a matching source checkout you are authorized to access. `source_identity.json` retains the expected helper, key source-file and original native-library SHA-256 values. CPU reading of saved results is unaffected.

```bash
python run_scalar3d_example.py --case enclosed \
    --source-root /path/to/compiled/StarWave \
    --gpu-ids 2,0,1,3 \
    --output-dir /path/to/new_results
```

`enclosed` and `layered` default to 100 epochs; `surface` defaults to 200. Use a new output directory; scripts refuse to overwrite an existing `history.json`. Changing GPU count changes Adam updates per epoch, so complete optimization trajectories need not match across GPU counts. The package does not install or compile software or include GPU binaries, and this publication does not validate a new compiled environment.

The original unified runner chooses quartile indices among available checkpoints: `0,20,50,80,100` at 100 epochs and `0,50,100,150,200` at 200. These differ from the archival figures. To match the report's gradient checkpoints, explicitly run the following after training:

```bash
# enclosed / layered
python record_checkpoint_gradients.py /path/to/new_results \
    --source-root /path/to/compiled/StarWave --gpu-ids 2,0,1,3 \
    --epochs 0,10,30,60,100

# surface: use the corresponding surface output directory
python record_checkpoint_gradients.py /path/to/new_surface_results \
    --source-root /path/to/compiled/StarWave --gpu-ids 2,0,1,3 \
    --epochs 0,20,60,100,200

python render_scalar3d.py /path/to/new_results
```

These are optional reproduction commands for the reader, not experiments executed again while preparing the manual. Training loops and the original entry-point logic are preserved.

## Data, metrics and what can be checked

- `case_manifest.json`: case-to-training-script mapping
- `results/<case>/summary.json`: configuration, timing, memory and initial/final model metrics
- `results/<case>/numerical_checks.json`: recorded full/boundary, interior directional-derivative and one/four-GPU checks
- `results/<case>/gradients/gradient_manifest.json`: gradient definitions, checkpoint model hashes and evaluation losses
- `validation_summary.json`: machine-readable overall summary
- `source_identity.json`: source and native-library fingerprints from the actual runs
- `PACKAGE_MANIFEST.json`: per-file integrity of this public package

This documentation pass independently checked types and finiteness of all 97 NPY arrays, every saved model's RMSE, norms and masks of 15 gradient sets, notebook outputs and original pixels of all 21 report figures. Only shot-0 predictions are saved, allowing that shot's residual metrics to be checked. The saved arrays alone cannot recompute the full-survey objective or independently repeat the directional-derivative test; those remain evidence recorded during the original run.

Saved model order is ZYX while the scripts propagate XYZ models, requiring an explicit transpose. Independently normalized gradient plots compare shape only; a shared raw scale is needed to compare amplitude. See the [overview](index.md) for complete numerical-check definitions and performance measurement scope.
