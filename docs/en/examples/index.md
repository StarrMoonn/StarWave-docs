# Scalar3D examples

These examples document three actual 3D scalar-acoustic CUDA experiments completed on four NVIDIA A30 GPUs on 2026-10-08. Each includes acquisition geometry, true and starting models, recovered models, convergence curves, intermediate gradients, shot gathers and numerical checks. The displayed sections belong to 3D volumes; they are not 2D propagation experiments.

- [Enclosed anomaly](enclosed.md): the primary functional example, with clear recovery under broad illumination after 100 epochs
- [Surface-only anomaly](surface.md): passing numerical checks but poor recovery inside the anomaly, retained as a limitation
- [Undulating thin layers](layered.md): a lightweight example with a Gaussian-smoothed starting model and about 20% lower RMSE after 100 epochs
- [Downloads and reproduction](reproduce.md): complete report, executed notebooks, scripts and result organization

## Results at a glance

| Case | Grid X × Y × Z | Epochs | Training time | Whole-model RMSE (m/s) | Fixed-model loss decrease |
|---|---|---:|---:|---|---:|
| Enclosed anomaly | 50 × 50 × 30 | 100 | 94.13 s | 39.486 → 4.413 | 99.9791% |
| Surface-only anomaly | 80 × 80 × 50 | 200 | 707.66 s | 25.305 → 23.276 | 96.4050% |
| Undulating thin layers | 50 × 50 × 10 | 100 | 77.10 s | 84.704 → 67.714 | 99.8340% |

A smaller loss establishes a better fit to the chosen observations. In the surface-only case the true centre is 2300 m/s, while the final value is 1948.37 m/s and anomaly-region RMSE remains 207.96 m/s. Its 96.4050% loss reduction must not be presented as complete model recovery.

## Shared experimental setup

The runs used 4 × NVIDIA A30, PyTorch 2.5.1 and CUDA 11.8, with GPU order `[2, 0, 1, 3]`. They used the compiled V11 Scalar3D radius-M source identified by the report, without editing propagation or CUDA code during these experiments. Preparing this documentation does not rerun the GPU experiments or relabel historical source-build tests as a new test of the PyPI wheel.

- Grid spacing 10 m, time step 0.001 s and a complete 25 Hz Ricker wavelet
- FWI: fourth-order spatial differences, PML width 12, buffer 5, `memory='boundary'` and `step_ratio=1`
- Adam learning rate 10.0; one shot per GPU per batch; seeded random shot ordering
- Noiseless synthetic observations and inversion on the same grid with the same operator
- Runtime model order `(X, Y, Z)` and saved-array order `(Z, Y, X)`; transpose explicitly before propagation rather than reshaping
- PML on all six faces; top-plane sensors lie inside the physical model and do not introduce a pressure-release free surface

Anomaly examples start from uniform 2000 m/s with the outer shell of three grid cells of known background fixed. Interior voxels are independently optimized, without centre/shape constraints or gradient/model smoothing. The layered starting model is Gaussian-smoothed truth with the top two depth grid planes (z=0 and z=1) fixed. Truth is used for synthetic observations, evaluation and visualization; its additional use to construct the layered starting model is explicitly disclosed.

See {ref}`Usage <scalar>` for the exact current public parameters, axis order and return contract. The scripts' XYZ/ZYX convention should not be generalized into a requirement that every user's geographical axes have those names.

## Objective, gradients and measurement definitions

The objective is normalized mean-squared error, with a denominator fixed throughout training:

```{math}
J(v) = \frac{\operatorname{mean}\left[(F(v)-d_{\mathrm{obs}})^2\right]}{\operatorname{mean}\left[d_{\mathrm{obs}}^2\right]}.
```

Training-pass means aggregate models updated batch by batch. Every ten epochs, all shots are also evaluated at one fixed model; these metrics are saved and plotted separately. RMSE uses the synthetic truth for evaluation and does not enter the objective. Incomplete reconstructions remain in the results.

Intermediate gradients recompute and sum every shot at a fixed saved model to obtain `dJ/dVp`. Raw gradients and known-region-masked gradients are stored separately, without preconditioning. Recomputed losses agree with the original fixed-model evaluations. Each example includes both gradient views:

1. Divide each checkpoint by its own volume-wide `max|g|` to compare shape; retain the original amplitude in the title
2. Use one shared, unnormalized scale across checkpoints to compare absolute amplitude

Training time includes all-shot fixed-model evaluation and saving every ten epochs. It excludes preparation, numerical checks, observation generation, later gradient recomputation and plotting. Other workloads were present on the server, so these are not dedicated-machine performance benchmarks.

| Case | Peak allocated MiB on GPUs 2 / 0 / 1 / 3 |
|---|---|
| Enclosed | 113.9 / 101.1 / 101.1 / 101.1 |
| Surface-only | 296.5 / 257.3 / 257.3 / 257.3 |
| Thin layers | 67.6 / 61.8 / 61.8 / 61.8 |

These are training-stage PyTorch allocated-memory peaks, excluding CUDA context, other users and the preceding full-memory checks. No unnecessary tensors were added to consume a target memory allowance.

(validation-scope)=
## Validation scope and limitations

All three cases record these checks:

- **full / boundary:** identical acquisition and sampling, separately at spatial orders 2, 4, 6 and 8. Relative L2 tolerances are `1e-6` for records and `1e-3` for model gradients. Measured gradient errors range from about `4.35e-7` to `7.27e-7`; record errors are zero
- **Interior directional derivative:** fourth order, comparing `[J(v+εh)−J(v−εh)]/(2ε)` against `∇J·h`, with `ε=1 m/s` and relative-error tolerance `5e-3`. The three measured errors are `1.474e-5`, `7.055e-5` and `9.925e-5`
- **One GPU / four GPUs:** the same four shots, with relative-error tolerances `1e-6` for records and `1e-4` for gradients. Record errors are zero; gradient errors range from about `4.54e-8` to `1.24e-7`

Propagation uses float32 and relative norms use float64. The finite-difference direction is restricted to the physical-model interior to avoid mixing in the inherited replicate-padding and gradient-cropping convention. It validates the interior discrete derivative for these tested configurations, not derivatives of unknown edge parameters or every direction, model and configuration.

These experiments do not test field data, a free surface, double precision, all CFL choices, arbitrary boundary placement or noise robustness. No independent controlled study separates the effects of shot density, band, starting model and learning rate in the surface-only case. Successful compilation, attractive figures and decreasing loss cannot independently prove gradient correctness; numerical consistency does not guarantee model recoverability.
