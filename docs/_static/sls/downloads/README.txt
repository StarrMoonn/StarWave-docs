# Marmousi2 SLS 100 epoch example

This example preserves the completed 2026-10-09 single-SLS synthetic inversions: fixed true Q with Vp inversion, and joint Vp/Q inversion, 100 epochs each. The saved GPU results were reviewed; publication preparation did not rerun the GPU inversions. Both experiments use fixed density 2000 kg/m³. Vp is phase velocity at 6 Hz; Q is the complex-modulus quality factor at 6 Hz. A single SLS is not broadband constant Q.

本例保存 2026-10-09 已完成的两组单 SLS 合成反演：固定真实 Q 只反演 Vp、联合反演 Vp/Q，各 100 轮。此次整理核验已有 GPU 结果，没有重新运行 GPU 反演。密度固定为 2000 kg/m³；Vp 是 6 Hz 相速度，Q 是同频复模量品质因子，单 SLS 不代表全频恒 Q。

## Results and interpretation

| Experiment | Initial full-survey normalized MSE | Final full-survey normalized MSE | Vp RMSE whole grid (m/s) | Q RMSE whole grid |
|---|---:|---:|---:|---:|
| Fixed true Q, Vp only | 0.0513674957 | 0.000436809653 | 365.096741 → 191.771973 | 0 → 0, fixed truth |
| Joint Vp/Q | 0.0719565267 | 0.003924163974 | 365.096741 → 206.753082 | 99.737328 → 82.819542 |

The trainable-region RMSE values are: Vp 393.076202 → 202.932907 m/s for fixed-Q; Vp 393.076202 → 219.526291 m/s and Q 107.706993 → 89.300110 for joint inversion. The mask excludes the first 16 water rows, both side columns, and the bottom row, leaving 56,500 of 66,339 cells trainable. Saved arrays confirm zero updates outside this mask. Whole-grid and trainable-region metrics must not be interchanged.

Joint Q RMSE reaches its best recorded value of 70.131958 at epoch 59, then rises to 82.819542 at epoch 100. Epoch 60 (Q RMSE 70.133904) is the nearest saved model; arrays are saved every 10 epochs. The training-batch loss continues to decrease over much of this later interval. The final Q image remains spatially distorted, with overestimates/underestimates and boundary-adjacent artifacts. A lower data residual does not establish faithful Q recovery.

联合 Q 的全网格 RMSE 在第 59 轮最低，为 70.131958；最终回升至 82.819542。模型每 10 轮保存一次，图中的第 60 轮只是最接近该最优轮次的已保存模型。后期 batch loss 下降，并不意味着 Q 模型误差也下降。合成实验才有真实 Q 可用于这一诊断；实际数据不能用未知真值 RMSE 选择停止轮次。

Vp-Q parameter trade-off is a plausible explanation, not a demonstrated unique cause. These runs do not include a Marmousi fixed-true-Vp Q-only inversion, a sensitivity/Hessian cross-term measurement, a learning-rate ablation, or a regularization ablation. Finite acquisition coverage, parameter scaling, initial-model smoothing, the absence of explicit model regularization, and constant learning rates are also possible contributors. This result alone neither proves a solver bug nor establishes its absence.

参数耦合是合理解释，但本组结果没有单独证明因果。Marmousi 缺少固定真实 Vp 的 Q-only 对照、灵敏度或 Hessian 交叉项分析，以及学习率/正则化消融。有限采集约束、参数尺度、初始模型平滑、未加入显式模型正则及恒定学习率也可能参与。可在后续分别控制变量检验，当前不把这些未经执行的建议当作实验结果。

A separate 41×41 surrounding-acquisition SLS test supports the difficulty of joint estimation within that small test: Q-only with true Vp gives Q RMSE 3.099631 → 0.114001, while joint Vp/Q gives 3.099631 → 0.993559. Its different geometry, 25 Hz wavelet and smaller model prevent directly transferring this contrast to Marmousi.

## Sampling acquisition and optimization

- Grid 117×567; spacing 30 m; float32; 8th-order spatial accuracy; PML width 10; full wavefield-history memory; fixed max_vel 6000 m/s.
- Source and reference frequency 6 Hz; Ricker peak at 0.45 s; unit source forcing in Pa/m²; dt=0.0015 s; nt=4000. Sample times run from 0 to 5.9985 s; nominal record duration nt×dt=6 s. Internal time sampling equals the requested sampling.
- 30 surface shots, index coordinates [z,x]=[0,18+18i] for i=0…29; 567 surface receivers per shot at x=0…566. Source spacing 540 m; receiver spacing 30 m. Fifteen interleaved batches per epoch, two shots each; two NVIDIA A30 GPUs, one shot per GPU per batch. No claimed free surface is introduced by placing sources/receivers at z=0.
- Vp and joint Q initial models are Gaussian smoothed with sigma=8 cells (240 m), then the first 16 water rows are restored. Fixed-Q uses the exact Q truth throughout. Q smoothing includes the high-Q water before restoring that water, so the immediately underlying initial rock Q can exceed the true rock maximum.
- Synthetic rock Q=3.516e-6×Vp[m/s]^2.2; water Q=1000. The relation generates the truth only; the optimizer treats Vp and Q as independent parameters.
- Adam betas=(0.5,0.99), eps=1e-8, Vp learning rate 10, Q learning rate 0.5; constant rates, no scheduler, no explicit model regularization. Each run has 1500 optimizer updates (100 epochs×15 batches).
- Experiment bounds: Vp∈[800,5000] m/s and joint Q∈[5,1200]. These are experiment choices, not general API limits. At epoch 100, joint Q has 29 cells at its lower bound (0.05133% of trainable cells), none at the upper bound; joint Vp has 15 upper-bound cells. Fixed-Q Vp has 12 upper-bound cells.
- Optimized objective=1e6×MSE/observed global mean-square. The fixed normalization denominator is 10.762818941596795. Reported normalized MSE omits the factor 1e6. Initial and final full-survey evaluations do not update the model. Training curves average batches evaluated before different parameter updates and are not the final-model full-survey objective.
- Different Q initial models produce different initial losses; this is a conditional fixed-Q reference versus a harder joint inverse problem, not an identical-start optimizer benchmark.

## Gradients and actual optimizer state

Gradient snapshots are the first batch of epochs 1, 51, and 100 after mask application. They are gradients of the 1e6-scaled objective. At joint epoch 1, whole-grid gradient RMS is 0.0603122 for Vp and 0.0212772 for Q; at epoch 100 these are 0.0140220 and 0.00769787. The parameters have different units and Adam rescales them, so the raw magnitudes or the 20:1 learning-rate ratio cannot by themselves diagnose identifiability.

The final saved optimizer states confirm step=1500 for both parameters. Bias-corrected Adam reconstructs the last unconstrained update proposal RMS as 6.40236 m/s for Vp and 0.285504 for Q (max absolute proposals 29.8431 m/s and 2.28983). These are pre-clamp proposals, not measured post-clamp increments; the immediately preceding per-batch models were not retained. Model changes over saved 10-epoch intervals can be computed directly from the provided arrays.

## Files and use

- SLS_Saved_Results.ipynb: inspect the saved 100-epoch results and plot/recompute RMSE; no simulation, build, installation or network requests.
- review_saved_results.py: command-line numerical review; requires NumPy and Matplotlib.
- marmousi2_sls_fwi.py: the original standalone experiment workflow, byte-identical to the validated script; operator implementations are not included. It requires an installed or separately supplied compatible StarWave SLS package with CUDA support, PyTorch, NumPy, SciPy and Matplotlib. The default is 5 epochs; pass --epochs 100 explicitly for these experiments.
- data/mar_big_117_567.bin: Vp truth reconstructed bit-for-bit from saved model arrays; SHA256 e0260ab520e364ce9e5485a882a43168cfde22a107c2297f34d527982dc09cd3. The binary is float32, read as reshape(567,117).T.
- data/fixed_Q_Vp and data/joint_Vp_Q: initial model, every-10-epoch model arrays, gradient snapshots, sanitized summaries and CSV histories.
- data/ring: separate small-model SLS controls. Do not mix their settings or numbers with the Marmousi experiment.
- figures and SLS_Example_Report.pdf: portable views of the same saved evidence.
- provenance.json and MANIFEST.sha256: source identities and file hashes.

The original executed notebook contains the five-epoch validation, not the later 100-epoch runs. The supplied viewer is new and is deliberately named as a saved-results notebook. The recorded observations were generated with the same SLS solver and model physics used for inversion, without an added-noise step. This is an idealized synthetic verification; it is not evidence of field-data Q recovery or a model-mismatch robustness test. Observed shot gathers are not included; the workflow regenerates them.

To rerun with an installed StarWave package, set SOURCE_ROOT to the parent of the installed starwave package directory, choose your own GPU indices, and use fresh output directories. Example shell command (replace SOURCE_ROOT and GPU IDs appropriately):

    python marmousi2_sls_fwi.py --source-root SOURCE_ROOT --data data/mar_big_117_567.bin --out new_fixed_Q --gpu-ids 0 1 --epochs 100 --num-batches 15 --mode vp
    python marmousi2_sls_fwi.py --source-root SOURCE_ROOT --data data/mar_big_117_567.bin --out new_joint --gpu-ids 0 1 --epochs 100 --num-batches 15 --mode vpq

The script saves rather than overwrites existing experiments and explicitly checks the imported package location. It calls the release's SLS coefficient helper for a time-plan diagnostic; this helper is not a new public API. Do not assume an arbitrary older StarWave wheel supplies it. The original runtime included two A30 GPUs; no equivalent wall time is promised on other hardware or shared systems.

## Figure captions

1. marmousi100_models_comparison.png: true models, fixed-true-Q Vp result, joint Vp/Q result at epoch 100; shared scales across each parameter row. Fixed water Q=1000 exceeds the rock-Q display scale. The middle Q panel is fixed truth, not recovered Q.
2. joint_models_initial.png: true, initial and epoch-zero models for joint inversion. The duplicated initial/epoch-zero panels are intentional and show the starting point.
3. marmousi100_loss_comparison.png: mean normalized pre-update batch losses over 100 epochs. These curves are not the separate full-survey final-model evaluations in the results table.
4. marmousi100_rmse_comparison.png: whole-grid model RMSE traces; Q is unchanged at zero in the fixed-truth-Q run.
5. joint_loss_and_model_errors.png: decreasing batch residuals with later Q-error increase. The vertical line is epoch 60, the closest saved-model checkpoint to the best recorded Q RMSE at epoch 59. A true-model metric is available only in this synthetic test.
6. joint_Q_all_epochs.png: true/initial Q plus checkpoints at epochs 10,20,…,100 on a common rock-Q scale; the title does not imply every single epoch was saved.
7. ring_Q_only_models.png and ring_joint_models.png: separate 41×41 surrounding-acquisition controls. They illustrate easier fixed-true-Vp Q estimation in that setting, not a matched Marmousi ablation.

## Reference for synthetic Q construction

Dou H, Zhang J F. 2016. An irregular grid method for acoustic modeling in inhomogeneous viscoelastic medium. Chinese Journal of Geophysics, 59(11), 4212–4222. Section 5.3 gives the empirical Q-Vp relation, citing Li (1993). https://doi.org/10.6038/cjg20161123 ; journal page https://html.rhhz.net/dqwlxb/2016-11-4212.htm . This reference supports the synthetic Q construction, not the claim that the present single-SLS implementation has that article's broadband constant-Q behavior or unstructured-grid discretization.
