# StarWave original teaching notebooks

Both language editions of the documentation link to the same original Chinese-language notebooks and matching percent-format Python scripts:

1. [Installation and environment smoke test](01_installation_environment_smoke.ipynb) · [Python script](01_installation_environment_smoke.py)
2. [Forward modelling and a velocity gradient](02_forward_and_gradient.ipynb) · [Python script](02_forward_and_gradient.py)
3. [A small full-waveform inversion](03_simple_fwi.ipynb) · [Python script](03_simple_fwi.py)

Each notebook is standalone and generates its analytic model and source locally. Start with 01 in a compatible CUDA environment, then run 02 and 03 independently from a fresh kernel. No external seismic dataset or project-specific helper module is needed. Installation commands appear only as documentation, not executable notebook cells.

## What these downloads establish

These public files are output-free privacy/editorial editions of the original teaching notebooks, not new experiments. Their numerical, plotting and result-export cells are unchanged byte-for-byte. The environment helper only loses two unnecessary development-provenance fields. [source_manifest.json](source_manifest.json) records original and public hashes, all edited cells, unchanged code-cell hashes and configuration hashes. The paired scripts are deterministic notebook exports with identical executable ASTs.

The public edited downloads have not been executed byte-for-byte. [results_summary.json](results_summary.json) contains curated measurements from the returned original notebooks on NVIDIA A30, Python 3.10.18, PyTorch 2.5.1, CUDA 11.8 and StarWave 0.1.0.dev9. The public 2.0.0 wheel was not tested by that run. Raw environment files, runtime logs, executed notebook outputs and private source details are not bundled here.

This is a small same-solver synthetic teaching experiment. Twenty-five optimizer updates were completed, with no convergence certification. Data fit improved much more than velocity-model recovery and the central lens remained substantially unresolved. The optional directional finite-difference check was not run; full adjoint correctness, multi-GPU acceptance and field-data performance are not established. Recorded pressure amplitudes are arbitrary units, not calibrated Pa.

## Validate without running CUDA

From the repository root:

```sh
python tools/check_tutorial_assets.py
```

The checker validates clean outputs and metadata, Python syntax, notebook/script equality, code safety, numerical config/cell hashes, finite reported metrics and required caveats. It does not import StarWave or run a solver. An optional `--originals` argument can compare authorized original files directly; it is not needed for a public checkout.

## 中文说明

中英文文档都下载这三份原始中文教学 notebook；没有另造一套英语数值实验。公开版只做隐私与说明文字清理，数值实验保持原样。记录来自原版在 0.1.0.dev9 环境的实际运行，不表示公开 2.0.0 wheel 已验证，也不表示清理后的文件逐字节重新运行过。请同时阅读来源清单与结果局限。
