# 下载与复现

下载包包含三组 Scalar3D 实测算例的训练脚本、三个带输出 Notebook、全部模型和观测数组、每十轮模型、原始及 mask 后梯度、采集系统、指标 JSON、运行日志与结果图。这里整理的是 2026-10-08 已完成的 CUDA 实验，没有重新进行 GPU 训练或数值验证。

## 完整算例包与独立文件

```{raw} html
<nav class="sw-example-downloads" aria-label="完整 Scalar3D 下载">
<button type="button" class="sw-button sw-button-primary" data-sw-package-download data-manifest="../_static/scalar3d/downloads/package/package.json" data-status="package-status" data-cancel="package-cancel">下载完整算例 ZIP（约 71 MiB）</button>
<button type="button" class="sw-button" id="package-cancel" hidden>暂停</button>
</nav>
<p id="package-status" class="sw-example-package-status" role="status" aria-live="polite">下载在当前浏览器内完成，逐块及整包校验 SHA-256；完成后保存为一个标准 ZIP。</p>
<noscript><p>JavaScript 已关闭。请下载下面的 Python 合并脚本；运行后会从本站下载并校验完整 ZIP。</p></noscript>
<script src="../_static/scalar3d/package-download.js" defer></script>
```

下载过程中可以暂停，或在网络错误后点击同一按钮继续。网页每次读取不超过 4 MiB，只在当前浏览器内合并文件，不向其他站点传输数据。若手机内存或下载限制导致失败，可在电脑使用下方脚本。

- [Python 下载与校验脚本](../_static/scalar3d/downloads/package/join_package.py)：无需 JavaScript，只用 Python 标准库
- [包清单与整包 SHA-256](../_static/scalar3d/downloads/package/package.json)
- [六面包围 Notebook](../_static/scalar3d/downloads/Scalar3D_enclosed_Results.ipynb)
- [地表采集 Notebook](../_static/scalar3d/downloads/Scalar3D_surface_Results.ipynb)
- [起伏薄层 Notebook](../_static/scalar3d/downloads/Scalar3D_layered_Results.ipynb)
- [统一入口脚本](../_static/scalar3d/downloads/run_scalar3d_example.py)，需配合完整包使用
- [公开整理说明](../_static/scalar3d/downloads/PUBLICATION_NOTES.json)

```bash
python join_package.py
```

公开包仅替换私人服务器绝对路径，补充说明并重新生成 `PACKAGE_MANIFEST.json`。原始数值数组、图像、指标和训练循环保持不变；无关的私有源码历史文件清单被排除，相关源码与编译库指纹保留。第二份输入包多出的旧 100 轮地表脚本、服务器生成脚本与重复预览图不替代最终 200 轮结果。没有为代码或数据声明新的许可证。

## 完整报告

```{raw} html
<nav class="sw-example-downloads" aria-label="Scalar3D 报告下载">
<a class="sw-button sw-button-primary" href="../_static/scalar3d/downloads/Scalar3D-CUDA-Report.pdf" download="Scalar3D-CUDA-Report.pdf">下载完整 PDF 报告</a>
<a class="sw-button" href="../_static/scalar3d/downloads/REPORT.html" download="Scalar3D-CUDA-Report.html">下载原始 HTML 报告</a>
<a class="sw-button" href="../_static/scalar3d/downloads/Scalar3D-CUDA-Report.pdf" target="_blank" rel="noopener">新窗口打开 PDF ↗</a>
</nav>
```

两份输入 ZIP 都没有 PDF。这里的 25 页 PDF 由原始 HTML 的文字、表格和 21 张原图重新排版生成，并加入发布说明；没有重新绘制或美化反演图。原始中文 HTML 全文保持不变，包括其当时“尚未发布”的历史状态。

原报告结尾关于真值用途的概括需要补充：薄层初始模型也由真值做 Gaussian 平滑得到；固定顶部两层指 z=0、1 两个网格面。当前双语正文和 PDF 发布说明明确保留这两点。英文案例正文完整翻译了实验内容，历史报告本身仍为中文。

## 读取和复绘已保存结果

解压完整 ZIP，进入 `Scalar3D_Examples_20261008` 目录。先核对完整性，再用 CPU 读取或复绘。哈希校验确认文件完整，不等于重新验证传播器的数值正确性。

```bash
python verify_package.py
python render_scalar3d.py results/enclosed
python render_scalar3d.py results/surface
python render_scalar3d.py results/layered
```

CPU 复绘需要 NumPy、Matplotlib；Notebook 还使用 IPython/Jupyter。它不需要 GPU、StarWave 或训练 helper。Notebook 的已执行单元读取真实保存结果并复绘；CUDA 训练由配套 Python 脚本在 A30 上完成。单独一个 Notebook 或入口脚本不是完整运行包，必须保留相对目录。

Notebook 可选训练默认 `RUN_FWI = False`。打开报告不会启动训练；显式开启之前需要配置源码、GPU 和新的输出目录。

## 从头运行 CUDA 实验

从头训练需要 NumPy、SciPy、Matplotlib、PyTorch，以及与记录匹配的已编译 Scalar3D 源码。脚本还依赖同一源码中的 `example_support/scalar_examples.py`。它不在两份原始结果 ZIP 内，也未包含在公开结果包中；应使用自己有权访问的匹配源码。`source_identity.json` 保留所需 helper、关键源码和原运行编译库的 SHA-256。已有结果的 CPU 读取不受此依赖影响。

```bash
python run_scalar3d_example.py --case enclosed \
    --source-root /path/to/compiled/StarWave \
    --gpu-ids 2,0,1,3 \
    --output-dir /path/to/new_results
```

`enclosed` 和 `layered` 默认 100 轮，`surface` 默认 200 轮。使用新输出目录；脚本拒绝覆盖已有 `history.json`。GPU 数量改变会改变每轮 Adam 更新次数，因此不能要求不同卡数的完整优化轨迹一致。公开包不自动安装、编译或携带 GPU 二进制，也没有在本次发布中验证新的编译环境。

原始统一入口按可用 checkpoint 的四分位索引选择梯度：100 轮选择 `0,20,50,80,100`，200 轮选择 `0,50,100,150,200`。它们与历史图中的 checkpoint 不同。若要重现报告所选梯度位置，训练后显式执行：

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

这些是供读者选择运行的复现命令，不是本次手册整理重新执行的实验。训练循环及原始入口逻辑均保持原样。

## 数据、指标与可核对范围

- `case_manifest.json`：三组算例和训练脚本对应关系
- `results/<case>/summary.json`：配置、耗时、显存和初末模型指标
- `results/<case>/numerical_checks.json`：已记录的 full/boundary、内部方向导数和单/四卡检查
- `results/<case>/gradients/gradient_manifest.json`：梯度定义、checkpoint 模型哈希和评估 loss
- `validation_summary.json`：机器可解析的总体汇总
- `source_identity.json`：实际运行源码与编译库指纹
- `PACKAGE_MANIFEST.json`：本公开包的逐文件完整性校验

此次整理独立核对了全部 97 个 NPY 数组的类型与有限性、每个保存模型的 RMSE、15 组梯度的范数和 mask、Notebook 输出及 21 张报告图的原始像素。只保存了第 0 炮预测，因此可以核对该炮的残差指标，不能仅凭这些保存数组重新计算所有炮的完整目标或重新证明方向导数正确性。后两者属于原运行记录的证据。

模型保存顺序为 ZYX，脚本传播模型使用 XYZ，需要显式转置。独立归一化梯度图仅比较形态；共同原始色标才能比较幅值。完整数值检查定义和性能统计口径见[总览](index.md)。
