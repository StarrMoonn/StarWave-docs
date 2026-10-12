# 常见问题

[V19 Release](https://github.com/StarrMoonn/StarWave/releases/tag/V19) 已发布；本手册对应冻结源码 `0.1.0.dev19`、提交 `46d32877e3336d531520a696d5f2e074e6f4ef74`。[PyPI **9.1.0**](https://pypi.org/project/starwave/9.1.0/) 已发布，对应 V19 源码契约并包含集成 VRR。现有 [9.0.0 wheel](https://pypi.org/project/starwave/9.0.0/) 仍对应 V18，不含集成 VRR。

[PyTorch 后端](pytorch-backend.md)支持显式 CPU/CUDA full/checkpoint，无需传播原生构建。MPS 仍仅限 Scalar2D accuracy=4 float32，需要源码安装；AMP 不支持。已发布 9.0.0 Linux wheel 对应 V18；在 9.1.0 发布核验前，VRR 需使用 V19 源码。更大的 checkpoint 间隔不保证更低峰值内存。

## 为什么 pip 找不到可用的 StarWave 版本？

检查 Linux x86_64、Python 3.10–3.12 与 glibc ≥2.35，并更新 pip。Windows 原生 Python、macOS、ARM 或较旧 glibc 不匹配这个 wheel。不要用修改 wheel 文件名的方式绕过平台标签。

## 需要编译 CUDA 吗？

使用已发布 9.0.0 wheel 不需要编译或安装 nvcc。原生 scalar/VRZ/VTI 和 elastic boundary 需要兼容 GPU/驱动；elastic full 与 GSLS native_cpu 也可在 CPU 上运行。显式 Torch full/checkpoint 路径无需传播原生库。如果完整性或兼容检查失败，在相同环境重装对应 wheel 并重启 Python；不要指望文档项目包含私有构建工具。

## import 成功，为什么 prepare_native 失败？

V19 导入会准备已有原生库与共享 dispatch，但不编译、不初始化 CUDA，也不传播。显式 prepare helper 仍可用于设备检查；import 成功不证明 CUDA 设备可用。未构建的库可跳过以使用受支持的显式 Torch，但保留的无效库/构建会报错，不静默忽略。导入前配置构建路径，更新后重建匹配源码并重启。较早的 7.1.0/V16 wheel 使用显式原生准备；其 CUDA 路径需检查 PyTorch 2.5.x 的 CUDA 可用性、逻辑设备号、C++ 运行库、驱动与包完整性。WSL 使用 Windows 主机驱动，不宜把加载错误直接归因于架构未覆盖。

## Born 在哪里，如何构建？

V19 根包包含 `starwave.scalar_born` / `starwave.ScalarBorn` 与 `starwave.vrr`。`python compile_all.py --arch 80` 构建 core、elastic、GSLS、Born、VRR。`python compile_born.py --cpu-only` 为原生 CPU full 选项；`python -m starwave._vrr.build --arch 80` 可仅重建 VRR。显式 Torch Born/VRR CPU/CUDA full/checkpoint 无需原生构建。见[安装](installation.md)。

## 将 pml_width 设为零，各方程的自由表面相同吗？

不相同。V19 scalar 接受任意零宽面组合（含全零 PML），其余正宽度须相等；零面使用固定零 halo，不是奇反射。VRZ 仍限最后一轴顶部零宽。VTI 使用零法向牵引，elastic 为第一轴改进真空表面，GSLS 为二维压力释放闭合，VRR 保留自己的顶部表面限制。Born 独立接受逐面非负宽度与固定零 ghost。请遵守[模型约定](modeling/conventions.md)及各自 [Usage](usage.md) 契约。

V19 保留 V18 的 23 参数 scalar 契约：任意零宽面组合（正宽度仍须相等）、二维/三维可训练源、可微最终物理波场与诊断前向回调。终态可用于 loss，回调快照为 detached 观测，不能传递 loss 梯度。见 [Scalar Function](usage.md#scalar-function)。



## A30 / RTX 4060 Laptop 可以运行吗？

对应 sm_80 / sm_89 已被编译覆盖；尚不能据此宣布实机数值正确、稳定或性能合格。请在实际系统中逐步完成加载、小模型记录、梯度、长时序及多卡测试。

## 波形首样本不严格为零，或振幅与另一个软件不同？

内部时间重采样可能出现 FFT 端点振铃。还应核对源单位、采样时钟、记录物理分量与空间交错位置；VTI 源和 scalar 源不能直接互换。相似的 Python 调用格式不保证与其它软件数值等价。

## 为什么有六点每波长的警告？

这是空间分辨率诊断；`pml_freq` 是频率代理，不是对实际波形频谱的自动测量。减小 `dt` 不会改善网格空间分辨率。应根据目标频带和模型速度重新评估网格。

## 显存不足怎么办？

先减少当前批次炮数、模型尺寸或时间长度，评估 full/boundary 的资源差别。多卡按炮拆分，单炮自身过大仍可能失败；库不会自动缩小实验参数。不要把减小配置后通过称为原始配置验收。

## loss 有限，为什么 backward 仍失败？

有限 loss 不保证其记录梯度有限。检查目标函数归一化、非线性与精度；一次 forward 仅支持一次一阶 backward。也应单独检查参数合法域、传播稳定性和模型梯度。不能用静默替换 NaN/Inf 证明计算正确。

## 文档构建需要 GPU 吗？

不需要。Sphinx 只渲染手写文档与示例文本，不导入 StarWave 或运行 CUDA。
