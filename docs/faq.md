# 常见问题

当前源码手册覆盖已发布 V17 的 Born、平顶边界与导入契约。**PyPI 8.0.0 尚未发布，7.1.0 仍为 V16。** V17 功能需要匹配的授权源码，见[安装](installation.md)与 [Born Function](usage.md#born-function)。

V16 新增 [PyTorch 后端](pytorch-backend.md)，可在 CPU/CUDA 上显式使用 full/checkpoint，无需传播原生库；MPS 仅支持 Scalar2D accuracy=4 float32，使用 backend="torch"、full/checkpoint 与 eager/compile；Scalar3D、VRZ、VTI、elastic、GSLS 和其他 MPS 组合明确拒绝。本次不宣称最终 Mac 实机、500 轮 FWI 或性能验收通过。 AMP 仍不支持。[PyPI 7.1.0](https://pypi.org/project/starwave/7.1.0/) 已发布，仅适用于 Linux x86_64。Mac MPS 需使用已获授权的完整源码。下方 wheel 平台和 CUDA 载入问答描述当前 Linux wheel 与原生路径。Torch 显存不足时可评估 checkpoint 间隔，较大间隔不保证更低峰值。

## 为什么 pip 找不到可用的 StarWave 版本？

检查 Linux x86_64、Python 3.10–3.12 与 glibc ≥2.35，并更新 pip。Windows 原生 Python、macOS、ARM 或较旧 glibc 不匹配这个 wheel。不要用修改 wheel 文件名的方式绕过平台标签。

## 需要编译 CUDA 吗？

使用公开 7.1.0 wheel 不需要编译或安装 nvcc。原生 scalar/VRZ/VTI 和 elastic boundary 需要兼容 GPU/驱动；elastic full 与 GSLS native_cpu 也可在 CPU 上运行。显式 Torch full/checkpoint 路径无需传播原生库。如果完整性或兼容检查失败，在相同环境重装对应 wheel 并重启 Python；不要指望文档项目包含私有构建工具。

## import 成功，为什么 prepare_native 失败？

V17 导入会准备已有原生库与共享 dispatch，但不编译、不初始化 CUDA，也不传播。显式 prepare helper 仍可用于设备检查；import 成功不证明 CUDA 设备可用。未构建的库可跳过以使用受支持的显式 Torch，但保留的无效库/构建会报错，不静默忽略。导入前配置构建路径，更新后重建匹配源码并重启。较早的 7.1.0/V16 wheel 使用显式原生准备；其 CUDA 路径需检查 PyTorch 2.5.x 的 CUDA 可用性、逻辑设备号、C++ 运行库、驱动与包完整性。WSL 使用 Windows 主机驱动，不宜把加载错误直接归因于架构未覆盖。

## Born 在哪里，如何构建？

Born 属于 V17 源码（`starwave.scalar_born` / `starwave.ScalarBorn`），原生库独立。冻结 V17 的三库 `compile_all.py` 之后须单独运行 `python compile_born.py --arch 80`；后续已审 dev `3f2b24d` 的统一构建已覆盖包括 Born 的四库。`python compile_born.py --cpu-only` 仍是独立 CPU 选项。显式 Torch Born CPU/CUDA full/checkpoint 不需要 Born 原生构建；VRR 扩展仍使用自己的安装包与构建入口。7.1.0 wheel 不包含这些新增源码能力。

## 将 pml_width 设为零，各方程的自由表面相同吗？

不相同。V17 按方程限定逐面宽度与物理闭合。scalar/VRZ 只允许最后一轴顶部零宽，其余面等宽且为正；外部 halo 固定为零，不是奇反射表面，scalar 同时拒绝 illumination。VTI 为顶部零法向牵引，elastic 为第一轴改进真空表面，GSLS 为二维压力释放闭合。Born 独立接受逐面非负宽度，零面使用固定零 ghost。请遵守[模型约定](modeling/conventions.md)及各自 [Usage](usage.md) 契约，不能把 Born 较宽的宽度取值域套给 scalar。

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
