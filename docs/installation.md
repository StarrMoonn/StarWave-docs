# 安装

## 运行条件

| 项目 | StarWave 2.0.0 wheel 要求 |
|---|---|
| 系统 | Linux x86_64，glibc ≥ 2.35 |
| C++ 运行库 | libstdc++ 提供 GLIBCXX_3.4.30 / CXXABI_1.3.13 或更新符号 |
| Python | 3.10–3.12 |
| PyTorch / NumPy | PyTorch 2.5.x；NumPy ≥ 1.23 |
| GPU | NVIDIA CUDA GPU；传播没有 CPU 后备实现 |
| 编译目标 | SASS 70/75/80/86/89/90 与 compute80 PTX |

安装预编译 wheel 不需要 nvcc 或本地 CUDA Toolkit。Linux 驱动保守目标为 520.61.05 或更新，同时必须支持实际 GPU；WSL 使用 Windows 主机驱动，见 [WSL 说明](wsl.md)。A30 与 RTX 4060 Laptop 属于编译目标覆盖范围，仍待实机数值与性能测试。

以上版本信息来自 [PyPI 2.0.0 发布说明](https://pypi.org/project/starwave/2.0.0/)。公开发行仅有 Linux wheel，没有 sdist；原生 Windows、macOS、ARM 主机不能直接使用这一 wheel。

## 新建运行环境

以下命令在符合条件的 Linux 或 WSL 2 Linux 终端中执行。选择 Python 3.10–3.12；示例使用 3.12。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --only-binary=starwave starwave==2.0.0
python -m pip check
```

PyTorch 安装命令依据 [PyTorch 历史版本页面](https://pytorch.org/get-started/previous-versions/)。`cu118` 是此版本的起始验证配置，不代表其它 PyTorch 版本已被验证。升级 StarWave 后重启 Python 或 Notebook kernel。

## 检查导入与原生库

```python
import torch
import starwave
from importlib.metadata import version

print(version("starwave"))
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available())
status = starwave.native_status()
print({key: status[key] for key in ("library_exists", "library_loaded")})
# 在主线程中，使用当前进程可见的逻辑设备编号：
starwave.prepare_native([0])
```

在全新 Python 进程、首次原生准备之前，wheel 应报告库文件存在且尚未加载。`prepare_native` 成功表示初始化与兼容检查通过，不能代替正演与梯度数值验收。随后运行[快速入门](quickstart.md)的小规模检查。

文档站本身的构建环境独立于计算环境；仅阅读或构建文档不需要 GPU。
