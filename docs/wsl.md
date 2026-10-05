# Windows 与 WSL 2

StarWave 2.0.0 没有原生 Windows wheel。在 Windows 上运行传播需要 WSL 2 的 Linux 用户环境、可用的 NVIDIA GPU 和支持 WSL GPU 计算的 Windows NVIDIA 驱动。

## 准备顺序

1. 按 [Microsoft WSL 安装说明](https://learn.microsoft.com/en-us/windows/wsl/install) 安装或更新 WSL，并确认发行版运行在 WSL 2。
2. 按 [NVIDIA CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) 安装 Windows NVIDIA 驱动。不要在 WSL 中另装 Linux 显卡内核驱动。
3. 在 Linux 终端检查架构和运行库，再按[安装页](installation.md)创建 Python 环境。

```bash
uname -m
getconf GNU_LIBC_VERSION
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

预期架构为 `x86_64`，glibc 至少为 2.35，PyTorch 能看到 CUDA GPU。`nvidia-smi` 的可用字段在 WSL 下可能与原生 Linux 不同；仍需实际运行 `prepare_native` 和小规模传播检查。

## 常见边界

RTX 4060 Laptop 的 sm_89 已列入 wheel 编译目标，但这不证明 WSL 驱动、显存、数值误差或速度验收完成。Windows 中的 Python 环境和 WSL Linux Python 环境是两个独立环境；应在 WSL 环境中安装并运行 StarWave。

WSL 的具体驱动兼容要求会变化，以 [Microsoft GPU 计算说明](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute) 与 NVIDIA 指南为准。本草稿未执行目标机器的 WSL 验收。
