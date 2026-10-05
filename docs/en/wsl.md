# Windows and WSL 2

StarWave 2.0.0 has no native Windows wheel. Running propagation on Windows requires a WSL 2 Linux user environment, an available NVIDIA GPU, and a Windows NVIDIA driver that supports GPU computing in WSL.

## Setup order

1. Install or update WSL using the [Microsoft WSL installation instructions](https://learn.microsoft.com/en-us/windows/wsl/install), and confirm that your distribution is running on WSL 2.
2. Install the Windows NVIDIA driver according to the [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). Do not install a separate Linux graphics kernel driver inside WSL.
3. Check the architecture and runtime libraries in a Linux terminal, then create a Python environment using the [installation page](installation.md).

```bash
uname -m
getconf GNU_LIBC_VERSION
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The expected architecture is `x86_64`, glibc must be at least 2.35, and PyTorch must be able to see a CUDA GPU. The fields available in `nvidia-smi` under WSL may differ from those on native Linux. You still need to run `prepare_native` and a small propagation check.

## Important limitations

The RTX 4060 Laptop's sm_89 is included in the wheel's compilation targets. This does not establish that WSL driver compatibility, GPU memory requirements, numerical error, or speed have been validated. The Windows Python environment and the WSL Linux Python environment are separate. Install and run StarWave inside the WSL environment.

Specific WSL driver compatibility requirements can change; refer to the [Microsoft GPU computing guidance](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute) and the NVIDIA guide. WSL validation on the target machine has not been performed for this draft.
