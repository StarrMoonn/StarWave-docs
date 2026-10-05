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

安装预编译 wheel 不需要 nvcc 或本地 CUDA Toolkit。Linux 驱动保守目标为 520.61.05 或更新，同时必须支持实际 GPU；WSL 使用 Windows 主机驱动，见 [WSL 说明](wsl.md)。A30 与 RTX 4060 Laptop 属于编译目标覆盖范围；公开 2.0.0 wheel 的实机数值与性能测试仍待完成。下文另列开发版 A30 实测记录。

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


(installation-smoke)=
## Notebook 01：实际运行检查

下载 {download}`01_installation_environment_smoke.ipynb <../examples/tutorials/01_installation_environment_smoke.ipynb>`，或 {download}`配套 Python 脚本 <../examples/tutorials/01_installation_environment_smoke.py>`。Notebook 还需要 NumPy、Matplotlib ≥ 3.6 和 Jupyter；在选定环境的独立终端安装，不在运行中的 kernel 内升级软件：

```bash
python -m pip install numpy "matplotlib>=3.6" jupyterlab ipykernel
python -m jupyter lab
```

选择对应的 CUDA kernel，从头运行全部单元。每份下载均是独立、清空输出的公开版；保留原始中文教学说明。数值实验未改动，环境来源字段及私有安装说明的整理范围见{ref}`来源记录 <tutorial-evidence>`。

```{admonition} 实测环境 · 2026-10-05
:class: sw-run-note
NVIDIA A30 · Python 3.10.18 · PyTorch 2.5.1 / CUDA 11.8 · 安装的 StarWave 版本 **0.1.0.dev9**。以下服务器结果验证的是这个开发版环境的小规模流程，不能据此宣称上述公开 **2.0.0 wheel** 已通过实机验收。
```

## 小规模实验设计

均匀速度 1800 m/s，模型 48 × 40、间距 10 m；一炮位于 (240, 50) m，12 个接收点位于 x=80…410 m、间隔 30 m、z=50 m。15 Hz Ricker 峰值时刻 0.055 s，dt=1 ms、160 个样点，最后样点在 0.159 s。

配置为 float32、accuracy=4、PML=12 格、boundary_buffer=5、memory="boundary"，固定 max_vel=2200 m/s。该运行的时间规划器返回内部步长 1 ms、子步比 1。四侧均使用 PML，没有自由表面。参数区为 `active[4:-4,8:-4]`，浅部和外缘保持固定。

```{figure} /_static/tutorials/installation_model.png
:alt: 1800 m/s 均匀模型、一炮十二道采集和固定边带，坐标以米表示
:class: sw-science-image
:figclass: sw-science-figure

最小实验的模型与几何。固定的源 forcing 由传播器施加内部注入系数，无需手动重复乘以 −v² dt²。
```

先执行一次 `scalar`，再取 `objective = record.square().mean()` 并执行 `objective.backward()`。这个能量型标量只用于检查反传接线，不是 FWI 数据失配或物理解的误差基准。

## 已得到的记录与检查结果

```{figure} /_static/tutorials/installation_gather.png
:alt: 一炮十二道实测合成记录，时间以秒、接收位置以米表示，振幅为任意单位
:class: sw-science-image
:figclass: sw-science-figure

服务器导出的炮集，采用完整零对称振幅范围；不截幅或逐道归一化，幅度单位不是校准后的 Pa。
```

| 检查项 | 实测结果 |
|---|---:|
| 记录形状 `[shot,receiver,time]` | `[1,12,160]` |
| 最大绝对振幅 / RMS | 25.1871452 / 4.5169382 |
| 记录平方均值 objective | 20.4027290 |
| 速度梯度 L2 范数 | 0.0019382348 |
| 活动区最大绝对梯度 | 0.0010059283 |
| 固定区最大绝对梯度 | 0 |

记录和梯度均为有限值，活动区梯度非零。**这是一项通过的小规模 smoke check，不是梯度数值正确性、FWI 收敛或多 GPU 验证。** 原生库准备成功、编译架构覆盖和上述数值实验属于不同证据层级。

继续运行[简单梯度计算](modeling/gradient.md)与[简单 FWI 示例](inversion/fwi.md)，或从[快速入门](quickstart.md)下载更短的命令行接线脚本。完整实验设置与精确数值见 {download}`结果摘要 <../examples/tutorials/results_summary.json>`。
