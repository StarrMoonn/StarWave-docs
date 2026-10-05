# 发布说明

## 2.0.0 · 2026-10-05

公开发行 Linux x86_64 预编译 wheel。该版本改进 VTI 二维/三维 boundary backward 的融合路径；公开 Python API 与默认值不变，scalar/VRZ 不因版本号升为 2.0 而增加新能力。浮点运算顺序变化意味着不能要求与前版逐位相同。

升级后应重启解释器或 Notebook kernel。此版本号不表示新增了目标 GPU 数值、多卡、FWI 或性能验收。已知模型扩展梯度、VRZ 强反差稳定性和 VTI `epsilon < delta` 限制仍需遵守。

安装条件和完整发布范围以 [PyPI 2.0.0](https://pypi.org/project/starwave/2.0.0/) 为公开版本来源。

## 本使用手册首稿

新增安装/WSL、合成 scalar 示例、三类方程使用说明、FWI/DataParallel/INR 入门、API 索引与 FAQ。站点采用 Sphinx + Read the Docs 主题，可输出到 GitHub Pages 或其它静态服务器。

本手册首稿不附带 CUDA 实现、传播 Python 实现、私人实验数据或内部记录。待完成的教程与验收见[文档状态](status.md)。
