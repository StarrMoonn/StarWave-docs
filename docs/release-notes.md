# 发布说明

## 4.0.0 · 2026-10-07

公开 [StarWave 4.0.0 wheel](https://pypi.org/project/starwave/4.0.0/) 对应 V11，新增基于 Deepwave 0.0.27 派生后端的 `starwave.elastic(lamb, mu, buoyancy, ...)`，支持二维/三维、2/4/6/8 阶及完整状态/记录返回。材料转换通过 `starwave.common` 显式完成；elastic 使用独立的 `prepare_elastic`。原有 scalar 仍为二维，scalar/VRZ/VTI 的公开签名与默认值保留。

## Elastic API 文档维护 · 2026-10-08

在原 Usage 单页内新增 Elastic Function、62 项参数、16/31 返回顺序、轴序和最小示例，并补充两种材料转换与 prepare_elastic。核对对象为 4.0.0 发布 wheel；审阅明确区分 full/boundary、采样间隔、survey_pad 和已有验证范围。同步安装版本与双语 API 契约检查，保留原 scalar/VRZ/VTI 正文、历史教程数值、版权署名、字体和导航。

本次仅维护文档，不修改传播运行时或发行产物，也未执行 GPU 数值/性能测试。核对来源、wheel 哈希与验收边界见[文档状态](status.md)。

## 2.0.0 · 2026-10-05

公开发行 Linux x86_64 预编译 wheel。该版本改进 VTI 二维/三维 boundary backward 的融合路径；公开 Python API 与默认值不变，scalar/VRZ 不因版本号升为 2.0 而增加新能力。浮点运算顺序变化意味着不能要求与前版逐位相同。

升级后应重启解释器或 Notebook kernel。此版本号不表示新增了目标 GPU 数值、多卡、FWI 或性能验收。已知模型扩展梯度、VRZ 强反差稳定性和 VTI `epsilon < delta` 限制仍需遵守。

安装条件和完整发布范围以 [PyPI 2.0.0](https://pypi.org/project/starwave/2.0.0/) 为公开版本来源。

## 本使用手册首稿

新增安装/WSL、合成 scalar 示例、三类方程使用说明、FWI/DataParallel/INR 入门、API 索引与 FAQ。初稿采用 Sphinx + Read the Docs 主题，可输出到 GitHub Pages 或其它静态服务器；当前双语版保留这一初始模板。

本手册首稿不附带 CUDA 实现、传播 Python 实现、私人实验数据或内部记录。待完成的教程与验收见[文档状态](status.md)。

## API 参考完善

scalar、VRZ、VTI 各自提供独立参考页，以 Sphinx Python 域展示完整签名、逐项参数类型/默认值/形状/单位、返回结构、自动微分范围及示例。补充 54 项参数说明；保留原 API 索引中的函数入口锚点。此为文档改进，不改变 StarWave 2.0.0 的运行接口或验收状态。

## 中英双语版

完整手册提供中文与 English，包括导航、搜索、提示与 18 个内容页面。各语言独立构建和索引，语言切换保留对应页面与章节；旧中文链接与 API 锚点保持可访问。站点保留最初的 PyFWI / Read the Docs 模板，API 内容组织参考 Deepwave 的 Sphinx 结构，增加类型签名，并继续保留 StarWave 自身的返回与数值限制。
