# API 参考

本参考以 StarWave **2.0.0** 公开 wheel 为准。每个传播函数提供真实签名、逐项参数、返回值、梯度范围、注意事项与调用示例。参数类型描述运行时接受的值；签名保留实际关键字边界和默认值。

页面组织参考 [Deepwave 官方 Usage](https://ausargeo.com/deepwave/usage) 的 Sphinx Python API 风格，正文按 StarWave 的实际契约重新编写。两者的参数集合、源单位、返回结构和可微范围不能互换。

```{toctree}
:maxdepth: 2

scalar
vrz
vti
```

## 传播函数速查

<span id="starwave.scalar"></span>

- {py:func}`starwave.scalar`：二维标量声学；速度模型 `v`；返回单元素记录元组。

<span id="starwave.vrz"></span>

- {py:func}`starwave.vrz`：二维变密度声学；`v` 加恰好一种 `impedance` / `density` 参数化。

<span id="starwave.vti"></span>

- {py:func}`starwave.vti`：二维/三维声学 VTI；`vp, epsilon, delta, rho`；按所选分量顺序返回记录。

符号约定：`B` 炮数、`S` 每炮源数、`R` 每炮接收点数、`T` 用户时间采样数、`D` 空间维数。源和接收点使用物理模型的整数网格下标，不是米坐标，不包含 PML 偏移。

## 原生运行库

```{py:function} starwave.native_status()

查询当前运行库状态，不执行编译或传播。

:returns: 状态字典；常用条目包括 `library_exists`、`library_loaded`、`torch_version`、`torch_cuda_version` 和 `cuda_available`。存在或加载成功均不代表数值验收通过。
:rtype: `dict`
```

```{py:function} starwave.prepare_native(device_ids)

在主线程验证并预加载已有的原生库，供后续传播或 DataParallel 使用，不执行编译。

:param device_ids: 必填。非空、无重复、非负的可见逻辑 CUDA 编号；拒绝布尔值。编号按当前进程的设备可见性映射填写。
:type device_ids: `list[int] | tuple[int, ...]`
:returns: 包含原生状态、`selected_device_ids` 与准备消息的字典。初始化成功不是 GPU 数值测试。
:rtype: `dict`
```

## 其它导出名称与范围

`ScalarIllumination`、`IlluminationFields`、`precondition_gradient` 是已核验导出的 scalar 照明相关名称。本版暂不提供它们的完整生命周期教程；传播函数页会解释 `illumination` 参数的使用边界。

StarWave 2.0.0 没有公开 `starwave.Scalar` 包装类。教程中的 Module wrapper 由教程定义；不能将其它库的类名、状态参数、`nt` 或存储选项直接添加到这里的调用中。

{ref}`genindex`
