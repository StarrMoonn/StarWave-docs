# API 索引

本页记录 2.0.0 wheel 中已核验的常用公开接口。签名静态核对，无需构建站点时导入软件或加载 CUDA。这里只写调用契约，不发布实现代码。

## `starwave.scalar`

```{py:function} starwave.scalar(v, grid_spacing, dt, *, source_amplitudes, source_locations, receiver_locations, accuracy=8, pml_freq=25.0, pml_width=20, boundary_buffer=5, memory="boundary", max_vel=None, freq_taper_frac=0.0, time_pad_frac=0.0, time_taper=False, illumination=None)

二维标量声学；返回单元素元组 `(records,)`，记录为 `[B,R,T]`。模型与源单位见 [scalar](../modeling/scalar.md)。
```

## `starwave.vrz`

```{py:function} starwave.vrz(v, grid_spacing, dt, *, impedance=None, density=None, source_amplitudes, source_locations, receiver_locations, accuracy=8, pml_freq=25.0, pml_width=20, boundary_buffer=5, memory="boundary", max_vel=None, freq_taper_frac=0.0, time_pad_frac=0.0, time_taper=False, illumination=None)

二维变密度声学；`impedance` 与 `density` 恰好提供一个。返回 `(records,)`。`illumination` 必须为 `None`。见 [VRZ](../modeling/vrz.md)。
```

## `starwave.vti`

```{py:function} starwave.vti(vp, epsilon, delta, rho, grid_spacing, dt, *, source_amplitudes, source_locations, receiver_locations, source_fields=("sH", "sV"), receiver_fields=("vz",), accuracy=4, pml_freq=25.0, pml_width=20, boundary_buffer=5, memory="boundary", max_vel=None, freq_taper_frac=0.0, time_pad_frac=0.0, time_taper=False)

二维/三维声学 VTI；返回元组按 `receiver_fields` 顺序排列，各项为 `[B,R,T]`。见 [VTI](../modeling/vti.md)。
```

## 公共参数速查

| 参数 | 含义与条件 |
|---|---|
| `grid_spacing`, `dt` | 正网格间距、正用户时间步；不参与求导 |
| `source_amplitudes` | `[B,S,T]` 固定源；VTI 与 scalar/VRZ 的单位不同 |
| `source_locations`, `receiver_locations` | 整数物理网格下标，详见[约定](../modeling/conventions.md) |
| `accuracy` | 空间阶数 2、4、6、8；scalar/VRZ 默认 8，VTI 默认 4 |
| `pml_freq` | 正频率，默认 25.0 Hz；不是波形生成器 |
| `pml_width` | 正整数或相同面宽序列；默认 20 个网格 |
| `boundary_buffer` | 非负空间网格数，默认 5；boundary 有最小值要求 |
| `memory` | `"boundary"` 或 `"full"`；默认 boundary |
| `max_vel` | 默认 `None`，每次调用动态规划；显式值必须覆盖当前所需包络 |
| `freq_taper_frac`, `time_pad_frac` | [0,1] 内的有限比例，默认 0.0；影响时间重采样 |
| `time_taper` | 布尔值，默认 False；时间重采样的 Hann taper |

三个 taper/padding 参数改变信号处理，不是单位转换；内部不需要时间重采样时有效参数不产生作用。没有公开 `nt` 参数，用户时间长度取自源的最后一维。

## 原生运行库

```{py:function} starwave.native_status()

返回状态字典，不编译、不传播。入门检查可读取 `library_exists` 和 `library_loaded`。库存在或已加载不等于数值验收通过。
```

```{py:function} starwave.prepare_native(device_ids)

在主线程准备已存在的兼容原生库。`device_ids` 是非空、无重复、非负的可见逻辑 CUDA 编号列表或元组；返回准备状态字典，不执行编译。
```

## 其它已导出的名称

`ScalarIllumination`、`IlluminationFields`、`precondition_gradient` 为可选 scalar 照明与显式梯度预条件相关接口。完整签名、生命周期和独立示例尚未纳入本入门版索引，不建议仅凭名称推断调用方法。不存在本手册可用的 `starwave.Scalar` 类；文中 Module wrapper 均由教程自定义。

{ref}`genindex`
