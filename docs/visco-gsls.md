(visco-gsls)=
# GSLS 黏声学 API（V16）

本页对应 V16 源码 `0.1.0.dev16`。二维 `starwave.visco_gsls` 是当前唯一黏声入口；单机制 SLS 通过显式 `mode="sls_compat", n_mechanisms=1` 使用。独立 `visco_sls`、`prepare_visco_sls` 和 `visco_sls_native_status` 已移除。既有公开 PyPI `7.0.0` 不包含 GSLS；更新此手册不表示发布了新 wheel。详见[文档状态](status.md)及[安装](installation.md)。

支持固定空间变密度、float32/float64、Vp/Q/source/provider 参数一阶梯度。原生 CPU/CUDA 支持 full/checkpoint；显式 Torch 支持 full/checkpoint 与 eager/compile。所有主传播入口均新增执行选项，原生默认行为保持；详见 [PyTorch 后端](pytorch-backend.md)。

(visco-gsls-function)=
## visco_gsls

```{py:function} starwave.visco_gsls(vp: torch.Tensor, q: torch.Tensor, rho: torch.Tensor, grid_spacing: float | int | list[float] | tuple[float, float], dt: float | int, *, source_amplitudes: torch.Tensor, source_locations: torch.Tensor, receiver_locations: torch.Tensor, f_ref: float | int, accuracy: int=4, pml_width: int | list[int] | tuple[int, int]=20, memory: str='full', max_vel: float | int | None=None, backend: str='cuda', mode: str='hao1', frequency_band: tuple[float, float] | list[float] | None=None, n_mechanisms: int | None=None, fit_tolerance: float | int | None=0.05, strict: bool=False, material_model: Callable | None=None, checkpoint_interval: int | None=None, execution: str='eager', compile_steps: int=1) -> tuple[torch.Tensor]

:param vp: **必填；位置或关键字。** `[nz,nx]` float32/float64 正有限张量，两个维度均至少 2；单位 m/s，严格为 `f_ref` 处相速度，可求导。
:type vp: `torch.Tensor`
:param q: **必填；位置或关键字。** 与 Vp 同形状/dtype/device，正有限或 +inf。Hao1 为名义 Q0，要求 Q0>S，不能称为精确 Qref；band_fit 为频带拟合目标；sls_compat 为参考频率精确模量 Q。可求导。
:type q: `torch.Tensor`
:param rho: **必填；位置或关键字。** 与模型匹配，正有限 kg/m³，倒数须可表示。支持空间变化，但 `requires_grad=True` 拒绝。
:type rho: `torch.Tensor`
:param grid_spacing: **必填；位置或关键字。** 正有限标量或 `(dz,dx)` tuple/list，单位 m。
:type grid_spacing: `float | int | list[float] | tuple[float, float]`
:param dt: **必填；位置或关键字。** 正有限秒；必须满足含未松弛速度和密度对比的保守 CFL 界，不自动重采样或子步进。
:type dt: `float | int`
:param source_amplitudes: **必填；仅关键字。** 与模型匹配的有限 `[B,S,T]` 张量，各维非空；单位 Pa/m²，二阶 forcing，可求导。
:type source_amplitudes: `torch.Tensor`
:param source_locations: **必填；仅关键字。** int32/int64 `[B,S,2]`，原始未扩边模型内 `[z,x]` 整数坐标；重复震源相加，无单源省略维度格式。
:type source_locations: `torch.Tensor`
:param receiver_locations: **必填；仅关键字。** int32/int64 `[B,R,2]`，R≥1；原模型坐标；重复接收点重复记录，反传 cotangent 累加。
:type receiver_locations: `torch.Tensor`
:param f_ref: **必填；仅关键字。** 正有限 Hz；Vp 的精确相速度参考频率，也参与固定 CPML setup。
:type f_ref: `float | int`
:param accuracy: **默认 `4`；仅关键字。** 空间 staggered FD 阶数 2/4/6/8，不是时间阶数。
:type accuracy: `int`
:param pml_width: **默认 `20`；仅关键字。** 非负整数或 `(z,x)`，单位网格点，各轴两侧对称；0 禁用对应轴 PML。replicate 扩边梯度完整累加。
:type pml_width: `int | list[int] | tuple[int, int]`
:param memory: **默认 `'full'`；仅关键字。** native CPU/CUDA 支持 `full` 或 `checkpoint`；Torch 也支持 `full` / `checkpoint`；不支持 boundary。
:type memory: `str`
:param max_vel: **默认 `None`；仅关键字。** 正有限 m/s 未松弛速度上界。任何可训练物性输出且启用非零 PML 时须显式固定；不能只覆盖 Vp。过小拒绝。
:type max_vel: `float | int | None`
:param backend: **默认 `'cuda'`；仅关键字。** 显式 `cuda`、`native_cpu` 或 `torch`；原生后端须匹配设备与已构建库，无回退或运行时编译。
:type backend: `str`
:param mode: **默认 `'hao1'`；仅关键字。** 默认 Hao 一阶 inverse-Q 模型，或显式 `band_fit`、`sls_compat`。不含论文二阶耦合模型。
:type mode: `str`
:param frequency_band: **默认 `None`；仅关键字。** 内置 Hao1 必须显式提供 `(scale,200*scale)` Hz 设计频带；band_fit 必须提供正递增拟合频带；不能把震源工作子频带当作 Hao1 任意带宽重新拟合。
:type frequency_band: `tuple[float, float] | list[float] | None`
:param n_mechanisms: **默认 `None`；仅关键字。** Hao1 默认 5，显式也只能 5；band_fit 默认 3，允许整数 2–6；sls_compat 须显式 1。
:type n_mechanisms: `int | None`
:param fit_tolerance: **默认 `0.05`；仅关键字。** 正相对 inverse-Q 容差；超过时默认 warning；None 跳过检查。采样诊断不构成全连续频率误差证明。
:type fit_tolerance: `float | int | None`
:param strict: **默认 `False`；仅关键字。** 超过拟合容差时是否由 warning 改为报错。
:type strict: `bool`
:param material_model: **默认 `None`；仅关键字。** 高级 callable `(vp_padded,q_padded,dt,f_ref)->(c_rel2,strength,A,B)`，接管内置物性选项，详见下文。
:type material_model: `Callable | None`
:param checkpoint_interval: **默认 `None`；仅关键字。** 仅 checkpoint 下可显式给正整数分段长度；原生 None 用形状/所需历史估计内存平衡；Torch None 固定为 32 个内部步，均不按 GPU 型号调参。full 下提供此参数会报错。
:type checkpoint_interval: `int | None`
:param execution: **默认 `'eager'`；仅关键字参数。** Torch 可选 `'eager'` 或 `'compile'`；compile 使用 PyTorch Inductor/AOTAutograd，失败不回退 eager。原生仅接受 eager。
:type execution: `str`
:param compile_steps: **默认 `1`；仅关键字参数。** 正整数，拒绝 bool；Torch 每个执行块包含的连续内部时间步数。compile 可尝试 4，完整处理尾块；不改变 dt、nt、炮数或损失。原生只接受 1。
:type compile_steps: `int`
:returns: 单元素元组 `(pressure_records,)`，形状 `[B,R,T]`、单位 Pa，与模型 dtype/device 一致。
:rtype: `tuple[torch.Tensor]`
```

<!-- api-doc:visco_gsls:signature:start -->
```text
starwave.visco_gsls(vp, q, rho, grid_spacing, dt, *, source_amplitudes,
           source_locations, receiver_locations, f_ref, accuracy=4,
           pml_width=20, memory="full", max_vel=None, backend="cuda",
           mode="hao1", frequency_band=None, n_mechanisms=None,
           fit_tolerance=0.05, strict=False, material_model=None,
           checkpoint_interval=None, execution="eager", compile_steps=1)
```
<!-- api-doc:visco_gsls:signature:end -->

前五项为位置或关键字参数；其余仅关键字。`source_amplitudes`、`source_locations`、`receiver_locations`、`f_ref` 必填。下表覆盖完整参数与默认值。

<!-- api-doc:visco_gsls:parameters:start -->
| Parameter / 参数 | Calling kind | Default when omitted | Type, shape and units | Purpose and behavior | Allowed values, rejections and limits |
|---|---|---|---|---|---|
| `vp` | `positional-or-keyword` | `required` | Tensor [nz,nx]; m/s | Reference phase speed / 参考相速度 | `[nz,nx]` float32/float64 正有限张量，两个维度均至少 2；单位 m/s，严格为 `f_ref` 处相速度，可求导。 |
| `q` | `positional-or-keyword` | `required` | Tensor [nz,nx]; dimensionless | Mode-specific Q / 按模式定义的 Q | 与 Vp 同形状/dtype/device，正有限或 +inf。Hao1 为名义 Q0，要求 Q0>S，不能称为精确 Qref；band_fit 为频带拟合目标；sls_compat 为参考频率精确模量 Q。可求导。 |
| `rho` | `positional-or-keyword` | `required` | Tensor [nz,nx]; kg/m³ | Fixed density / 固定密度 | 与模型匹配，正有限 kg/m³，倒数须可表示。支持空间变化，但 `requires_grad=True` 拒绝。 |
| `grid_spacing` | `positional-or-keyword` | `required` | real or pair; m | Grid spacing / 网格间距 | 正有限标量或 `(dz,dx)` tuple/list，单位 m。 |
| `dt` | `positional-or-keyword` | `required` | real; s | Fixed time step / 固定时间步 | 正有限秒；必须满足含未松弛速度和密度对比的保守 CFL 界，不自动重采样或子步进。 |
| `source_amplitudes` | `keyword-only` | `required` | Tensor [B,S,T]; Pa/m² | Second-order forcing / 二阶 forcing | 与模型匹配的有限 `[B,S,T]` 张量，各维非空；单位 Pa/m²，二阶 forcing，可求导。 |
| `source_locations` | `keyword-only` | `required` | int32/int64 Tensor [B,S,2]; cells | Unpadded source indices / 原域源坐标 | int32/int64 `[B,S,2]`，原始未扩边模型内 `[z,x]` 整数坐标；重复震源相加，无单源省略维度格式。 |
| `receiver_locations` | `keyword-only` | `required` | int32/int64 Tensor [B,R,2]; cells | Unpadded receiver indices / 原域接收坐标 | int32/int64 `[B,R,2]`，R≥1；原模型坐标；重复接收点重复记录，反传 cotangent 累加。 |
| `f_ref` | `keyword-only` | `required` | real; Hz | Phase reference / 相速度参考频率 | 正有限 Hz；Vp 的精确相速度参考频率，也参与固定 CPML setup。 |
| `accuracy` | `keyword-only` | `4` | integer; dimensionless | Staggered spatial order / 空间阶数 | 空间 staggered FD 阶数 2/4/6/8，不是时间阶数。 |
| `pml_width` | `keyword-only` | `20` | integer or pair; cells | Symmetric PML / 对称 PML | 非负整数或 `(z,x)`，单位网格点，各轴两侧对称；0 禁用对应轴 PML。replicate 扩边梯度完整累加。 |
| `memory` | `keyword-only` | `'full'` | str | Native history strategy / 原生历史策略 | native CPU/CUDA 支持 `full` 或 `checkpoint`；Torch 也支持 `full` / `checkpoint`；不支持 boundary。 |
| `max_vel` | `keyword-only` | `None` | None or real; m/s | Unrelaxed envelope / 未松弛速度包络 | 正有限 m/s 未松弛速度上界。任何可训练物性输出且启用非零 PML 时须显式固定；不能只覆盖 Vp。过小拒绝。 |
| `backend` | `keyword-only` | `'cuda'` | str | Explicit implementation / 显式实现 | 显式 `cuda`、`native_cpu` 或 `torch`；原生后端须匹配设备与已构建库，无回退或运行时编译。 |
| `mode` | `keyword-only` | `'hao1'` | str | Material mapping / 物性映射 | 默认 Hao 一阶 inverse-Q 模型，或显式 `band_fit`、`sls_compat`。不含论文二阶耦合模型。 |
| `frequency_band` | `keyword-only` | `None` | None or positive pair; Hz | Design/fitting band / 设计或拟合频带 | 内置 Hao1 必须显式提供 `(scale,200*scale)` Hz 设计频带；band_fit 必须提供正递增拟合频带；不能把震源工作子频带当作 Hao1 任意带宽重新拟合。 |
| `n_mechanisms` | `keyword-only` | `None` | None or integer | Mechanism count / 机制数 | Hao1 默认 5，显式也只能 5；band_fit 默认 3，允许整数 2–6；sls_compat 须显式 1。 |
| `fit_tolerance` | `keyword-only` | `0.05` | None or positive real | Relative inverse-Q tolerance / 相对逆 Q 容差 | 正相对 inverse-Q 容差；超过时默认 warning；None 跳过检查。采样诊断不构成全连续频率误差证明。 |
| `strict` | `keyword-only` | `False` | bool | Fit warning/error policy / 拟合警告或报错 | 超过拟合容差时是否由 warning 改为报错。 |
| `material_model` | `keyword-only` | `None` | None or callable | Material provider / 物性提供函数 | 高级 callable `(vp_padded,q_padded,dt,f_ref)->(c_rel2,strength,A,B)`，接管内置物性选项，详见下文。 |
| `checkpoint_interval` | `keyword-only` | `None` | None or positive integer; time steps | Replay segment length / 重放分段长度 | 仅 checkpoint 下可显式给正整数分段长度；原生 None 用形状/所需历史估计内存平衡；Torch None 固定为 32 个内部步，均不按 GPU 型号调参。full 下提供此参数会报错。 |
| `execution` | `keyword-only` | `'eager'` | str | Torch execution | Torch eager/compile；原生仅 eager。 |
| `compile_steps` | `keyword-only` | `1` | positive integer | Steps per execution block | Torch 正整数（非 bool），尾块完整执行；原生仅 1。 |
<!-- api-doc:visco_gsls:parameters:end -->

(visco-gsls-modes)=
## 模式、Q 与参考速度

| mode | q 的物理含义 | 机制数与频带 |
|---|---|---|
| `hao1`（默认） | Hao & Greenhalgh 的名义 Q0，不是精确 Q(f_ref) | `None` 解析为 5；显式也必须 5；必须显式提供 `(scale,200*scale)` Hz 设计频带 |
| `band_fit` | 近似恒 Q 的频带拟合目标，不是精确 Q(f_ref) | `None` 解析为 3；支持整数 2–6；必须给正有限递增 `(fmin,fmax)` Hz |
| `sls_compat` | f_ref 处精确复体积模量 Q | 必须显式 `n_mechanisms=1`；无需 frequency_band，不使用频带拟合选项 |

所有内置模式的 `vp` 都是 f_ref 处精确相速度，不能直接以论文 v0 替代。正频率复模量采用 `exp(i*omega*t)` 约定；模量 Q 不与波数衰减定义混用。Q=+inf 是精确无耗散极限，相应 Q 灵敏度为零。

Hao1 使用论文 Table 4 五组权重及 Eq. (45) 的统一频率缩放。“一阶”指 inverse-Q 展开阶数，并非一个机制。设 `x_l=2*pi*f_ref*t_l`、`S=sum(k_l*x_l**2/(1+x_l**2))`，被动性要求 Q0>S；不合法值报错而不截断。震源工作频带可为设计频带的子集，例如 [1,50] Hz；不能把 (1,50) 当作 Hao1 设计频带。该实现不包含论文二阶耦合模型。

Hao1 和 band_fit 默认在稠密采样频率上检查相对 inverse-Q 误差，超过 0.05 会 warning；`strict=True` 改为报错，`fit_tolerance=None` 跳过诊断。容差不改变已定义的 Hao1 物性，也不保证任意 Q 的拟合精度。band_fit 活动集/秩变化处不可微，其他区域分段可微。sls_compat 不执行频带误差检查；仍校验 strict 为 bool。

从旧 SLS 迁移须同时指定 mode 与机制数，不能只替换函数名而保留默认 Hao1。其准备函数现按 backend 在前、device 在后的 GSLS 签名调用。

(visco-gsls-physics)=
## 源、坐标、采样与输入域

模型为 `[nz,nx]` 即 `[z,x]`，两维至少 2；三个模型及源的 dtype/device 必须一致。支持 CPU/CUDA、float32/64 strided 张量和非连续视图，连续化保留 autograd 链；不隐式转换模型/源的 dtype 或设备。rho 必须固定、正有限且倒数可表示。只有 Q 可含 +inf。

几何必须完整提供 `[B,S,2]` 和 `[B,R,2]`，按未扩边原域 `[z,x]` 整数网格坐标索引，不接受省略源维度的简写或域外点。坐标会复制到模型设备并转换为 int64。重合源相加，重复接收点重复记录，其反传余切量累加。B、S、R、T 均至少 1。

源为二阶 forcing，单位 Pa/m²；压力更新含 `-dt**2*c_rel2*forcing`。首次源更新乘 1/2，满足零初始压力与压力时间导数。没有隐藏网格体积因子；Pa/s 压力率源不能不经转换直接替代。

记录是更新前 `p[n]`，对应 `t=n*dt`，共 T 点。第一个样点严格为零，最后一个源样点只影响未返回的 `p[T]`，因此该样点对记录的 VJP 为零。T=1 时记录及模型/源梯度均为零。迁移观测与旧波形时须分别核对符号、幅度单位和时间原点。

(visco-gsls-gradients)=
## 梯度、扩边与 PML

- Vp、Q、源及 provider 闭包参数可分别或同时训练；只承诺一阶传播梯度。密度、几何、dt、间距、FD 与 CPML 设置固定。
- 对称 replicate 扩边形成 `[nz+2*pml_z,nx+2*pml_x]` 域。完整转置将扩边敏感度累加回原始边缘与角点；不会以 crop-backward 替代。源处 c_rel2 的模型依赖也保留在梯度链中。
- PML 某轴宽度为 0 则关闭该轴 PML；两轴均为 0 时为有限零外延域，没有吸收层，不是自由表面。CPML 节点/面记忆按离散转置反传，但阻尼、alpha 和速度包络作为固定数值设置不求导。
- 开启梯度、PML 非零且 Vp/Q 或任意物性输出可训练时（含 provider 闭包参数），须给显式固定 max_vel，覆盖 `sqrt(c_rel2*(1+strength.sum(0)))`，不能仅覆盖 Vp。训练迭代及配对有限差分共用同一包络。
- 仅正演、仅源求导或 PML 全零时可用 max_vel=None，自动上界与 autograd 分离；显式上界过小会报错。
- 每次 forward 拥有独立状态/历史。loss、归一化、mask、批次与优化器由调用者管理。不隐式截断参数、缩炮、修改 dt 或替换 NaN；非法系数、非有限输入/接收余切量直接报错。

(visco-gsls-stability)=
## 时间步检查

令 D 为所选交错差分权重绝对值之和，v_bound 覆盖未松弛速度，则保守检查为：

```text
dt <= 0.8 / (v_bound * sqrt(max(rho)/min(rho))
             * D * sqrt(1/dz**2 + 1/dx**2))
```

超限直接报错并给出允许的最大 dt；调用者重新选择 dt 并生成对应采样的源，不隐式重采样或子步进。dt² 和逆间距须能以模型 dtype 表示为有限正 normal 数。此界不是任意强反差、极端 Q、异质 CPML 或长时运行的稳定性证明；还须检查实际实验有限性和边界反射。accuracy=2/4/6/8 是空间阶数，压力时间离散为二阶。

(visco-gsls-memory)=
## full 与 checkpoint 内存

原生 native_cpu/CUDA 与 Torch 均支持 full 和 checkpoint。不支持 boundary。full 下 `checkpoint_interval` 必须为 None；checkpoint 显式间隔为正整数时间步。原生默认 None 依据形状与所需历史估算；Torch 默认为 32 个内部步。均不按 GPU 型号自动调速。

原生 full 按所请求的 primitive 梯度选择历史：c_rel2 保存 force；strength/B 保存空间驱动；A 保存全部 M 个旧物性记忆（例如可训练松弛时间）。仅源求导或 no_grad 观测不保存这些体积历史。固定 Q 的 Hao1 Vp 梯度存 T*V 项，联合 Vp/Q 存 2*T*V，通用 provider 全 primitive 梯度存 (M+2)*T*V；五个物理记忆仍实际演化。

以下公式仅描述原生历史，不能用于 Torch autograd 峰值估算。令 N 为扩边网格点数，V=B*N，Fx=B*nz_padded*(nx_padded+1)，Fz=B*(nz_padded+1)*nx_padded，M 为机制数。完整 checkpoint 重启状态含 p、前一 p、全部物性记忆、节点与面 CPML，共 S=(4+M)*V+Fx+Fz 项。分段长度 K 的 tape/replay 历史约为 `ceil(T/K)*S + K*H*V` 项；H 为请求的 scalar 历史数，旧物性记忆贡献 M。默认 `K=ceil(sqrt(S*T/(H*V)))`（需要历史时），限制于 [1,T]。这些是张量项数，乘 dtype 字节数后也不等于完整峰值显存：还包括输入、输出、图、每炮梯度、优化器及工作区。

原生 checkpoint 恢复完整状态并正向重放每段，再按反序应用离散伴随；不是逆衰减。首次半权重、全局源时钟和尾段均保留，伴随状态跨段连续。需要 checkpoint 重放历史时，每个时间步额外重算一次，用计算换历史内存。没有边界逆重建或压缩。示例和内存公式不构成 GPU 性能测量。

Torch checkpoint 保存完整段起始状态（含 PML/物性记忆），由非重入 checkpoint 正向重算并由 autograd 反传，不使用逆衰减或上述原生伴随历史。短记录/小模型不保证降低峰值显存。

(visco-gsls-example)=
## 最小调用与完整 CPU 示例

以下最小调用假定已准备匹配 CUDA 库、固定模型、源与几何均在同一 CUDA 设备，且 dt 合法。frequency_band 虽默认 None，但默认 Hao1 语义要求显式提供。

<!-- api-doc:visco_gsls:minimal:start -->
```python
records, = starwave.visco_gsls(vp, q, rho, 10.0, 0.0005,
    source_amplitudes=source, source_locations=src, receiver_locations=rec,
    f_ref=18.0, frequency_band=(1.0, 200.0))
```
<!-- api-doc:visco_gsls:minimal:end -->

下面完整接线使用 CPU Torch full，不需原生库，全部可选参数显式提供。1000 为示例 Pa/m² 波形标定；能量 loss 只检查求导接线，不表示 FWI 收敛。Q0=40 的 Hao1 可能触发默认 5% 误差 warning；保留并诊断它，不以放宽容差宣称物性更准确。

```python
import torch
import starwave

# V16 source; explicit CPU Torch reference, no native build required.
dtype = torch.float64
nz, nx, nt = 12, 16, 80
vp = torch.full((nz, nx), 1800.0, dtype=dtype, requires_grad=True)
q = torch.full((nz, nx), 40.0, dtype=dtype, requires_grad=True)
depth = torch.linspace(0.0, 1.0, nz, dtype=dtype)
rho = (1800.0 + 200.0 * depth[:, None]).expand(nz, nx).contiguous()
dt = 0.0005
t = torch.arange(nt, dtype=dtype) * dt
a = torch.pi * 18.0 * (t - 0.02)
source = (1000.0 * (1.0 - 2.0 * a.square()) * torch.exp(-a.square()))
source = source.reshape(1, 1, nt).requires_grad_()
src = torch.tensor([[[3, 8]]], dtype=torch.int64)
rec = torch.tensor([[[3, 4], [3, 8], [3, 12]]], dtype=torch.int64)

records, = starwave.visco_gsls(
    vp, q, rho, (10.0, 10.0), dt,
    source_amplitudes=source, source_locations=src,
    receiver_locations=rec, f_ref=18.0,
    accuracy=4, pml_width=(4, 4), memory="full",
    max_vel=4000.0, backend="torch", mode="hao1",
    frequency_band=(1.0, 200.0), n_mechanisms=5,
    fit_tolerance=0.05, strict=False,
    material_model=None, checkpoint_interval=None,
    execution="eager", compile_steps=1,
)
assert records.shape == (1, 3, nt)
assert torch.isfinite(records).all()
assert torch.count_nonzero(records[..., 0]) == 0
records.square().mean().backward()
for grad in (vp.grad, q.grad, source.grad):
    assert grad is not None and torch.isfinite(grad).all()
assert torch.count_nonzero(source.grad[..., -1]) == 0
```

完整 native checkpoint 调用（模型/源/几何先移至同一 CUDA 设备并预加载，或显式使用 native_cpu 与 CPU 张量）：

<!-- api-doc:visco_gsls:full:start -->
```python
records, = starwave.visco_gsls(vp, q, rho, (10.0, 12.0), 0.0005,
    source_amplitudes=source, source_locations=src, receiver_locations=rec,
    f_ref=18.0, accuracy=4, pml_width=(8, 10), memory="checkpoint",
    max_vel=4000.0, backend="cuda", mode="hao1",
    frequency_band=(1.0, 200.0), n_mechanisms=5, fit_tolerance=0.05,
    strict=False, material_model=None, checkpoint_interval=32,
    execution="eager", compile_steps=1)
```
<!-- api-doc:visco_gsls:full:end -->

示例 max_vel 仅用于此配置；实际模型必须重新核对未松弛速度与 CFL。Torch 例可保持 backend="torch"，改为 memory="checkpoint" 并设置 checkpoint_interval=32；execution="compile" 与 compile_steps=4 另行控制编译。

(visco-gsls-coefficients)=
## 物性与频谱辅助函数

辅助函数导入自 `starwave.visco_gsls_coefficients`，不使用独立 starwave_gsls 命名空间。下列为签名清单（非执行语句）：

```text
from starwave.visco_gsls_coefficients import (
    coefficients, hao1_weighting, hao1_relaxation_times,
    coefficients_from_relaxation, band_relaxation_times,
    quality_spectrum, normalized_modulus, fit_diagnostics,
)

coefficients(vp, q, dt, f_ref, *, mode="hao1", frequency_band=None,
             n_mechanisms=None, fit_tolerance=0.05, strict=False)
hao1_weighting(frequency_band=(1., 200.), *, dtype=torch.float64, device=None)
hao1_relaxation_times(frequency_band=(1., 200.), *, dtype=torch.float64, device=None)
coefficients_from_relaxation(vp, strength, relaxation_times, dt, f_ref)
band_relaxation_times(frequency_band, n_mechanisms=3, *,
                      dtype=torch.float64, device=None)
quality_spectrum(strength, relaxation_times, frequencies)
normalized_modulus(strength, relaxation_times, frequencies)
fit_diagnostics(q, strength, relaxation_times, frequency_band, *,
                sample_count=1001, chunk_size=1024,
                target_convention="band_fit_target")
```


- coefficients 返回 `(c_rel2,strength,A,B)`，首项 `[z,x]`、单位 m²/s²，其余 `[M,z,x]`、无量纲，保留 Vp/Q 完整链。该 helper 可接受标量 Q；传播入口仍要求 Q 与模型同形状。
- hao1_weighting 返回 `[5]` 秒时间与无量纲权重 `(times,k)`；hao1_relaxation_times 只返回时间。helper 的默认设计频带 (1,200) 不代表传播器自动采用该频带。
- coefficients_from_relaxation 接受非负强度 `[M,z,x]` 和正有限时间 `[M]` 或 `[M,z,x]`，保留 Vp、强度、时间梯度，并校准参考相速度。band_relaxation_times 仅适用于 band_fit，不能与 Hao1 权重混配。
- quality_spectrum 返回 `[frequency,*model_shape]` 实际模量 Q，零衰减为 +inf；normalized_modulus 返回同形状复数 F=M/M_R。频率是 Hz，时间是秒。
- fit_diagnostics 返回 detached 字典，包括 max_relative_inverse_q_error、max_relative_q_error、minimum_strength、all_nonnegative、worst_frequency_hz、worst_model_flat_index、sample_count、frequency_band、target_convention。sample_count≥2，chunk_size≥1；包含频带两端的对数网格逐模型单元检查，不是连续全频率误差证明。Hao1 应指定 hao1_nominal_Q0，可只诊断设计频带内的工作子带。

下例接在完整 CPU 示例之后，检查 [1,50] Hz 工作子带：

```python
from starwave.visco_gsls_coefficients import (
    coefficients, hao1_relaxation_times, quality_spectrum, fit_diagnostics,
)
c_rel2, strength, A, B = coefficients(
    vp, q, dt, 18.0, mode="hao1", frequency_band=(1.0, 200.0),
)
times = hao1_relaxation_times((1.0, 200.0), dtype=vp.dtype, device=vp.device)
frequencies = torch.logspace(0, torch.log10(torch.tensor(50.0)).item(),
                            101, dtype=vp.dtype, device=vp.device)
actual_q = quality_spectrum(strength, times, frequencies)
report = fit_diagnostics(q, strength, times, (1.0, 50.0),
                        target_convention="hao1_nominal_Q0")
print(actual_q.shape, report["max_relative_inverse_q_error"])
```

(visco-gsls-provider)=
## 高级 material_model 协议

`provider(vp_padded,q_padded,dt,f_ref)` 在每次 forward 完成 replicate 扩边后调用一次。它接管内置 mode、frequency_band、n_mechanisms、fit_tolerance、strict；这些参数不自动传给 provider 或诊断它。原始输入合法性、系数和 CFL 检查仍生效。频带、额外参数由闭包持有，provider 自行负责 Q 意义、相速度归一化与实际频谱准确性。

返回四个同 dtype/device 的 strided Tensor：c_rel2 `[zp,xp]`，strength/A/B `[M,zp,xp]`，M≥1。全部有限，c_rel2>0，strength≥0，B≥0，−1<A≤1，且 B≈strength*(1−A)，一致性容差为 32 倍 dtype eps。它是被动梯形 GSLS 递推的物性接口，不能通过任意系数替换波动方程。

原生反向返回全部四个物性输出的 VJP，再经 Torch 链式传到 provider 参数。避免 detach、NumPy、对可训练值使用 item 或 torch.tensor 重新包装。闭包中的原域空间张量不会自动扩边，必须自行按完全相同域扩展并保留梯度。非零 PML 下所有可训练物性输出仍要求固定显式 max_vel。

以下接在 CPU 示例后定义每机制可训练权重和时间；修改后属于实验物性，不再是固定发表的 Hao1 谱。优化器必须保持 Q0>S，并重新诊断实际谱。外部 provider 变更本身无需重编译；修改包内指纹跟踪文件须重建并重启。

```python
from starwave.visco_gsls_coefficients import (
    coefficients_from_relaxation, hao1_weighting,
)

base_times, base_k = hao1_weighting(
    (1.0, 200.0), dtype=vp.dtype, device=vp.device,
)
log_weight = torch.nn.Parameter(torch.zeros_like(base_k))
log_time = torch.nn.Parameter(torch.zeros_like(base_times))

def provider(vp_padded, q_padded, dt, f_ref):
    times = base_times * log_time.exp()
    k = base_k * log_weight.exp()
    x = 2 * torch.pi * f_ref * times
    S = (k * x.square() / (1 + x.square())).sum()
    if not bool(torch.all(q_padded > S)):
        raise ValueError("Learned material requires Q0 > S")
    invq = q_padded.reciprocal()
    strength = k[:, None, None] * invq[None] / (1 - S * invq)[None]
    return coefficients_from_relaxation(vp_padded, strength, times, dt, f_ref)

# Pass material_model=provider and a covering fixed max_vel to visco_gsls.
# Add log_weight and log_time to the caller's optimizer when learning them.
```

(visco-gsls-runtime)=
## 原生准备、构建与重启

```{py:function} starwave.prepare_visco_gsls(backend: str='native_cpu', device: str | torch.device | None=None) -> dict

显式加载已有匹配库，核对 source/library/metadata 与 ABI；不编译或传播。

:param backend: **默认 `'native_cpu'`；位置或关键字。** 仅 'native_cpu' 或 'cuda'；不接受 'torch'。
:type backend: `str`
:param device: **默认 `None`；位置或关键字。** CUDA 必须为有效显式逻辑设备，如 'cuda:0' 或 torch.device('cuda:0')；不能只写 'cuda'。native_cpu 忽略此项。
:type device: `str | torch.device | None`
:returns: 状态字典；成功仅表示加载与兼容核验通过。
:rtype: `dict`
```

```{py:function} starwave.visco_gsls_native_status(backend: str='native_cpu') -> dict

只读查询库与 metadata 是否存在及是否载入；不载库、不编译。

:param backend: **默认 `'native_cpu'`；位置或关键字。** 仅 'native_cpu' 或 'cuda'；不接受 'torch'。
:type backend: `str`
:returns: 状态字典，含 backend、library_path、library_exists、metadata_exists、library_loaded、cuda_available、gpu_numerically_verified、supported_memory、gradient_order。gpu_numerically_verified 恒为 False，不代表 GPU 验收。
:rtype: `dict`
```

注意：prepare/status 默认 native_cpu，传播默认 cuda；prepare 两项都不是 keyword-only。建议显式写关键字防止与旧 SLS 的参数顺序混淆。

```python
import starwave

print(starwave.visco_gsls_native_status(backend="native_cpu"))
starwave.prepare_visco_gsls(backend="native_cpu", device=None)
# After a matching CUDA build, with an available logical CUDA device:
starwave.prepare_visco_gsls(backend="cuda", device="cuda:0")
```

在已获准取得的完整 V16 源码根目录显式构建，使用已安装且与 Torch 匹配的工具链：

```bash
# All three independent CUDA libraries; replace 80 with the actual target SM.
python compile_all.py --arch 80
# GSLS CUDA only:
python compile_visco_gsls.py --arch 80
# Separate native CPU library:
python compile_visco_gsls.py --backend cpu
```


compile_all 构建/核验 core、elastic、GSLS 三库；combined_build 仅 core/elastic。源树 GSLS 默认在 native/build/gsls，安装模式使用 StarWave cache。可选 STARWAVE_BUILD_DIR 下的 gsls 子目录必须与运行时一致；不用独立 STARWAVE_GSLS_BUILD_DIR。完整源树运行不要求 pip 注册或 export，也不下载编译器；可选 editable 安装与编译是两步。

GSLS ABI=2、semantics=1、capabilities=255；核心 ABI=2 是另一套独立库。完整源/库/metadata 指纹必须匹配，不借用其他树的库，不绕过保护。更新已载入 Python/native 后须显式重建并重启进程/Notebook kernel。既有 prepare_native 或 prepare_elastic 不替代 GSLS 准备。没有运行时自动编译或 backend fallback。

CUDA 数值、实际多卡、长 FWI 和性能均须在目标设备单独验证；库存在、编译或预加载成功不能替代这些证据。本页示例不提供新的 GPU 或性能验收结果。

(visco-gsls-references)=
## 范围与参考

不支持 3D、rho 梯度、PML/dt 训练、boundary 重建、初始状态输入、终态返回、自由表面、压缩、照明、AMP、CUDA graphs 或受支持的高阶传播梯度。Torch 图的偶然高阶行为不构成契约。

- Hao & Greenhalgh (2021), “Nearly constant Q models of the generalized standard linear solid type and the corresponding wave equations”, GEOPHYSICS 86(4), T239–T260. [DOI](https://doi.org/10.1190/geo2020-0548.1).

论文描述物理模型；本 API 的负源、参考相速度归一化、离散采样、PML、扩边与伴随遵循本页契约，不声称与其他实现逐位相同。常规目标函数组织见[反演](inversion/fwi.md)。历史 SLS 实验不是当前 GSLS 的运行证据。
