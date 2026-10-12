# 波传播

本章当前契约对应冻结 V19。完整参数与后端限制见 [Usage](../usage.md)，发布渠道见[安装](../installation.md)。历史推导与实测例子保留其注明版本。

选择下方的 [VRZ](#wave-vrz) 或[声学 VTI](#wave-vti)，查看模型约定、调用示例与 V19 平顶限制。常密度线性化散射见 [Born Function](../usage.md#born-function)；主包矢量反射率 API见 [VRR Function](../usage.md#vrr-function)。V19 源码与历史 wheel 属于不同发布渠道。

(wave-vrz)=
## VRZ：速度与阻抗或密度

```{include} vrz.md
:start-line: 1
:heading-offset: 1
```

(wave-vti)=
## 声学 VTI：二维与三维

```{include} vti.md
:start-line: 1
:heading-offset: 1
```
