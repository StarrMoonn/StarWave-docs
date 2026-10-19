(wavefield-snapshots)=
# 2D wavefield snapshots: six wave equations

Use one three-layer model, one source centroid and three observation times to inspect Acoustic, Elastic, VTI, VRZ, VRR and GSLS propagation. Each section gives the governing equations, the displayed field, primary references and actual numerical snapshots.

```{important}
These figures are **CPU results computed on 2026-10-18**, using source baseline `391f4a45efdee2f88a9d1432997fdca20ecd2d6f`. No PDE was rerun for this publication, and the historical figures are not relabelled as current-dev or CUDA measurements. The companion CUDA notebook/script contain no embedded GPU output and have not passed CUDA numerical acceptance; they are separate reproduction entry points.
```

## Common model and sampling

The physical domain is 2.0 km × 1.6 km, with 401 × 321 nodes and Δx = Δz = 5 m. Interfaces are centred at depths 650 / 1100 m with tanh transitions and a 15 m half-width parameter. All boundaries use 30-cell PML, with no free surface. PML cells are excluded from the figures.

| Parameter | Top layer | Middle layer | Bottom layer |
|---|---:|---:|---:|
| Vp (m/s) | 1800 | 2300 | 2800 |
| Vs (m/s, elastic only) | 1000 | 1300 | 1600 |
| ρ (kg/m³) | 1900 | 2150 | 2400 |
| VTI ε | 0.18 | 0.22 | 0.26 |
| VTI δ | 0.06 | 0.08 | 0.10 |
| GSLS nominal Q₀ | 30 | 30 | 30 |

The source centroid is (x,z) = (1000,450) m. Its 12 Hz Ricker wavelet peaks at 0.10 s. Fourth-order spatial differences, user Δt = 0.0005 s and 1002 source samples are used. Snapshot times **0.20 / 0.35 / 0.50 s** are measured from simulation start, not the wavelet peak. The extra final sample permits elastic half-time collocation. The constant-density Scalar equation does not use the density model.


```{raw} html
<figure class="sw-example-figure">
  <div class="sw-example-viewport" tabindex="0" role="region" aria-label="Scroll to inspect the full figure">
    <img src="../_static/wavefield2d/model_context.png" alt="Three-layer Vp, Vs and density models with source centroid at x=1000 m, z=450 m" loading="lazy" decoding="async">
  </div>
  <figcaption>Common physical models, layer interfaces and source geometry. Vs is used only by Elastic. <a href="../_static/wavefield2d/model_context.png">PNG</a> · <a href="../_static/wavefield2d/model_context.pdf">PDF</a> · <a href="../_static/wavefield2d/model_context.svg">SVG</a></figcaption>
</figure>
```


## How to read the figures

Each main linear row uses its own A = max|field| over all three times. The same A is retained within that row; plots show field/A without per-frame normalization or percentile clipping. Source units and displayed fields differ between equations, so **brightness across rows is not a comparison of absolute physical amplitudes**. Pa labels follow the chosen forcing units, rather than a calibrated physical source.

Non-elastic snapshots are dense receiver samples on physical grid nodes. Elastic vertical velocity remains on staggered faces z = (j+½)Δz and adjacent half-time records are averaged to the common integer times. Face locations are not relabelled as nodes. Coordinates x and z denote horizontal distance and depth, with ∇=(∂x,∂z). The equations below describe the continuum physical-domain interior, omitting PML auxiliary variables; finite differences, staggered storage and injection conventions follow each implementation.


```{raw} html
<figure class="sw-example-figure">
  <div class="sw-example-viewport" tabindex="0" role="region" aria-label="Scroll to inspect the full figure">
    <img src="../_static/wavefield2d/six_equation_snapshots.png" alt="Six two-dimensional wave-equation families at three times, with a fixed linear scale within each row" loading="lazy" decoding="async">
  </div>
  <figcaption>Overview: compare evolution within rows, and wavefront geometry and field definitions across rows. <a href="../_static/wavefield2d/six_equation_snapshots.png">PNG</a> · <a href="../_static/wavefield2d/six_equation_snapshots.pdf">PDF</a> · <a href="../_static/wavefield2d/six_equation_snapshots.svg">SVG</a></figcaption>
</figure>
```


## 1. Acoustic: constant-density scalar acoustics

```{math}
\partial_t^2p=v_p^2(\partial_x^2p+\partial_z^2p-f).
```

Here p is pressure (Pa), v_p is model speed (m/s), and f is the grid-point second-order forcing (Pa/m²). Speed may vary spatially while density is constant. Source increments are −v_source²Δt_internal² f, rather than direct pressure additions. The equation describes compression and reflection/refraction from speed changes, without elastic S waves. Implementation: `starwave/scalar.py` and its scalar propagation kernel.



```{raw} html
<figure class="sw-example-figure sw-wavefield-family" data-family="acoustic">
  <div class="sw-example-viewport" tabindex="0" role="region" aria-label="Acoustic: wavefields at 0.20, 0.35 and 0.50 s; scroll horizontally">
    <div class="sw-wavefield-row">
    <a href="../_static/wavefield2d/panels/acoustic_t0200ms.png"><img src="../_static/wavefield2d/panels/acoustic_t0200ms.png" alt="Acoustic p at t=0.20 s: original CPU field in the three-layer model, one linear scale for all three times" loading="lazy" decoding="async"></a>
    <a href="../_static/wavefield2d/panels/acoustic_t0350ms.png"><img src="../_static/wavefield2d/panels/acoustic_t0350ms.png" alt="Acoustic p at t=0.35 s: original CPU field in the three-layer model, one linear scale for all three times" loading="lazy" decoding="async"></a>
    <a href="../_static/wavefield2d/panels/acoustic_t0500ms.png"><img src="../_static/wavefield2d/panels/acoustic_t0500ms.png" alt="Acoustic p at t=0.50 s: original CPU field in the three-layer model, one linear scale for all three times" loading="lazy" decoding="async"></a>
    </div>
  </div>
  <figcaption>Pressure propagation, interface reflection and transmission. Density is constant in this equation. <a href="../_static/wavefield2d/data/acoustic_snapshots.npz">Raw NPZ</a> · <a href="../_static/wavefield2d/data/acoustic_metadata.json">Run metadata</a></figcaption>
</figure>
```


## 2. Elastic: isotropic elasticity

```{math}
\begin{aligned}
\rho\partial_t v_x&=\partial_x\sigma_{xx}+\partial_z\sigma_{xz}+b_x,\\
\rho\partial_t v_z&=\partial_x\sigma_{xz}+\partial_z\sigma_{zz}+b_z,\\
\partial_t\sigma_{xx}&=(\lambda+2\mu)\partial_xv_x+\lambda\partial_zv_z,\\
\partial_t\sigma_{zz}&=\lambda\partial_xv_x+(\lambda+2\mu)\partial_zv_z,\\
\partial_t\sigma_{xz}&=\mu(\partial_zv_x+\partial_xv_z),\\
\mu&=\rho v_s^2,\qquad\lambda=\rho(v_p^2-2v_s^2).
\end{aligned}
```

Stresses and Lamé parameters λ, μ are in Pa; v_x, v_z are particle velocities (m/s), and b is body-force density (N/m³). The vertical force excites P and SV motion, with possible interface mode conversion. We plot v_z (named `v_y` by the 2D API); its lobes and polarity depend on the force direction and observed component. [Virieux (1986)](https://doi.org/10.1190/1.1442147) gives the classical staggered velocity–stress method; see the [Deepwave elastic derivation](https://ausargeo.com/deepwave/elastic).

The source is split equally between z=447.5 / 452.5 m faces and sampled at (k−½)Δt. Averaging receiver records k and k+1 collocates at kΔt. This is time interpolation, retaining all 320 physical depth faces.



```{raw} html
<figure class="sw-example-figure sw-wavefield-family" data-family="elastic">
  <div class="sw-example-viewport" tabindex="0" role="region" aria-label="Elastic: wavefields at 0.20, 0.35 and 0.50 s; scroll horizontally">
    <div class="sw-wavefield-row">
    <a href="../_static/wavefield2d/panels/elastic_t0200ms.png"><img src="../_static/wavefield2d/panels/elastic_t0200ms.png" alt="Elastic v_z at t=0.20 s: original CPU field in the three-layer model, one linear scale for all three times" loading="lazy" decoding="async"></a>
    <a href="../_static/wavefield2d/panels/elastic_t0350ms.png"><img src="../_static/wavefield2d/panels/elastic_t0350ms.png" alt="Elastic v_z at t=0.35 s: original CPU field in the three-layer model, one linear scale for all three times" loading="lazy" decoding="async"></a>
    <a href="../_static/wavefield2d/panels/elastic_t0500ms.png"><img src="../_static/wavefield2d/panels/elastic_t0500ms.png" alt="Elastic v_z at t=0.50 s: original CPU field in the three-layer model, one linear scale for all three times" loading="lazy" decoding="async"></a>
    </div>
  </div>
  <figcaption>Vertical particle velocity from a vertical force source. P/S labels interpret the wavefronts; no modal decomposition was performed. <a href="../_static/wavefield2d/data/elastic_snapshots.npz">Raw NPZ</a> · <a href="../_static/wavefield2d/data/elastic_metadata.json">Run metadata</a></figcaption>
</figure>
```


## 3. VTI: vertical-axis pseudo-acoustic system

```{math}
\begin{aligned}
\rho\partial_t v_x&=\partial_xs_H,\qquad\rho\partial_t v_z=\partial_zs_V,\\
\partial_t s_H&=c_{11}\partial_xv_x+c_{13}\partial_zv_z+q_H,\\
\partial_t s_V&=c_{13}\partial_xv_x+c_{33}\partial_zv_z+q_V,\\
c_{11}&=\rho v_p^2(1+2\epsilon),\quad c_{33}=\rho v_p^2,\quad
c_{13}=\rho v_p^2\sqrt{1+2\delta}.
\end{aligned}
```

v_p is vertical P speed; ε, δ are dimensionless. The equal stress rates q_H=q_V (Pa/s) are injected with +Δt into both s_H and s_V. We record s_V (Pa). This is the zero-shear pseudo-acoustic family of [Duveneck et al. (2008)](https://doi.org/10.1190/1.3059320), rather than full elastic VTI. Its coupled system may retain a qSV-like secondary branch despite having no physical v_s input. The exact coefficients and signs above were checked against this implementation.



```{raw} html
<figure class="sw-example-figure sw-wavefield-family" data-family="vti">
  <div class="sw-example-viewport" tabindex="0" role="region" aria-label="VTI: wavefields at 0.20, 0.35 and 0.50 s; scroll horizontally">
    <div class="sw-wavefield-row">
    <a href="../_static/wavefield2d/panels/vti_t0200ms.png"><img src="../_static/wavefield2d/panels/vti_t0200ms.png" alt="VTI sV at t=0.20 s: original CPU field in the three-layer model, one linear scale for all three times" loading="lazy" decoding="async"></a>
    <a href="../_static/wavefield2d/panels/vti_t0350ms.png"><img src="../_static/wavefield2d/panels/vti_t0350ms.png" alt="VTI sV at t=0.35 s: original CPU field in the three-layer model, one linear scale for all three times" loading="lazy" decoding="async"></a>
    <a href="../_static/wavefield2d/panels/vti_t0500ms.png"><img src="../_static/wavefield2d/panels/vti_t0500ms.png" alt="VTI sV at t=0.50 s: original CPU field in the three-layer model, one linear scale for all three times" loading="lazy" decoding="async"></a>
    </div>
  </div>
  <figcaption>Vertical pseudo-acoustic stress, with both sH/sV excited. The slow inner branch is retained and is not a physical elastic S wave. <a href="../_static/wavefield2d/data/vti_snapshots.npz">Raw NPZ</a> · <a href="../_static/wavefield2d/data/vti_metadata.json">Run metadata</a></figcaption>
</figure>
```


The isotropic control below sets only ε and δ to zero, retaining velocity, density, source, grid, time step and PML. Both rows share A and use the explicitly labelled signed square-root colour mapping to reveal weak branches. This supports an association of the slow inner branch with the anisotropic pseudo-acoustic system. It is not a modal projection of the original heterogeneous field and does not identify a physical elastic S wave.


```{raw} html
<figure class="sw-example-figure">
  <div class="sw-example-viewport" tabindex="0" role="region" aria-label="Scroll to inspect the full figure">
    <img src="../_static/wavefield2d/vti_isotropic_control.png" alt="Original VTI and epsilon=delta=0 isotropic control, sharing one signed square-root colour scale across three times" loading="lazy" decoding="async">
  </div>
  <figcaption>VTI isotropic-limit control; the slow-branch interpretation is not a complete modal decomposition. <a href="../_static/wavefield2d/vti_isotropic_control.png">PNG</a> · <a href="../_static/wavefield2d/vti_isotropic_control.pdf">PDF</a> · <a href="../_static/wavefield2d/vti_isotropic_control.svg">SVG</a></figcaption>
</figure>
```


## 4. VRZ: variable-density acoustics

```{math}
\begin{aligned}
\partial_t^2p&=v_pZ\nabla\!\cdot\!\left(\frac{v_p}{Z}\nabla p\right)-v_p^2f\\
&=v_p^2\Delta p+\left[v_p\nabla v_p+v_p^2Z\nabla(Z^{-1})\right]\!\cdot\!\nabla p-v_p^2f,\qquad Z=\rho v_p.
\end{aligned}
```

Z=ρv_p is impedance and ρ>0. For smooth models this equals the variable-density form ∂²_t p=ρv_p²∇·(ρ⁻¹∇p)−v_p²f. The actual kernel uses expanded-product finite differences, not a face-flux discretization; discrete chain rules are not exact. See the velocity–impedance transformation in [Whitmore et al. (2020), equations 1–3](https://doi.org/10.3997/2214-4609.202010332).



```{raw} html
<figure class="sw-example-figure sw-wavefield-family" data-family="vrz">
  <div class="sw-example-viewport" tabindex="0" role="region" aria-label="VRZ: wavefields at 0.20, 0.35 and 0.50 s; scroll horizontally">
    <div class="sw-wavefield-row">
    <a href="../_static/wavefield2d/panels/vrz_t0200ms.png"><img src="../_static/wavefield2d/panels/vrz_t0200ms.png" alt="VRZ p at t=0.20 s: original CPU field in the three-layer model, one linear scale for all three times" loading="lazy" decoding="async"></a>
    <a href="../_static/wavefield2d/panels/vrz_t0350ms.png"><img src="../_static/wavefield2d/panels/vrz_t0350ms.png" alt="VRZ p at t=0.35 s: original CPU field in the three-layer model, one linear scale for all three times" loading="lazy" decoding="async"></a>
    <a href="../_static/wavefield2d/panels/vrz_t0500ms.png"><img src="../_static/wavefield2d/panels/vrz_t0500ms.png" alt="VRZ p at t=0.50 s: original CPU field in the three-layer model, one linear scale for all three times" loading="lazy" decoding="async"></a>
    </div>
  </div>
  <figcaption>Variable-density pressure field; Vp and density both affect reflection and transmission. <a href="../_static/wavefield2d/data/vrz_snapshots.npz">Raw NPZ</a> · <a href="../_static/wavefield2d/data/vrz_metadata.json">Run metadata</a></figcaption>
</figure>
```


## 5. VRR: velocity and vector reflectivity

```{math}
\partial_t^2p=v_p^2\Delta p+\left(v_p\nabla v_p-2v_p^2\mathbf r\right)\!\cdot\!\nabla p-v_p^2f,\qquad\mathbf r=(r_x,r_z).
```

For smooth positive impedance, the continuum correspondence is r=½∇ln Z=−½Z∇(1/Z), in m⁻¹, rather than a dimensionless interface reflection coefficient. This example actually computes a fourth-order centred derivative with a replicated two-cell halo, then crops that halo:

```{math}
r_i=-\tfrac12ZD_i(Z^{-1}),\qquad
D_i g_j=\frac{\tfrac23(g_{j+1}-g_{j-1})-\tfrac1{12}(g_{j+2}-g_{j-2})}{h}.
```

Do not substitute ½D_i(log Z). The API accepts independent, stationary r_x/r_z and propagates scalar p; it does not reconstruct or evolve an impedance potential. Arbitrary coefficient fields need not derive from positive impedance. The equations correspond to [Whitmore et al. (2020), equations 4–5](https://doi.org/10.3997/2214-4609.202010332).



```{raw} html
<figure class="sw-example-figure sw-wavefield-family" data-family="vrr">
  <div class="sw-example-viewport" tabindex="0" role="region" aria-label="VRR: wavefields at 0.20, 0.35 and 0.50 s; scroll horizontally">
    <div class="sw-wavefield-row">
    <a href="../_static/wavefield2d/panels/vrr_t0200ms.png"><img src="../_static/wavefield2d/panels/vrr_t0200ms.png" alt="VRR p at t=0.20 s: original CPU field in the three-layer model, one linear scale for all three times" loading="lazy" decoding="async"></a>
    <a href="../_static/wavefield2d/panels/vrr_t0350ms.png"><img src="../_static/wavefield2d/panels/vrr_t0350ms.png" alt="VRR p at t=0.35 s: original CPU field in the three-layer model, one linear scale for all three times" loading="lazy" decoding="async"></a>
    <a href="../_static/wavefield2d/panels/vrr_t0500ms.png"><img src="../_static/wavefield2d/panels/vrr_t0500ms.png" alt="VRR p at t=0.50 s: original CPU field in the three-layer model, one linear scale for all three times" loading="lazy" decoding="async"></a>
    </div>
  </div>
  <figcaption>Scalar pressure propagated with velocity and vector reflectivity, discretely constructed from the common impedance. <a href="../_static/wavefield2d/data/vrr_snapshots.npz">Raw NPZ</a> · <a href="../_static/wavefield2d/data/vrr_metadata.json">Run metadata</a></figcaption>
</figure>
```


Here the coefficients are constructed from the same impedance Z = ρVp, so the VRZ / VRR fields are close. Measured relative L2 differences are approximately 1.57×10⁻⁵ / 3.80×10⁻⁵ / 7.49×10⁻⁵ at the three times. A continuum transformation does not imply identical arbitrary discrete, PML or boundary implementations, nor does it make VRR an alias of VRZ.

## 6. GSLS: multi-mechanism viscoacoustics

```{math}
\begin{aligned}
\mathcal D_\rho p&=\rho\nabla\!\cdot(\rho^{-1}\nabla p),\\
\partial_t^2p&=c_{\mathrm{rel}}^2\left[\left(1+\sum_{l=1}^{L}a_l\right)\mathcal D_\rho p-\sum_{l=1}^{L}m_l-f\right],\\
\partial_t m_l&=(a_l\mathcal D_\rho p-m_l)/\tau_l.
\end{aligned}
```

The material memories are denoted m_l here to distinguish them from VRR vector r (the implementation calls them r_l). Strengths a_l≥0 are dimensionless; τ_l>0 is in seconds; m_l is in Pa/m² and c_rel is relaxed speed (m/s). Memories enter pressure acceleration with a minus sign. Source scaling is −c_rel²Δt² f; for zero initial states the first forcing sample is f₀/2.

This example uses **five mechanisms, hao1**, the first-order inverse-Q parameterization of [Hao and Greenhalgh (2021)](https://doi.org/10.1190/geo2020-0548.1). “First order” means approximation order in inverse Q, not one mechanism or first-order time integration. The design band is 1–200 Hz, Q₀=30 is nominal, and model Vp is exactly the continuous-model phase speed at the 12 Hz reference frequency. Actual Q_eff(12 Hz)≈30.267; the sampled maximum relative inverse-Q error over the design band is about 9.03%. The original default 5% fit warning remains; this is not exact constant Q=30.

For the exp(+iωt) convention, the frequency response and speed normalization are:

```{math}
F(\omega)=1+\sum_l a_l\frac{i\omega\tau_l}{1+i\omega\tau_l},\qquad
Q_{\mathrm{eff}}=\frac{\Re F}{\Im F},\qquad
c_{\mathrm{rel}}^2=v_p^2\frac{|F_r|+\Re F_r}{2|F_r|^2},\quad F_r=F(2\pi f_{\mathrm{ref}}).
```

At Q=∞, a_l=0 and c_rel=v_p, giving the lossless limit of the same solver.



```{raw} html
<figure class="sw-example-figure sw-wavefield-family" data-family="gsls">
  <div class="sw-example-viewport" tabindex="0" role="region" aria-label="GSLS: wavefields at 0.20, 0.35 and 0.50 s; scroll horizontally">
    <div class="sw-wavefield-row">
    <a href="../_static/wavefield2d/panels/gsls_t0200ms.png"><img src="../_static/wavefield2d/panels/gsls_t0200ms.png" alt="GSLS p at t=0.20 s: original CPU field in the three-layer model, one linear scale for all three times" loading="lazy" decoding="async"></a>
    <a href="../_static/wavefield2d/panels/gsls_t0350ms.png"><img src="../_static/wavefield2d/panels/gsls_t0350ms.png" alt="GSLS p at t=0.35 s: original CPU field in the three-layer model, one linear scale for all three times" loading="lazy" decoding="async"></a>
    <a href="../_static/wavefield2d/panels/gsls_t0500ms.png"><img src="../_static/wavefield2d/panels/gsls_t0500ms.png" alt="GSLS p at t=0.50 s: original CPU field in the three-layer model, one linear scale for all three times" loading="lazy" decoding="async"></a>
    </div>
  </div>
  <figcaption>Viscoacoustic pressure for nominal Q0=30; separately normalized rows alone cannot establish attenuation. <a href="../_static/wavefield2d/data/gsls_snapshots.npz">Raw NPZ</a> · <a href="../_static/wavefield2d/data/gsls_metadata.json">Run metadata</a></figcaption>
</figure>
```


Assess attenuation using a matched control. The same GSLS solver is also run with Q=∞, retaining velocity, density, source and geometry. Both runs and all three times share one linear A below. Full-physical-domain pressure L2 norm ratios (attenuating / lossless) are 0.829 / 0.699 / 0.597. These are field-norm ratios, not energy ratios or stand-alone estimates of Q.


```{raw} html
<figure class="sw-example-figure">
  <div class="sw-example-viewport" tabindex="0" role="region" aria-label="Scroll to inspect the full figure">
    <img src="../_static/wavefield2d/gsls_lossless_comparison.png" alt="GSLS nominal Q0=30 versus the matched lossless control, with one linear scale for both runs and all times" loading="lazy" decoding="async">
  </div>
  <figcaption>Matched, actually computed attenuating / lossless GSLS control. <a href="../_static/wavefield2d/gsls_lossless_comparison.png">PNG</a> · <a href="../_static/wavefield2d/gsls_lossless_comparison.pdf">PDF</a> · <a href="../_static/wavefield2d/gsls_lossless_comparison.svg">SVG</a></figcaption>
</figure>
```


## Downloads, reproduction and validation limits

- [CUDA notebook](https://github.com/StarrMoonn/StarWave/blob/7c1743fe7791c7a52de6900a1ae67c9afe4d9548/notebook/WavefieldSnapshots/WavefieldSnapshots2D_All_CUDA.ipynb) · [synchronized Python script](https://github.com/StarrMoonn/StarWave/blob/7c1743fe7791c7a52de6900a1ae67c9afe4d9548/notebook/WavefieldSnapshots/WavefieldSnapshots2D_All_CUDA.py): install and explicitly build StarWave in the selected kernel, then run in order. The script does not install, build or silently fall back to CPU. StarWave source-repository links require the appropriate repository access.
- Download the actual CPU NPZ and original run metadata below each family. Arrays are stored as [time,z,x]: [3,320,401] for Elastic and [3,321,401] for the other main families. These saved arrays are independent of the CUDA notebook's execution status.
- [Model parameters](../_static/wavefield2d/data/config.json) · [numerical-setting checks](../_static/wavefield2d/data/validation_summary.json) · [GSLS matched metrics](../_static/wavefield2d/data/gsls_comparison_metrics.json) · [lossless arrays](../_static/wavefield2d/data/gsls_lossless_snapshots.npz) · [VTI control arrays](../_static/wavefield2d/data/vti_isotropic_control.npz) · [asset hashes and provenance](../_static/wavefield2d/manifest.json)
- CPU runs are single-shot float32. Acoustic / VTI / VRZ / VRR / GSLS use native_cpu; Elastic uses public Torch CPU eager. Existing source checks and small-grid CPU compatibility checks of the CUDA example do not establish GPU acceptance, full-display-grid convergence or cross-backend agreement.
- Original GSLS metadata is byte-preserved. Its generic source-scaling text `−v²Δt²` is imprecise for GSLS: read it as `−c_rel²Δt²`, with the existing half-source first step. This erratum is also recorded in the provenance manifest.
- Checks covered finite nonzero fields, axes, source positions, time sampling and stable-step settings. They are not an analytic-solution comparison, grid-convergence study or complete adjoint test. At 30 Hz the minimum S velocity gives about 6.67 cells per wavelength; the Ricker wavelet is not strictly bandlimited. VTI and Elastic use internal step ratio 1 in this configuration.
- PDF / SVG labels and axes retain vector elements; numerical fields are embedded raster images. Displayed figures, arrays and historical metadata are byte-identical to the delivered CPU results.

## References and implementation mapping


1. Virieux, J. (1986). P-SV wave propagation in heterogeneous media: velocity-stress finite-difference method. *Geophysics*, 51(4), 889–901. [DOI: 10.1190/1.1442147](https://doi.org/10.1190/1.1442147).
2. Duveneck, E., Milcik, P., Bakker, P. M., & Perkins, C. (2008). Acoustic VTI wave equations and their application for anisotropic reverse-time migration. *SEG Expanded Abstracts*, 2186–2190. [DOI: 10.1190/1.3059320](https://doi.org/10.1190/1.3059320).
3. Whitmore, N. D., Ramos-Martinez, J., Yang, Y., & Valenciano, A. A. (2020). Full Wavefield Modeling with Vector Reflectivity. *EAGE 2020*, 1–5. [DOI: 10.3997/2214-4609.202010332](https://doi.org/10.3997/2214-4609.202010332) · [Primary text](https://www.tgs.com/hubfs/Technical%20Library/Technical%20Library%20Files/eage2020_pgs_whitmore_etal_fwi-modeling_we_dome1_07.pdf).
4. Hao, Q., & Greenhalgh, S. (2021). Nearly constant Q models of the generalized standard linear solid type and the corresponding wave equations. *Geophysics*, 86(4), T239–T260. [DOI: 10.1190/geo2020-0548.1](https://doi.org/10.1190/geo2020-0548.1) · [Author-hosted paper](https://xqihao.github.io/publication/hao-greenhalgh-2021-a/hao-greenhalgh-2021-a.pdf).

Continuum equations, field definitions and source signs were checked against StarWave. The papers establish the physical parameterizations and numerical-method background; they are not numerical acceptance of this code. The pinned [CPU-result source](https://github.com/StarrMoonn/StarWave/tree/391f4a45efdee2f88a9d1432997fdca20ecd2d6f) and [reviewed dev source](https://github.com/StarrMoonn/StarWave/tree/61152f2569815c60a999fa6babfd0abc461fccae) are recorded separately. Relevant files under `starwave/` are `scalar.py`, `elastic.py`, `torch_backend/_vti_portable.py`, `torch_backend/_scalar_portable.py`, `_vrr/_reference.py` and `_visco_gsls_reference.py`.

