(free-surface)=
# Free surface

This chapter describes the flat upper boundary of each 2-D operator and compares CPU seismograms with an absorbing top and with the operator's free-surface setting. “Free surface” is an entry-point label, not a claim that all operators share one discrete closure or record pressure.

```{important}
This is a development-source supplement pinned to StarWave commit `200621cc31c0ebeebaa01ffaa3591c0ebd7e0adb`. It does not rewrite the V19/PyPI contracts elsewhere in this manual. All examples are 2-D, flat-top CPU runs; no CUDA or 3-D example was executed. Source inspection, paired CPU execution, and physical-accuracy acceptance are separate evidence.
```

(free-surface-meaning)=
## Boundary conditions and discrete closures

Acoustic pressure release normally constrains surface pressure to zero; an elastic free surface constrains normal and tangential traction to zero. A finite-difference description must also locate the wall, define ghost extension and staggering, and identify any near-wall reduction of stencil order. “Zero top PML width” alone does not specify these choices.

- A fixed-zero halo keeps exterior ghost values at zero. It is not odd reflection. The first physical sample remains active, and high-order interior differences do not establish equally high surface accuracy.
- Odd pressure or stress extension mirrors the interior with a sign change; even vertical-face extension does not. These choices must match staggered locations and discrete transposes rather than being copied between operators.
- Improved vacuum, projected constitutive relations, and a variational traction-free wall are distinct constructions. Acoustic VTI and scalar qP TTI closures must not be described as full elastic P/S conversion.

(free-surface-table)=
## 2-D operator comparison

The table describes the pinned implementation above. `A > 0`; face lists use low/high sides of each model axis. Here `z` denotes physical depth; tensor-axis order remains specific to the entry point.

| Operator | Continuum model / boundary target | Current discrete closure | Order and limits |
|---|---|---|---|
| Scalar | Scalar acoustics with a flat pressure-release target `p=0` | Fixed-zero exterior halo; first physical sample remains active, rather than clamping `p[0]` | Interior `accuracy=2/4/6/8`; not an odd image or proof of matching surface order |
| Born | Corresponding pressure-release target for background and linearized scattered pressure | Both fields inherit fixed-zero FD ghosts; first physical sample remains active | Interior `2/4/6/8`; scattered is not total field; interior order does not guarantee boundary order |
| Acoustic VRZ | Velocity/impedance acoustics with a flat `p=0` target | No top PML/buffer; `accuracy//2` zero exterior cells and an active first physical row | Interior `2/4/6/8`; a truncated stencil, not odd reflection |
| Acoustic VRR | Velocity/vector-reflectivity acoustics with a flat `p=0` target | Fixed-zero halo and active first physical row | Interior `2/4/6/8`; continuous variable changes do not establish identical VRZ boundaries |
| Acoustic VTI | Acoustic normal traction `sV=0`; `sH` need not vanish | Projected surface constitutive relation, odd normal stress, even vertical velocity | Interior `2/4/6/8`; surface sources containing `sV` are rejected; not elastic VTI |
| qP TTI | Local approximate qP scalar release `u(x,0)=0` | Six states projected on/above the surface; order-two vertical derivatives on the first `M=accuracy//2` interior rows | Interior `2/4/6/8`, near-wall vertical order two; no odd ghosts or elastic qSV |
| Elastic | Isotropic elastic zero traction: `σ·n=0` | Improved vacuum; row 0 is fixed vacuum and row 1 is the first solid stress row | Interior `2/4/6/8`; not collocated zeroing of all stresses at the first solid node; no acquisition-index shift |
| Elastic VRR | Zero-traction target for a momentum/stress elastic system | Fixed vacuum row, zero vacuum stiffness, half-mass normal face | Interior `2/4/6/8`; more than removing PML; `pz` is not `vz` |
| Elastic TTI | Full 2-D elastic P–SV zero-traction wall `σ·n=0` | Variational wall at `z=-dz/2`; local Schur condensation of wall shear strain and half-mass vertical wall velocity | Free surface only with `accuracy=2, coupling_order=2`; higher orders are rejected, not silently downgraded |
| GSLS | Multi-mechanism viscoacoustic pressure release `p(z=0)=0` | Odd pressure, even vertical faces, paired discrete `H=-Gᵀ` | Interior `2/4/6/8`; pressure sources require `z>=1`; not elastic traction release |

Activation and axes: Scalar/Born use zero width on the selected upper face. Acoustic VRZ/VRR/VTI in `[x,z]` use `[A,A,0,A]`. qP TTI accepts `free_surface=True` or a supported top-zero face list. Ordinary Elastic uses depth-first axes and `[0,A,B,C]`; Elastic VRR and Elastic TTI use `free_surface=True`. GSLS uses `[z,x]` and `[0,bottom,left,right]`. For Elastic TTI, `pml_width=0` alone means a zero-extended exterior, not a free surface. Retain each entry's constraints on the remaining widths and initial states.

Each entry retains its own model, source, and receiver-component constraints. See [Conventions](conventions.md), [Usage](../usage.md), and the [PyTorch backend](../pytorch-backend.md). This chapter does not expand any operator's device or automatic-differentiation support.

(free-surface-experiment)=
## Paired experiment design

Each operator is run twice on the same small 2-D two-layer model: once with positive top PML width and once with its flat-top closure. Within each pair, the physical model, source, receiver locations, requested sampling, and lateral/bottom absorption settings are held fixed; only settings required by the top boundary change. Resulting differences in padding, internal grids, or timestep planning are recorded in metadata rather than being described as identical discrete domains.

Plots identify the actual recorded component. The absorbing-top and free-surface gathers within each pair share one color scale; differences are computed directly from raw records, without normalizing the gathers independently. Different operators can use different source units, fields, components, and amplitude scales. Brightness across operator plots is therefore not a direct physical-amplitude comparison.

The measured common setup uses `80×64` grid samples, `dx=dz=10 m`, a 300 m layer interface, Vp=2000/2600 m/s, a nominal source at `(400,80) m`, and 61 receivers at `x=100…700 m` with nominal depth 50 m. Requested sampling is `dt=1 ms, nt=750`. All runs use PyTorch `2.14.1+cpu`, `backend="torch", execution="eager", memory="full"`, float32, two CPU threads, and disabled gradients.

Scalar/Born/VRZ/VRR/VTI/qP TTI/GSLS use order-four spatial stencils and a 20 Hz Ricker wavelet. The three elastic examples use order-two stencils and a 15 Hz Ricker wavelet; Elastic TTI uses `coupling_order=2`. Elastic Vs=1000/1400 m/s and density=2000/2200 kg/m³. Source amplitudes, PML widths, and anisotropy/attenuation parameters differ between groups and are recorded per JSON. The 15 Hz elastic source avoids the package's five-point S-wavelength warning at 20 Hz on this grid; this is not full-bandwidth dispersion or convergence acceptance.

Nine paired operators use an internal timestep of `1 ms`, 750 internal steps, and subdivision ratio one. Elastic TTI instead uses `0.5 ms`, 1500 internal steps, and ratio two; its public output still contains 750 samples. VTI and the three elastic plots record vertical particle velocity; qP TTI records its scalar `u`. Born records linearized scattered pressure, while the other acoustic plots record scalar pressure under their own source conventions. Time coordinates are not forced to agree across operators: ordinary Elastic velocity uses `(n-0.5)dt`, Elastic VRR records completed steps `dt…nt·dt`, and Elastic TTI aligns records to `0…(nt-1)dt`. Plot axes and raw `time_s` retain each entry point's actual convention.

Every figure links its raw NPZ and run JSON. The NPZ keys `absorbing`, `free_surface`, and `difference` hold the absorbing-top record, free-surface record, and the latter minus the former. `time_s` and `receiver_x_m` supply coordinates. Any retained Born background fields have separate names from the scattered records.

(free-surface-records)=
## Actual CPU seismograms by operator

The differences show the response of the specified example to its upper boundary, not numerical error. A finite top PML is not an exact infinite-domain solution either; treating one record as ground truth requires an additional independent reference.

(free-surface-scalar)=
### Scalar acoustics

The scalar field is recorded with this entry point's source scaling and interpretation. Reflections from a fixed-zero halo must not be relabeled as odd images.

```{raw} html
<figure class="sw-science-figure sw-free-surface-figure" data-operator="scalar">
  <a href="../_static/free-surface/scalar.png"><img src="../_static/free-surface/scalar.png" alt="2-D CPU seismograms for Scalar acoustics: absorbing top, free surface, and raw-record difference with a shared pair scale" loading="lazy" decoding="async"></a>
  <figcaption>Paired 2-D CPU records. The figure and run metadata identify the component, source, and grid settings. <a href="../_static/free-surface/scalar.npz">Raw NPZ</a> · <a href="../_static/free-surface/scalar.json">Run JSON</a></figcaption>
</figure>
```

(free-surface-born)=
### Born scattered field

The main panel shows the Born scattered record separately from the background. The upper boundary affects background and scattered propagation, so the difference is not merely an isolated surface reflection.

```{raw} html
<figure class="sw-science-figure sw-free-surface-figure" data-operator="born">
  <a href="../_static/free-surface/born.png"><img src="../_static/free-surface/born.png" alt="2-D CPU seismograms for Born scattered field: absorbing top, free surface, and raw-record difference with a shared pair scale" loading="lazy" decoding="async"></a>
  <figcaption>Paired 2-D CPU records. The figure and run metadata identify the component, source, and grid settings. <a href="../_static/free-surface/born.npz">Raw NPZ</a> · <a href="../_static/free-surface/born.json">Run JSON</a></figcaption>
</figure>
```

```{raw} html
<figure class="sw-science-figure sw-free-surface-figure" data-operator="born_background">
  <a href="../_static/free-surface/born_background.png"><img src="../_static/free-surface/born_background.png" alt="Born background records shown separately from the linearized scattered records above. Both sets of raw arrays are in the Born NPZ." loading="lazy" decoding="async"></a>
  <figcaption>Born background records shown separately from the linearized scattered records above. Both sets of raw arrays are in the Born NPZ.</figcaption>
</figure>
```

(free-surface-vrz)=
### Acoustic VRZ

The velocity and impedance parameters are held fixed within the comparison. The first physical sample at the top still participates in updates.

```{raw} html
<figure class="sw-science-figure sw-free-surface-figure" data-operator="vrz">
  <a href="../_static/free-surface/vrz.png"><img src="../_static/free-surface/vrz.png" alt="2-D CPU seismograms for Acoustic VRZ: absorbing top, free surface, and raw-record difference with a shared pair scale" loading="lazy" decoding="async"></a>
  <figcaption>Paired 2-D CPU records. The figure and run metadata identify the component, source, and grid settings. <a href="../_static/free-surface/vrz.npz">Raw NPZ</a> · <a href="../_static/free-surface/vrz.json">Run JSON</a></figcaption>
</figure>
```

(free-surface-vrr)=
### Acoustic vector-reflectivity VRR

Acoustic VRR and elastic VRR are different operators. This comparison uses the acoustic scalar record and does not conflate their state variables or boundary implementations.

```{raw} html
<figure class="sw-science-figure sw-free-surface-figure" data-operator="vrr">
  <a href="../_static/free-surface/vrr.png"><img src="../_static/free-surface/vrr.png" alt="2-D CPU seismograms for Acoustic vector-reflectivity VRR: absorbing top, free surface, and raw-record difference with a shared pair scale" loading="lazy" decoding="async"></a>
  <figcaption>Paired 2-D CPU records. The figure and run metadata identify the component, source, and grid settings. <a href="../_static/free-surface/vrr.npz">Raw NPZ</a> · <a href="../_static/free-surface/vrr.json">Run JSON</a></figcaption>
</figure>
```

(free-surface-vti)=
### Acoustic VTI projected closure

This plot records vertical velocity `vz`; receiver index z=5 is at its native half-cell depth of 55 m, not the 50 m integer pressure node. Zero sV at the wall does not make all subsurface stresses zero. Both boundary records contain slow/coda features associated with the additional acoustic-VTI branch. Do not label them Rayleigh waves or classify every late arrival as a free-surface multiple without an independent mode reference.

```{raw} html
<figure class="sw-science-figure sw-free-surface-figure" data-operator="vti">
  <a href="../_static/free-surface/vti.png"><img src="../_static/free-surface/vti.png" alt="2-D CPU seismograms for Acoustic VTI projected closure: absorbing top, free surface, and raw-record difference with a shared pair scale" loading="lazy" decoding="async"></a>
  <figcaption>Paired 2-D CPU records. The figure and run metadata identify the component, source, and grid settings. <a href="../_static/free-surface/vti.npz">Raw NPZ</a> · <a href="../_static/free-surface/vti.json">Run JSON</a></figcaption>
</figure>
```

(free-surface-tti)=
### Local approximate qP TTI

This is a single-field local qP approximation, not full elastic TTI. Interpret surface-state projection together with its second-order near-wall vertical stencils.

```{raw} html
<figure class="sw-science-figure sw-free-surface-figure" data-operator="tti">
  <a href="../_static/free-surface/tti.png"><img src="../_static/free-surface/tti.png" alt="2-D CPU seismograms for Local approximate qP TTI: absorbing top, free surface, and raw-record difference with a shared pair scale" loading="lazy" decoding="async"></a>
  <figcaption>Paired 2-D CPU records. The figure and run metadata identify the component, source, and grid settings. <a href="../_static/free-surface/tti.npz">Raw NPZ</a> · <a href="../_static/free-surface/tti.json">Run JSON</a></figcaption>
</figure>
```

(free-surface-elastic)=
### Elastic improved vacuum

The vacuum row retains the original grid coordinates. The plotted component and its staggering are identified by the figure and run JSON; a velocity record is not pressure.

```{raw} html
<figure class="sw-science-figure sw-free-surface-figure" data-operator="elastic">
  <a href="../_static/free-surface/elastic.png"><img src="../_static/free-surface/elastic.png" alt="2-D CPU seismograms for Elastic improved vacuum: absorbing top, free surface, and raw-record difference with a shared pair scale" loading="lazy" decoding="async"></a>
  <figcaption>Paired 2-D CPU records. The figure and run metadata identify the component, source, and grid settings. <a href="../_static/free-surface/elastic.npz">Raw NPZ</a> · <a href="../_static/free-surface/elastic.json">Run JSON</a></figcaption>
</figure>
```

(free-surface-elastic_vrr)=
### Elastic VRR momentum and stress

The half-mass normal face is part of the boundary discretization. A velocity output is momentum divided by the relevant face density; pz itself remains momentum.

```{raw} html
<figure class="sw-science-figure sw-free-surface-figure" data-operator="elastic_vrr">
  <a href="../_static/free-surface/elastic_vrr.png"><img src="../_static/free-surface/elastic_vrr.png" alt="2-D CPU seismograms for Elastic VRR momentum and stress: absorbing top, free surface, and raw-record difference with a shared pair scale" loading="lazy" decoding="async"></a>
  <figcaption>Paired 2-D CPU records. The figure and run metadata identify the component, source, and grid settings. <a href="../_static/free-surface/elastic_vrr.npz">Raw NPZ</a> · <a href="../_static/free-surface/elastic_vrr.json">Run JSON</a></figcaption>
</figure>
```

(free-surface-elastic_tti)=
### Elastic TTI variational traction-free wall

The pair uses accuracy=2 and coupling_order=2, with the wall half a cell above the first normal-stress node. Do not substitute default higher-order derivatives for this supported combination.

```{raw} html
<figure class="sw-science-figure sw-free-surface-figure" data-operator="elastic_tti">
  <a href="../_static/free-surface/elastic_tti.png"><img src="../_static/free-surface/elastic_tti.png" alt="2-D CPU seismograms for Elastic TTI variational traction-free wall: absorbing top, free surface, and raw-record difference with a shared pair scale" loading="lazy" decoding="async"></a>
  <figcaption>Paired 2-D CPU records. The figure and run metadata identify the component, source, and grid settings. <a href="../_static/free-surface/elastic_tti.npz">Raw NPZ</a> · <a href="../_static/free-surface/elastic_tti.json">Run JSON</a></figcaption>
</figure>
```

(free-surface-gsls)=
### Multi-mechanism GSLS viscoacoustics

The attenuation model is held fixed, so this is a boundary comparison rather than a lossy/lossless comparison. It uses hao1, five mechanisms, nominal Q0=60, a 1–200 Hz design band, and a 20 Hz reference frequency. Actual Q_eff(20 Hz) is about 60.1720 and the maximum relative inverse-Q fit error over the band is about 11.1017%; the original fit warning is retained. This is not exact constant Q=60. Odd pressure and even vertical faces must retain their discrete transpose pairing.

```{raw} html
<figure class="sw-science-figure sw-free-surface-figure" data-operator="gsls">
  <a href="../_static/free-surface/gsls.png"><img src="../_static/free-surface/gsls.png" alt="2-D CPU seismograms for Multi-mechanism GSLS viscoacoustics: absorbing top, free surface, and raw-record difference with a shared pair scale" loading="lazy" decoding="async"></a>
  <figcaption>Paired 2-D CPU records. The figure and run metadata identify the component, source, and grid settings. <a href="../_static/free-surface/gsls.npz">Raw NPZ</a> · <a href="../_static/free-surface/gsls.json">Run JSON</a></figcaption>
</figure>
```


(free-surface-mirror)=
## Mirror sources: checking pressure release independently

For a flat pressure-release plane, place the physical source on one side and a reflected source with opposite amplitude on the other. In a homogeneous medium, or with material coefficients, discrete stencils, and exterior treatment symmetrically extended across the plane, the full-space physical-plus-negative-image solution restricted to the half-space provides a pressure-release reference. The image location is `z_image = 2*z_wall - z_source`, not an unconditional negation of an array index. This construction is not an elastic traction-free P/S conversion solution and does not apply unchanged to a general heterogeneous model lacking symmetric extension.

A separate actual Scalar CPU mirror experiment was run, rather than only checking zero values on the wall: velocity 1500 m/s; physical wall `z=0`; source `(0,60 m)` and negative image `(0,-60 m)`; nine receivers at `z=30 m, x=-120…120 m`. A 25 Hz Ricker wavelet and `dt=0.5 ms` are used, with raw pressure compared over `0–0.24 s`. The half-space first active sample is explicitly located at `z=h`, aligning the order-two zero-halo wall with the full-space image plane. The shortest geometric exterior return is at least 0.32 s; extending the exterior extent from 300 m to 360 m did not change the checked records in this window.

Full-gather relative L2 differences, divided by the corresponding full-space image-reference norm:

| Interior order | h=10 m | h=7.5 m | h=5 m |
|---|---:|---:|---:|
| 2 | 0 | 0 | 0 |
| 4 | 5.496% | 4.638% | 3.384% |
| 8 | 8.876% | 6.977% | 4.774% |

Order two matches sample for sample in this floating-point run and time window. This validates the tested wall location, source sign, and discrete symmetry, not zero error against the continuum. Order-four and order-eight fixed-zero halos are not odd extensions and retain visible waveform differences. Refinement reduces those differences, but these three grids do not establish fourth- or eighth-order surface convergence. The order-two result cannot certify higher orders or other operators. Orders four and eight are each compared with their own same-order full-space reference. The larger order-eight difference measures a boundary-closure discrepancy, not proof that eighth-order interior propagation is less accurate than fourth order.

```{raw} html
<figure class="sw-science-figure sw-free-surface-figure" data-validation="scalar_mirror">
  <a href="../_static/free-surface/scalar_mirror.png"><img src="../_static/free-surface/scalar_mirror.png" alt="Scalar 2-D CPU negative-image-source validation with matched wall locations: raw waveforms, differences, and refinement for orders two, four, and eight" loading="lazy" decoding="async"></a>
  <figcaption>A separate homogeneous mirror validation, not an eleventh absorbing-top/free-surface operator pair. The full-space reference uses the same interior finite differences, not an independent continuum solver. <a href="../_static/free-surface/scalar_mirror.npz">Raw NPZ</a> · <a href="../_static/free-surface/scalar_mirror.json">Validation JSON</a></figcaption>
</figure>
```

(free-surface-verification)=
## Further acceptance checks for each operator

“Zero on the surface,” “backpropagation runs,” and “two backends agree” are not complete physical acceptance. The following is a recommended verification matrix. Apart from the Scalar mirror experiment above, these specialized acceptance checks were not all executed for this paired demonstration and must not be read as all passing.

| Operator | Independent physical or discrete reference | Additional checks |
|---|---|---|
| Scalar | Homogeneous negative image; refinement at a fixed wall; horizontal-layer reference | Arrival time, phase, reflection amplitude, and multiples; separate interior dispersion from surface error |
| Born | Finite differences/linearization of the background propagator with the same boundary | Linear perturbation scaling, residual order, source-term derivative; Born does not include arbitrary-order multiple scattering by the perturbation |
| VRZ / acoustic VRR | Consistent physical velocity, density, impedance, and discrete reflectivity, compared with the corresponding variable-density acoustic baseline | Align source units, observables, grids, and walls; continuous algebraic equivalence is not discrete equivalence |
| Acoustic VTI | Isotropic limit and an independent anisotropic reference or manufactured solution | Normal-traction closure, near-surface reflection phase/amplitude, and anisotropy range; do not require this acoustic model to reproduce physical elastic qSV |
| qP TTI | Isotropic/zero-tilt controls; an independent reference and refinement for this local approximate equation | Scalar pressure release, near-wall order, and tilt dependence; a full elastic solution is not pointwise truth for the same PDE |
| Elastic / Elastic VRR | Isotropic oblique P/S reflection coefficients; independent half-space/Lamb or Rayleigh reference | Normal/tangential traction residuals at staggered locations, conversion, surface waves, and long-time stability; VRR additionally requires consistent reflectivity and face density |
| Elastic TTI | Isotropic limit, independent anisotropic P/S half-space reference, and manufactured solution | Half-cell wall, variational closure, and traction residuals; the current free surface supports only accuracy two / coupling order two |
| GSLS | Symmetric-image/frequency-domain reference in homogeneous attenuating media; lossless limit | Pressure release, dispersion/attenuation, relaxation-state evolution and parameter gradients; `H=-Gᵀ` and dot-product tests cover only part of the discrete algebra |

Every test needs a declared error metric, time window, grid, tolerance, and reference independence. Extra arrivals in layered or complex models may be physical free-surface multiples, but visual appearance alone cannot identify every mode.

(free-surface-interpretation)=
## What the comparisons do and do not establish

These examples establish that the stated source, backend, and parameters produced finite, comparable records and show reflections, interference, and later arrivals caused by changing the upper condition. They do not by themselves establish surface convergence order, correct reflection coefficients for arbitrary incidence or media, long-time stability, adjoint/gradient correctness, or CPU/CUDA equivalence.

In particular, the Scalar/Born/VRZ/VRR zero halos are not odd images. A visually plausible gather does not remove that distinction. Physical-accuracy acceptance needs separate analytic or manufactured-reference comparisons at a fixed physical wall, grid refinement, oblique-incidence reflection coefficients, and sufficiently long stability runs, with PML, source, and sampling errors identified.

Only a flat top is represented here; no topographic free-surface capability is inferred. These 2-D examples are not acceptance evidence for any 3-D implementation.

(free-surface-reproduce)=
## Data and reproduction scope

Download the <a href="../_static/free-surface/manifest.json">aggregate manifest</a> to check the source commit, per-operator backend, dtype, file sizes, and SHA-256 hashes. Each run JSON gives its model, acquisition, timesteps, boundary settings, and norms. Reproduce with the matching source and recorded parameters rather than assuming an installed wheel has the same API.

Reproduction scripts: [seven acoustic operators](https://github.com/StarrMoonn/StarWave-docs/blob/main/tools/run_free_surface_acoustic.py), [three elastic operators](https://github.com/StarrMoonn/StarWave-docs/blob/main/tools/run_free_surface_elastic.py), and [Scalar mirror test](https://github.com/StarrMoonn/StarWave-docs/blob/main/tools/run_scalar_mirror.py). They use ordinary `import starwave`; first install the source matching this chapter's pinned commit. Consult each script's `--help` for its command and output destination.

The numerical files are CPU experiment outputs and do not include StarWave kernel source. This documentation supplement does not modify stable kernels or announce a new PyPI binary release.

(free-surface-source)=
## Implementation evidence

Implementation descriptions are pinned to one commit so later default-branch changes cannot silently change their meaning:

- [Scalar public contract](https://github.com/StarrMoonn/StarWave/blob/200621cc31c0ebeebaa01ffaa3591c0ebd7e0adb/starwave/scalar.py) and [per-face fixed-zero boundary description](https://github.com/StarrMoonn/StarWave/blob/200621cc31c0ebeebaa01ffaa3591c0ebd7e0adb/docs/reference/SCALAR_FACE_BOUNDARIES.md).
- [Born public interface](https://github.com/StarrMoonn/StarWave/blob/200621cc31c0ebeebaa01ffaa3591c0ebd7e0adb/starwave/scalar_born.py) and [Born guide](https://github.com/StarrMoonn/StarWave/blob/200621cc31c0ebeebaa01ffaa3591c0ebd7e0adb/docs/guides/BORN_DEEPWAVE.md).
- [qP TTI public contract](https://github.com/StarrMoonn/StarWave/blob/200621cc31c0ebeebaa01ffaa3591c0ebd7e0adb/starwave/tti.py) and [near-wall reference update](https://github.com/StarrMoonn/StarWave/blob/200621cc31c0ebeebaa01ffaa3591c0ebd7e0adb/starwave/_tti_reference.py).
- [Elastic improved-vacuum preparation](https://github.com/StarrMoonn/StarWave/blob/200621cc31c0ebeebaa01ffaa3591c0ebd7e0adb/starwave/_elastic_free_surface.py) and [Elastic VRR vacuum/half-mass closure](https://github.com/StarrMoonn/StarWave/blob/200621cc31c0ebeebaa01ffaa3591c0ebd7e0adb/starwave/_elastic_vr_free_surface.py).
- [Elastic TTI public contract](https://github.com/StarrMoonn/StarWave/blob/200621cc31c0ebeebaa01ffaa3591c0ebd7e0adb/starwave/elastic_tti.py).
- [VTI projection and image differences](https://github.com/StarrMoonn/StarWave/blob/200621cc31c0ebeebaa01ffaa3591c0ebd7e0adb/starwave/torch_backend/_vti_portable.py) and [GSLS discrete-transpose closure](https://github.com/StarrMoonn/StarWave/blob/200621cc31c0ebeebaa01ffaa3591c0ebd7e0adb/starwave/torch_backend/_gsls_free_surface.py).

Source contracts and available test entry points are implementation evidence, not a statement that all tests were executed for this chapter. Executed scope is recorded in the actual run metadata.
