(reconstruction)=
# Wavefield Reconstruction

This chapter first explains 5.0.0 Scalar3D Radius-M six-face pressure reconstruction, followed by the existing elastic derivation.

StarWave's elastic `memory="boundary"` mode reconstructs the forward physical fields while the loss gradient travels backward. It replaces time histories of volume-wide material derivatives with a **boundary tape and a terminal physical state**. This chapter explains the native 2D/3D method in the 4.0.0 source contract. For arguments and runnable call patterns, see {ref}`Elastic Function <elastic>` and its {ref}`memory options <elastic-memory>`.

**Reconstruction and adjoint propagation have different jobs.** Reconstruction recovers earlier forward values. The adjoint carries the sensitivity of a loss to those values. They meet at matching time levels to accumulate model gradients. Neither task is equivalent to playing the receiver traces backward.

```{raw} html
<figure class="sw-reconstruction-figure">
  <div class="sw-reconstruction-scroll" role="region" tabindex="0" aria-label="Horizontally scrollable schematic">
    <img src="../_static/reconstruction/flow-en.svg" width="760" height="540" alt="Forward modeling saves a boundary tape and terminal physical state; reconstructed forward fields and separate adjoint fields meet in material-gradient accumulation." loading="lazy">
  </div>
  <figcaption>Figure 1. Data flow of the native boundary method. The saved forward information supplies reconstruction, while loss cotangents of records or terminal outputs drive the adjoint. Both trajectories run through decreasing time indices; they are separate fields with separate roles.</figcaption>
</figure>
```


(reconstruction-scalar3d)=
## Scalar3D: Radius-M six-face pressure reconstruction

Public 5.0.0 uses a separate pressure-reconstruction path for 3D `starwave.scalar`. Its pressure storage differs from the elastic velocity/traction tape discussed below; the formulas in this section apply only to scalar3D. See {ref}`Scalar Function <scalar>` for model axes, source units, temporal resampling, and calls.

Let `M=accuracy//2` be the stencil radius, `a=pml_width`, `b=boundary_buffer`, and {math}`n_x,n_y,n_z` the model-axis sizes. Padding on each side is {math}`p=a+b+M`, giving runtime sizes {math}`L_i=n_i+2p`. Each of the six independent pressure faces is exactly M cells wide, with tangential sizes {math}`Q_i=L_i-2(a+M)=n_i+2b`. Public source/receiver coordinates still index the physical model directly, without adding p manually.

At every internal step, the low/high z, low/high y, and low/high x pressure faces are stored. Two final full padded pressure fields seed the reverse second-order time recurrence. Face values supply the interior stencil without directly inverting all dissipative PML states. Reconstructed forward pressure and a separate adjoint meet at matching time levels to accumulate velocity gradients. This is neither reversed playback of receiver traces nor reuse of the elastic staggered velocity/stress updates below.

Let B be the shot count and N the internal step count after CFL resampling (`N=T*r`), with e=4 bytes per FP32 element. These are the main allocated saved payloads, not GPU peak memory:

```{math}
\begin{aligned}
M_{\mathrm{faces}} &= eBN\,2M\,(Q_yQ_z+Q_xQ_z+Q_xQ_y),\\
M_{\mathrm{terminal}} &= 2eB L_xL_yL_z,\\
M_{\mathrm{full}} &= eBN L_xL_yL_z.
\end{aligned}
```

The six faces are allocated separately, so overlapping strips are counted separately in the first expression. Boundary mode saves the first two terms. Full mode saves the third, an unscaled `Lap(u)` history rather than every pressure/PML state. Both retain history only when backpropagation is needed; current 3D source-only gradients also allocate the chosen history. With fixed PML, buffer, and order, boundary time history grows with surface area and full with volume, but boundary still grows linearly with duration and shots.

Models, source/receiver arrays, propagation pressure, adjoints, reconstruction workspaces, gradients, and the autograd graph require additional memory. CPML auxiliary state uses directional slabs, which does not eliminate volume-sized work fields. Native signed 32-bit indexing and CUDA launch-grid limits also constrain actual dimensions; oversized requests are rejected. There is no CPU/disk offload, compression, automatic full fallback, or public checkpoint/restart state.

Both modes require CUDA FP32 on the default stream, support orders 2/4/6/8, and permit one first-order backward per forward. Velocity and source gradients include the source-location `-v²*internal_dt²` factor and the actual temporal-resampling transposes, with CFL/PML setup fixed. Model edges retain replicate-forward/crop-backward; full does not supply the complete transpose of extension. 3D illumination is unsupported.

These are source/public-wheel storage and interface contracts; this documentation update did not rerun a GPU. Compare full/boundary records, velocity gradients, source gradients, and measured peak memory for each real configuration, and check long-time stability separately. Neither six-face reconstruction nor compilation-target coverage establishes production-scale memory fit or GPU numerical acceptance.

(reconstruction-state)=
## The staggered elastic state

For isotropic elasticity, velocity and stress obey the following interior equations. Here {math}`\beta=1/\rho`, {math}`\lambda,\mu` are the Lamé parameters, {math}`f_i` is a body-force source, and {math}`p` is the pressure-source convention used by the API:

```{math}
\partial_t v_i=\beta\sum_j\partial_j\sigma_{ij}+\beta f_i,
\qquad
\partial_t\sigma_{ij}
=\lambda\delta_{ij}\nabla\!\cdot v
+\mu(\partial_i v_j+\partial_j v_i)-p\delta_{ij}.
```

StarWave uses a staggered spatial grid and a staggered time update. Write the pre-step physical state as {math}`x^n=(v^{n-1/2},\sigma^n)`. The complete forward state {math}`X^n` also includes PML memory fields. There are five physical fields in 2D and nine in 3D; the complete state has 13 and 27 fields, respectively. The API axes are `[y,x]` and `[z,y,x]`.

With internal timestep {math}`\Delta t`, staggered buoyancies {math}`B_i`, discrete divergence {math}`D_\sigma`, and constitutive update {math}`C D_v`, the **undamped interior** step is

```{math}
\begin{aligned}
v^{n+1/2}&=v^{n-1/2}+\Delta t\,B D_\sigma\sigma^n+s_v^n,\\
\sigma^{n+1}&=\sigma^n+\Delta t\,C D_vv^{n+1/2}+s_\sigma^n.
\end{aligned}
```

The prepared sources are {math}`s_{v_i}^n=\Delta t B_i f_i^n` and {math}`s_\sigma^n=-\Delta t p^n I`, after frontend time resampling. Velocity is updated first, then stress. Receiver extraction and tape capture happen **before** this pair of updates. The half-step notation therefore matters for both source removal and gradient alignment. PML derivatives replace these interior derivatives near absorbing boundaries; the reconstruction does not assume that the dissipative PML update is stably invertible.

(reconstruction-tape)=
## Record the boundary information that the stencil needs

At each propagated internal step, StarWave packs {math}`\mathcal B^n` from the pre-step state:

- All velocity components in the union of the PML slabs, including the active high-side half-grid plane. Intersections are stored once.
- Direction-specific traction components in narrow strips whose depth is at most the finite-difference radius {math}`r=\mathrm{accuracy}/2`. For a face normal to axis {math}`a`, the required stress components are {math}`\sigma_{ia}`.

StarWave also retains **one** final physical state {math}`x^N` for starting the reverse trajectory, separately from the time tape.

```{raw} html
<figure class="sw-reconstruction-figure">
  <div class="sw-reconstruction-scroll" role="region" tabindex="0" aria-label="Horizontally scrollable schematic">
    <img src="../_static/reconstruction/domain-en.svg" width="760" height="624" alt="Two- and three-dimensional domains with blue PML velocity shells and orange direction-specific traction strips around an undamped interior." loading="lazy">
  </div>
  <figcaption>Figure 2. Geometry of the saved physical information. Blue denotes the velocity shell and orange the narrow traction strips. The 3D cutaway exposes the interior schematically; tape widths and the high-side half-grid plane follow the discrete layout, not the drawing scale.</figcaption>
</figure>
```

In 2D, y-normal strips carry `sigmayy, sigmaxy`; x-normal strips carry `sigmaxy, sigmaxx`. In 3D, z-, y-, and x-normal strips carry the three corresponding traction components. These traction blocks may overlap, so a shear value can appear in more than one block. This is a stencil-dependent layout, not a claim that one boundary row is always sufficient.

Both dimensions use this velocity/traction tape. The tape is **not a time history of PML memory variables**, nor a complete history of stresses throughout the absorbing region. Terminal PML fields remain public outputs, initial PML states remain available as inputs, and their adjoints still participate in backward propagation. See the pinned [layout and allocation](https://github.com/StarrMoonn/StarWave/blob/17fc161363b60879a94029d0af2d3f97f23879de/starwave/_deepwave/boundary_loop.py#L95-L115).

(reconstruction-reverse)=
## Reconstruct in reverse update order

Starting from {math}`x^{n+1}`, a reverse step performs the following operations:

1. Subtract the pressure-source increment from the normal stresses.
2. Undo the interior stress update using {math}`v^{n+1/2}` and restore the saved {math}`\sigma^n` traction strips.
3. Subtract the force-source increment from velocity.
4. Undo the interior velocity update using the recovered {math}`\sigma^n`, and replace PML-shell velocities with their recorded {math}`v^{n-1/2}` values.

For the undamped interior, this is the algebraic reversal

```{math}
\begin{aligned}
\sigma^n&=\sigma^{n+1}-s_\sigma^n-\Delta t\,C D_vv^{n+1/2},\\
v^{n-1/2}&=v^{n+1/2}-s_v^n-\Delta t\,B D_\sigma\sigma^n.
\end{aligned}
```

The force source must remain in {math}`v^{n+1/2}` until stress reconstruction has used that field. Reversing the source-removal order would reconstruct a different trajectory. Boundary restoration supplies the exterior stencil values that cannot be recovered by simply reversing absorption.

```{raw} html
<figure class="sw-reconstruction-figure">
  <div class="sw-reconstruction-scroll" role="region" tabindex="0" aria-label="Horizontally scrollable schematic">
    <img src="../_static/reconstruction/timeline-en.svg" width="760" height="575" alt="Forward updates velocity then stress; reverse reconstruction removes pressure, reverses stress, removes force and reverses velocity, separately from transposed adjoint updates." loading="lazy">
  </div>
  <figcaption>Figure 3. A staggered step and its physical reconstruction. Read the lower row from right to left. The force contribution remains in velocity until the stress update has been undone. Adjoint substeps and gradient contractions are interleaved in the actual kernels.</figcaption>
</figure>
```

The kernels reconstruct interior stresses, the necessary traction strips, and all required velocities. They deliberately do not reconstruct every outer-PML stress. Floating-point subtraction also need not undo floating-point addition bit for bit. Stable forward propagation, matching source/initial-state conventions, identical geometry and coefficients, and a consistent boundary tape are essential; test long trajectories and sharp material contrasts in the precision you intend to use. This construction is not a general inverse of a dissipative wave solver.

(reconstruction-gradient)=
## Couple reconstruction to the discrete adjoint

Let {math}`X^{n+1}=F_n(X^n,m)` denote a full discrete forward step and let {math}`r^n` be the receiver-data cotangent. The adjoint recurrence is

```{math}
\bar X^n=(\partial_XF_n)^T\bar X^{n+1}+R^T r^n.
```

This is a **transpose-Jacobian action**, not {math}`F_n^{-1}\bar X^{n+1}`. StarWave keeps reconstructed physical fields and adjoint fields in separate buffers. Its CUDA reverse loop interleaves reconstruction, source-gradient extraction, material-image accumulation and transposed updates. Receiver cotangents enter at the end of each reverse step because receiver values were recorded at the start of the corresponding forward step.

In the undamped interior, the material images contain the following contractions at their matching substeps:

```{math}
\begin{aligned}
g_\lambda &\mathrel{+}=\kappa_n\Delta t
\Big(\sum_i\bar\sigma_{ii}\Big)\Big(\sum_i D_i v_i\Big),\\
g_{\mu,\mathrm{normal}} &\mathrel{+}=2\kappa_n\Delta t\sum_i\bar\sigma_{ii}D_i v_i,\\
g_{\mu_{ij}} &\mathrel{+}=\kappa_n\Delta t\,\bar\sigma_{ij}(D_i v_j+D_j v_i),\quad i<j,\\
g_{B_i}&\mathrel{+}=\kappa_n\bar v_i\,\Delta v_i/B_i.
\end{aligned}
```

Here {math}`\Delta v_i=v_i^{n+1/2}-s_{v_i}^n-v_i^{n-1/2}` excludes the force-source increment. In the PML shell, the two velocities are available from reconstruction and the tape, so this buoyancy image does not need a history of PML stresses. It requires nonzero active staggered {math}`B_i` when buoyancy gradients are requested. Force-source dependence on buoyancy is differentiated separately through source preparation.

For Lamé images, PML derivative filtering must also be included. A single direction has the recurrence {math}`m^{n+1}=a m^n+b d^n` and filtered derivative {math}`d^n+m^{n+1}`. To evaluate {math}`G=\sum_n w_n(d^n+m^{n+1})`, the implementation initializes {math}`G=0`, {math}`R=0` and traverses time backward:

```{math}
G\mathrel{+}=[(1+b)w_n+bR]d^n,\qquad
R\leftarrow a(w_n+R),\qquad
G\mathrel{+}=R m^0\ \text{after the loop}.
```

The weight {math}`w_n` contains the sampled, matching stress cotangent; the timestep and constitutive factors follow the contractions above. Here {math}`\kappa_n` is a gradient-sampling weight, not a PML stretching coefficient.

This transpose of the temporal filter multiplies by the damping coefficient; it does not divide by it. It also retains the contribution of a nonzero initial PML memory. The [2D](https://github.com/StarrMoonn/StarWave/blob/17fc161363b60879a94029d0af2d3f97f23879de/starwave/_deepwave/boundary_loop.h#L94-L102) and [3D](https://github.com/StarrMoonn/StarWave/blob/17fc161363b60879a94029d0af2d3f97f23879de/starwave/_deepwave/boundary_loop_3d.h#L59-L73) kernels implement this filter alongside the material images.

The frontend then differentiates spatial material averaging, survey extraction, replicated padding and time resampling. After that chain, the {ref}`public conversion <elastic-conversions>` maps raw-input gradients to velocity and density. Its actual regularized buoyancy is {math}`\beta=\rho/(\rho^2+\epsilon)`, so

```{math}
\begin{aligned}
g_{v_p}&=2\rho v_p g_\lambda,\\
g_{v_s}&=2\rho v_s(g_\mu-2g_\lambda),\\
g_\rho&=(v_p^2-2v_s^2)g_\lambda+v_s^2g_\mu
+\frac{\epsilon-\rho^2}{(\rho^2+\epsilon)^2}g_\beta.
\end{aligned}
```

These gradients apply to the complete parameterization chain, not to a hand-assembled subset of stress correlations. The familiar density term {math}`-g_\beta/\rho^2` is only the unregularized limit.

(reconstruction-memory)=
## Memory scales with boundary area and time

Let {math}`d` be dimension, {math}`B` the shot count, {math}`N` the number of propagated internal steps and {math}`e` bytes per element. Let {math}`L_a` include PML but exclude the finite-difference halo. Define low/high saved widths and cell counts by

```{math}
\begin{gathered}
\ell_a=p_{a,-},\qquad h_a=p_{a,+}+\mathbf1[p_{a,+}>0],\qquad V=\prod_a L_a,\\
N_v=V-\prod_a(L_a-\ell_a-h_a),\\
T_a=[\min(\ell_a,r)+\min(h_a,r)]\prod_{b\ne a}L_b,\\
K=d\Big(N_v+\sum_aT_a\Big).
\end{gathered}
```

When at least one material gradient is requested, the exact reconstruction payload is

```{math}
M_\mathrm{tape}=eBNK,\qquad
M_\mathrm{terminal}=eBV\,\frac{d(d+3)}2.
```

Source/initial-state-only differentiation requires no time tape. If all PML widths are zero, {math}`K=0`. These cases do not remove the remaining states and workspaces.

Thus the terminal seed has five fields in 2D or nine in 3D. For fixed PML width and accuracy, {math}`K` grows like perimeter in 2D and surface area in 3D. The time tape still grows linearly with both duration and shot count. A large 3D boundary tape can therefore remain expensive.

```{raw} html
<figure class="sw-reconstruction-figure">
  <div class="sw-reconstruction-scroll" role="region" tabindex="0" aria-label="Horizontally scrollable schematic">
    <img src="../_static/reconstruction/memory-en.svg" width="760" height="568" alt="Full mode stores sampled volume-wide material derivative histories; boundary mode stores a tape at every internal step plus terminal and working states." loading="lazy">
  </div>
  <figcaption>Figure 4. Storage structure, not a measured memory or performance chart. Full uses selected, sampled derivative histories. Boundary changes their time-volume scaling to time-boundary scaling plus volume-sized state/workspace terms; it does not eliminate time dependence or GPU working memory.</figcaption>
</figure>
```

The actual `full` mode stores **selected material-derivative volumes**, not every field of the complete forward state. If {math}`s` is the material-gradient sampling stride, its history count is proportional to {math}`BNV_\mathrm{padded}H/s`, where

```{math}
H=d\,\mathbf1[\lambda\text{ or }\mu]
+\frac{d(d-1)}2\,\mathbf1[\mu]
+d\,\mathbf1[\beta].
```

Here the indicators mean that a gradient is requested; the maximum is five channels in 2D and nine in 3D. Boundary storage can lose its advantage on small domains, with thick PML, or with aggressive full-history time subsampling.

Neither payload formula is total peak GPU memory. Models and their autograd graph, initial states, source/receiver arrays, output cotangents, gradients, terminal clones and working fields remain allocated. The adjoint includes its own PML memories. The Lamé image filters use fixed-volume auxiliary arrays in 2D and direction-slab arrays in 3D. Reconstruction adds reverse-stencil work and tape traffic; no universal speedup is implied. CPU/disk offload and compression address different storage costs and are not enabled for this boundary path.

(reconstruction-scope)=
## Supported scope and numerical checks

- **Native execution:** CUDA 2D/3D, `float32`/`float64`, accuracy 2/4/6/8, `storage_mode="device"`, uncompressed storage and `python_backend=False`. Each physical extent must exceed twice the stencil radius. See {ref}`API restrictions <elastic-notes>`.
- **Sampling:** with CFL subdivision {math}`q`, let {math}`I` denote `model_gradient_sampling_interval`. The material-image stride is {math}`s=qI` and {math}`\kappa_n=s\mathbf1[n\bmod s=0]`. Tape capture, state/source adjoints and PML-filter recurrences still advance every propagated internal step. Agreement with native `full` means agreement with this sampled material-gradient convention; it does not establish all-internal-step material AD when {math}`s>1`.
- **Time tails:** this path propagates complete stride groups, {math}`N=\lfloor N_\mathrm{requested}/s\rfloor s`, and pads remaining internal receiver rows with zeros before frontend downsampling. The zero-complete-group edge is not a supported no-op; avoid relying on it.
- **Surfaces and differentiation:** the elastic API has no generic `free_surface` switch. Setting a PML width to zero does not by itself establish a traction-free surface. Density differentiation also requires nonzero active staggered buoyancy. Native higher-order derivatives are unsupported; forward callbacks are read-only observers and boundary backward callbacks are unsupported.

The source tests cover full/boundary comparisons, multiple orders and precisions, asymmetric/zero PML, force and pressure sources, CFL substeps, nonzero initial physical/PML states, shared/per-shot models and long trajectories. Their existence is a **test specification**, not evidence that every case passed for every GPU or for the public wheel. The source release verification records CPU/static checks and explicitly does not claim a fresh CUDA equivalence run. This documentation update did not run a GPU experiment. See [documentation status](../status.md).

For a new workload, first compare `full` and `boundary` under identical settings: receiver records, terminal outputs, and every requested model/source/initial-state gradient. Inspect relative and absolute errors, especially near zero-valued reference gradients. Use directional-derivative checks with a consistent sampling convention; begin with no material-gradient subsampling before assessing a sampled approximation. Check long-time behavior and actual peak memory separately. Algebraic reversibility is not a blanket guarantee of bitwise equality, precision-independent stability or convergence of an inversion.

(reconstruction-references)=
## Theory and implementation references

Boundary saving has an established literature. The references below provide background, while the pinned source links specify the StarWave layout and update order described here. They are not a claim that another paper's formulation, minimum-storage proof or validation transfers unchanged to this implementation.

1. Yang, P., Gao, J. and Wang, B. (2014). [RTM using effective boundary saving: A staggered grid GPU implementation](https://doi.org/10.1016/j.cageo.2014.04.004). *Computers & Geosciences*, 68, 64–72. Background on stencil-dependent boundary storage and CPML; [author-hosted reproducible text](https://ahay.org/RSF/book/xjtu/gpurtm/paper_html/).
2. Aaker, O. E. et al. (2020). [Wavefield reconstruction for velocity–stress elastodynamic full-waveform inversion](https://doi.org/10.1093/gji/ggaa147). *Geophysical Journal International*, 222, 595–609. Background on elastic reconstruction and its role in gradient computation.
3. StarWave source contract, commit `17fc161`: [2D reverse loop](https://github.com/StarrMoonn/StarWave/blob/17fc161363b60879a94029d0af2d3f97f23879de/starwave/_deepwave/boundary_loop.h#L251-L377), [3D reverse loop](https://github.com/StarrMoonn/StarWave/blob/17fc161363b60879a94029d0af2d3f97f23879de/starwave/_deepwave/boundary_loop_3d.h#L264-L367), [allocation and guards](https://github.com/StarrMoonn/StarWave/blob/17fc161363b60879a94029d0af2d3f97f23879de/starwave/_deepwave/boundary.py#L131-L218), and [conversion formula](https://github.com/StarrMoonn/StarWave/blob/17fc161363b60879a94029d0af2d3f97f23879de/starwave/_deepwave/common.py#L2515-L2542).
4. Verification boundaries: [numerical comparison tests](https://github.com/StarrMoonn/StarWave/blob/17fc161363b60879a94029d0af2d3f97f23879de/tests/test_deepwave_boundary_cuda.py), [sampling regression](https://github.com/StarrMoonn/StarWave/blob/17fc161363b60879a94029d0af2d3f97f23879de/tests/test_deepwave_api27.py#L384-L399), and [source release verification](https://github.com/StarrMoonn/StarWave/blob/17fc161363b60879a94029d0af2d3f97f23879de/docs/releases/v11/V11_RELEASE_NOTES.md#executed-validation-and-limits).
