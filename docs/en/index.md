# StarWave User Manual

```{raw} html
<div class="sw-home-hero">
  <div class="sw-home-intro">
    <p class="sw-home-kicker">Differentiable waves · PyTorch · CUDA</p>
    <h2>Wave Physics.<br><span>Automatic Differentiation.</span></h2>
    <p class="sw-home-lead">From a simulated wavefield to a model update. StarWave connects wave propagation with PyTorch autograd, bringing forward modeling, full-waveform inversion, and neural model representations into one differentiable workflow.</p>
    <nav class="sw-home-actions" aria-label="Get started with StarWave">
      <a class="sw-button sw-button-primary" href="installation.html">Install StarWave <span aria-hidden="true">↗</span></a>
      <a class="sw-button" href="quickstart.html">Quickstart <span aria-hidden="true">→</span></a>
      <a class="sw-home-api-link" href="usage.html">Usage / API <span aria-hidden="true">→</span></a>
    </nav>
  </div>
  <div class="sw-home-identity">
    <img class="sw-home-logo" src="_static/brand/starwave-main.svg" alt="StarWave" width="950" height="645" fetchpriority="high">
    <p>SEE A DEEPER EARTH</p>
  </div>
</div>
```

(home-capabilities)=
## From physical models to learnable models

```{raw} html
<div class="sw-home-capabilities">
  <div>
    <svg class="sw-capability-icon" viewBox="0 0 96 64" width="96" height="64" aria-hidden="true" focusable="false"><path d="M10 33h13c5 0 5-18 10-18s7 36 12 36 5-30 10-30 6 12 11 12h20"/><path class="sw-icon-soft" d="M14 46h10m39 0h19M18 56h12m28 0h19"/></svg>
    <h3>Connect models to observations</h3>
    <p>CUDA wave propagation for 2D scalar acoustics, 2D VRZ, and 2D/3D acoustic VTI. Clear entry points for different descriptions of the medium.</p>
    <a href="usage.html#propagators">Explore the propagators <span aria-hidden="true">→</span></a>
  </div>
  <div>
    <svg class="sw-capability-icon" viewBox="0 0 96 64" width="96" height="64" aria-hidden="true" focusable="false"><path d="M12 17h56a13 13 0 0 1 0 26H20m10-10L20 43l10 10"/><path class="sw-icon-soft" d="M42 27v7m12-12v12m12-7v7"/><circle cx="12" cy="17" r="4"/></svg>
    <h3>Put gradients to work</h3>
    <p>Connect simulated records to a PyTorch loss and backpropagate model gradients. Build FWI experiments with the optimizers you already use.</p>
    <a href="modeling/gradient.html">See the gradient experiment <span aria-hidden="true">→</span></a>
  </div>
  <div>
    <svg class="sw-capability-icon" viewBox="0 0 96 64" width="96" height="64" aria-hidden="true" focusable="false"><path class="sw-icon-soft" d="m18 15 27 17-27 17m0-34 27 0 30 17-30 17H18m27-34v34m0-17h30"/><circle cx="18" cy="15" r="5"/><circle cx="18" cy="49" r="5"/><circle cx="45" cy="15" r="5"/><circle cx="45" cy="32" r="5"/><circle cx="45" cy="49" r="5"/><circle cx="75" cy="32" r="7"/></svg>
    <h3>Explore new model representations</h3>
    <p>Start with grid parameters or a user-defined implicit neural representation (INR). Connect physics and learning through a differentiable model.</p>
    <a href="inversion/inr.html">Explore experimental INR wiring <span aria-hidden="true">→</span></a>
  </div>
</div>
```

(home-workflow)=
## Familiar PyTorch. Built around waves.

Propagation produces records, a loss measures their difference, and gradients connect back to the model. This fragment shows the core workflow; see the [Quickstart](quickstart.md) for environment and acquisition setup.

```{raw} html
<div class="sw-home-code">
<p class="sw-home-code-label">Model → propagation → records → loss → gradients</p>
```

```python
import torch
import starwave

predicted = starwave.scalar(
    velocity, grid_spacing=10.0, dt=0.001,
    source_amplitudes=amplitudes,
    source_locations=sources,
    receiver_locations=receivers,
    pml_freq=15.0,
)[0]
loss = torch.nn.functional.mse_loss(predicted, observed)
loss.backward()
```

```{raw} html
</div>
<p class="sw-home-code-note">This fragment assumes the native runtime is ready and the model, fixed acquisition, and observation tensors satisfy the API contracts on one CUDA device. velocity is a float32 model with gradients enabled. The INR guide covers concepts and user-network wiring; full convergence validation is not yet provided.</p>
```

(home-start)=
## Start with a small experiment

```{raw} html
<div class="sw-home-paths">
  <a href="installation.html#installation-smoke"><span class="sw-path-icon"><svg class="sw-tutorial-icon" viewBox="0 0 96 64" width="96" height="64" aria-hidden="true" focusable="false"><rect x="14" y="13" width="66" height="43" rx="5"/><path d="m25 26 9 7-9 7m19 0h14"/><path class="sw-icon-soft" d="M14 21h66"/></svg></span><span class="sw-path-copy"><strong>Install and run</strong><span>Set up your environment and simulate a shot gather.</span></span><span aria-hidden="true">→</span></a>
  <a href="modeling/gradient.html"><span class="sw-path-icon"><svg class="sw-tutorial-icon" viewBox="0 0 96 64" width="96" height="64" aria-hidden="true" focusable="false"><path class="sw-icon-soft" d="M16 53V13m0 40h65"/><path d="m24 43 15-17 15 8 23-19m-13 0h13v13"/></svg></span><span class="sw-path-copy"><strong>Follow the gradient</strong><span>Connect the data residual to a model update direction.</span></span><span aria-hidden="true">→</span></a>
  <a href="inversion/fwi.html"><span class="sw-path-icon"><svg class="sw-tutorial-icon" viewBox="0 0 96 64" width="96" height="64" aria-hidden="true" focusable="false"><path d="M23 22a24 24 0 0 1 43-1m0 0V10m0 11H55M70 44a24 24 0 0 1-43 1m0 0v11m0-11h11"/><path class="sw-icon-soft" d="M37 33h7l4-10 5 20 4-10h7"/></svg></span><span class="sw-path-copy"><strong>Build an inversion</strong><span>Bring propagation, a loss, and an optimizer together.</span></span><span aria-hidden="true">→</span></a>
</div>
<div class="sw-home-next"><p>Start with a small model. Bring your own ideas to the next inversion experiment.</p><a href="quickstart.html">Open the Quickstart →</a><a href="wsl.html">Windows / WSL →</a><a href="status.html">Documentation and validation scope →</a></div>
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Getting started

About StarWave <about>
Installation <installation>
Docker <docker>
WSL <wsl>
Quickstart <quickstart>
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Forward Modeling

Conventions <modeling/conventions>
Acquisition <modeling/acquisition>
Scalar <modeling/scalar>
Gradient <modeling/gradient>
Wave propagation <modeling/wave-propagation>
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Inversion

FWI <inversion/fwi>
DataParallel <inversion/dataparallel>
INR <inversion/inr>
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Usage

Usage <usage>
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Reference

FAQ <faq>
Release notes <release-notes>
Status <status>
```
