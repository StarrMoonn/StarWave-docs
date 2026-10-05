# StarWave User Manual

```{raw} html
<div class="sw-home-hero">
  <div class="sw-home-intro">
    <p class="sw-home-kicker">Differentiable waves · PyTorch · CUDA</p>
    <h2>Wave physics.<br><span>Ready for gradients.</span></h2>
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
    <span class="sw-home-number" aria-hidden="true">01 / PROPAGATE</span>
    <h3>Connect models to observations</h3>
    <p>CUDA wave propagation for 2D scalar acoustics, 2D VRZ, and 2D/3D acoustic VTI. Clear entry points for different descriptions of the medium.</p>
    <a href="usage.html#propagators">Explore the propagators <span aria-hidden="true">→</span></a>
  </div>
  <div>
    <span class="sw-home-number" aria-hidden="true">02 / DIFFERENTIATE</span>
    <h3>Put gradients to work</h3>
    <p>Connect simulated records to a PyTorch loss and backpropagate model gradients. Build FWI experiments with the optimizers you already use.</p>
    <a href="modeling/gradient.html">See the gradient experiment <span aria-hidden="true">→</span></a>
  </div>
  <div>
    <span class="sw-home-number" aria-hidden="true">03 / CONNECT</span>
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
  <a href="installation.html#installation-smoke"><span>01</span><strong>Install and check a forward run</strong><span aria-hidden="true">→</span></a>
  <a href="modeling/gradient.html"><span>02</span><strong>Compute a model gradient</strong><span aria-hidden="true">→</span></a>
  <a href="inversion/fwi.html"><span>03</span><strong>Run a simple FWI experiment</strong><span aria-hidden="true">→</span></a>
</div>
<p class="sw-home-footnote">This manual describes the public 2.0.0 interfaces. The measured A30 tutorial results use 0.1.0.dev9; see <a href="status.html">documentation status</a> for requirements and validation scope. On Windows, start with <a href="wsl.html">WSL 2</a>.</p>
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Getting started

installation
wsl
quickstart
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Forward modeling

modeling/conventions
modeling/scalar
modeling/gradient
modeling/vrz
modeling/vti
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Inversion and neural networks

inversion/fwi
inversion/dataparallel
inversion/inr
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Usage

usage
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Reference and maintenance

faq
release-notes
status
```
