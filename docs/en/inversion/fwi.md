# Introduction to FWI

Full-waveform inversion (FWI) uses a model to generate predicted records, then computes model gradients of an objective comparing predictions with observations. This section demonstrates one scalar parameter update. It is an integration example, not a complete convergence experiment.

## Run one update

Use the same script as in the [Quickstart](../quickstart.md):

```bash
python scalar_demo.py --mode fwi --device 0
```

The script first generates fixed observations using a synthetic true model, then makes predictions from a uniform initial velocity. It computes mean-squared error normalized by observation energy, runs `loss.backward()` once, and performs an Adam update. The learning rate, normalization, fixed edges, and post-update velocity range are explicit choices in this example, not default StarWave policies.

## Key points for the loop

1. Put trainable models in `torch.nn.Parameter`; keep source waveforms, source coordinates, and receiver coordinates fixed.
2. Clear gradients before each update and run a new forward pass at every iteration.
3. Construct a scalar loss from the returned records, and call backward only once per forward pass.
4. Check that the loss and gradients are finite before updating the model; record the constraints and optimization parameters you choose.

```python
# model, observed, inputs, and propagate are prepared by scalar_demo.py.
optimizer.zero_grad(set_to_none=True)
predicted = propagate(model, inputs)
loss = torch.nn.functional.mse_loss(predicted, observed)
loss.backward()
optimizer.step()
```

For real data, also check source timing, components, units, preprocessing, and acquisition geometry. VTI `vz` observations cannot be directly compared with scalar pressure-like output. The two VRZ medium parameterizations also change the optimization variables and the meaning of their gradients.

## Validation limits

Start with small tests on the target device and finite-difference or directional-derivative checks using model perturbations in the interior. Only then consider long time series, multiple shots, and multiple GPUs. Edge gradients are subject to the [known model-extension chain limitation](../modeling/conventions.md). Fixing the edges must not be described as having resolved the entire gradient problem.

This section still lacks error tables for target GPUs, a basis for choosing step sizes, and complete convergence curves. A single loss value or smoke check cannot establish scientific validation.
