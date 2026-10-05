# INR: Representing Models with an Implicit Network

An INR (Implicit Neural Representation) uses a network to predict model values from fixed spatial coordinates, then trains the network parameters through propagation error. The propagation API still receives a complete CUDA float32 model. INR is neither a built-in StarWave propagation equation nor a public network class.

## Single-GPU computation graph

```text
Fixed coordinates → User network → Full velocity model → starwave.scalar → Records → Loss
                         ↑            Model gradients flow back through autograd
```

The following integration snippet assumes that `network`, `coords`, the acquisition tensors, and `observed` are prepared on the same CUDA device. `coords` contains `nx*nz` coordinates, and the network outputs one value per coordinate:

```python
optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
optimizer.zero_grad(set_to_none=True)
raw = network(coords).reshape(nx, nz)
velocity = 1500.0 + 1000.0 * torch.sigmoid(raw)
prediction = starwave.scalar(
    velocity, 10.0, 0.001,
    source_amplitudes=amplitudes,
    source_locations=sources,
    receiver_locations=receivers,
    pml_freq=15.0,
)[0]
loss = torch.nn.functional.mse_loss(prediction, observed)
loss.backward()
optimizer.step()
```

The 1500–2500 m/s mapping and learning rate are illustrative only. Do not call `detach()` on `velocity` or wrap it in a new `nn.Parameter`; either would sever the computation graph to the network. Train only the network parameters. Using an INR does not turn a fixed source waveform into a differentiable StarWave waveform input.

## Combining INR with multiple GPUs

When using DataParallel, put the network in a Module and register the fixed coordinates as a buffer. Each replica computes the complete model inside forward, and only the acquisition shot dimension is split. PyTorch aggregates the network parameter gradients. Prior parameterization, coordinate normalization, learning rate, network architecture, and the observational objective are all experimental-design choices.

This page introduces the concepts and integration pattern. A complete standalone network script, single-/multi-GPU numerical comparisons, network dependency versions, and real FWI convergence and performance results are still pending. No private notebooks, tool implementations, or datasets are included.
