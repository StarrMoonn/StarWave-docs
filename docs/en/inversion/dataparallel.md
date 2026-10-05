# Shot-wise Splitting with DataParallel

The key to a multi-GPU setup is to **keep the full model in the Module and split only the acquisition tensors along the shot dimension**. Passing a 2D model as an ordinary forward input causes DataParallel to split it along its first dimension, breaking up the model's spatial domain.

## Minimal integration snippet

The following snippet requires an existing valid CUDA model, acquisition data for at least two shots, and observations. It is not a standalone program. `ScalarForward` is a class defined by this tutorial, not a `starwave.Scalar` API.

```python
import torch
import starwave

class ScalarForward(torch.nn.Module):
    def __init__(self, velocity):
        super().__init__()
        self.velocity = torch.nn.Parameter(velocity)

    def forward(self, amplitudes, sources, receivers):
        return starwave.scalar(
            self.velocity, 10.0, 0.001,
            source_amplitudes=amplitudes,
            source_locations=sources,
            receiver_locations=receivers,
            pml_freq=15.0,
        )[0]

device_ids = [0, 1]  # Replace with logical IDs visible to the current process.
starwave.prepare_native(device_ids)  # Call in the main thread before creating multi-GPU workers.
primary = torch.device(f"cuda:{device_ids[0]}")
module = ScalarForward(initial_velocity.to(primary))
parallel = torch.nn.DataParallel(module, device_ids=device_ids, dim=0)
optimizer = torch.optim.Adam(module.parameters(), lr=1.0)
optimizer.zero_grad(set_to_none=True)
prediction = parallel(
    amplitudes.to(primary), sources.to(primary), receivers.to(primary)
)
loss = torch.nn.functional.mse_loss(prediction, observed.to(primary))
loss.backward()
optimizer.step()
```

The three inputs have shapes `[B,S,T]`, `[B,S,D]`, and `[B,R,D]`, respectively, and use the same `B` in a given call. Each replica receives the full model and a subset of shots. Records are gathered on the first GPU before a single global objective is computed. Keep other fixed models in buffers or explicitly fixed parameters.

## Checks before use

`CUDA_VISIBLE_DEVICES` remaps logical device IDs. Check `torch.cuda.device_count()` first, and ensure that the model and observations are on the first GPU. Multi-GPU execution does not automatically resolve insufficient memory for a single shot. Too few shots can also leave some GPUs idle.

This snippet has been reviewed for syntax and API call shapes, but has not yet been validated on multiple GPUs. Compare single-GPU and multi-GPU records, global loss, and model gradients using identical inputs. Do not claim that DDP is supported or that multi-GPU execution achieves linear speedup. See [PyTorch DataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html).
