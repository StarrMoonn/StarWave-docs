"""Synthetic scalar smoke check / one FWI step; GPU acceptance is pending.

Run: python examples/scalar_demo.py --mode forward
Run: python examples/scalar_demo.py --mode fwi
No files, datasets, network access, or compilation are required by this script.
"""
import argparse
import math

import torch
import starwave


def acquisition(device):
    """Create one shot with one source and 16 receivers in [x,z] order."""
    nt, dt, frequency = 256, 0.001, 15.0
    time = torch.arange(nt, device=device, dtype=torch.float32) * dt
    a = (math.pi * frequency * (time - 0.08)) ** 2
    wavelet = ((1.0 - 2.0 * a) * torch.exp(-a))[None, None, :]
    sources = torch.tensor([[[16, 10]]], device=device, dtype=torch.long)
    receivers = torch.empty((1, 16, 2), device=device, dtype=torch.long)
    receivers[0, :, 0] = torch.arange(8, 24, device=device)
    receivers[0, :, 1] = 10
    return wavelet, sources, receivers


def propagate(velocity, inputs):
    amplitudes, sources, receivers = inputs
    return starwave.scalar(
        velocity, grid_spacing=10.0, dt=0.001,
        source_amplitudes=amplitudes,
        source_locations=sources,
        receiver_locations=receivers,
        pml_freq=15.0, memory="boundary",
    )[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mode", choices=("forward", "fwi"), default="forward")
    args = parser.parse_args()
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise RuntimeError("Choose an available logical CUDA device.")
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    starwave.prepare_native([args.device])
    inputs = acquisition(device)
    truth = torch.full((32, 32), 1800.0, device=device, dtype=torch.float32)
    truth[10:22, 16:24] = 2000.0
    with torch.no_grad():
        observed = propagate(truth, inputs)
    if observed.shape != (1, 16, 256) or not torch.isfinite(observed).all():
        raise RuntimeError("Unexpected or nonfinite receiver records.")
    if observed.abs().max().item() == 0:
        raise RuntimeError("All receiver records are zero.")
    print("Forward smoke check:", tuple(observed.shape), "finite and nonzero")
    if args.mode == "fwi":
        model = torch.nn.Parameter(torch.full_like(truth, 1800.0))
        optimizer = torch.optim.Adam([model], lr=1.0)
        optimizer.zero_grad(set_to_none=True)
        predicted = propagate(model, inputs)
        # Fixed observation normalization is a tutorial choice, not an API default.
        scale = observed.square().mean().clamp_min(1e-12)
        loss = (predicted - observed).square().mean() / scale
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss.")
        loss.backward()
        if model.grad is None or not torch.isfinite(model.grad).all():
            raise RuntimeError("Missing or nonfinite model gradient.")
        # Hold the physical model edges fixed; this is not a proof of gradient accuracy.
        with torch.no_grad():
            model.grad[:4] = 0
            model.grad[-4:] = 0
            model.grad[:, :4] = 0
            model.grad[:, -4:] = 0
        optimizer.step()
        with torch.no_grad():
            model.clamp_(1500.0, 2500.0)
        print("One FWI wiring step; pre-update loss:", loss.item())
    print("This smoke check is not numerical, performance, or convergence acceptance.")


if __name__ == "__main__":
    main()
