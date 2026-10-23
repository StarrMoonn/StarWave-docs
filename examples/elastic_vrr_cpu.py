"""V21 elastic VRR synthetic CPU example; requires matching authorized source."""
import starwave
import torch

# 2D local-expanded momentum/stress equation; model axes [x, z].
vp = torch.full((24, 24), 2500.0, dtype=torch.float64)
vs = torch.full_like(vp, 1200.0)
rho = torch.full_like(vp, 2000.0)
models = starwave.elastic_vrr_parameters(vp, vs, rho, 10.0)
# Independent six-coordinate optimization leaves (not physical-model AD).
models = tuple(x.detach().clone().requires_grad_() for x in models)
source = torch.zeros((1, 1, 32), dtype=vp.dtype)
source[0, 0, 0] = 1.0  # momentum rate, integrated by the solver
src = torch.tensor([[[12, 10]]], dtype=torch.long)
rec = torch.tensor([[[12, 10], [13, 10]]], dtype=torch.long)
(pz,) = starwave.elastic_vrr(
    *models, 10.0, 0.0002,
    source_amplitudes=source, source_locations=src, receiver_locations=rec,
    source_component="pz", receiver_components=("pz",),
    source_type="momentum_rate", reflectivity_layout="edge",
    accuracy=4, pml_width=4, pml_freq=25.0, max_vel=3000.0,
    backend="torch", execution="eager", memory="full",
)
assert pz.shape == (1, 2, 32)
pz.square().sum().backward()
assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in models)
assert torch.isfinite(pz).all()
print("Elastic VRR CPU forward/backward passed:", tuple(pz.shape))
