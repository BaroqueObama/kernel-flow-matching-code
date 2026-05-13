"""ODE integration wrapper for sample generation via torchdiffeq."""

import torch
import torch.nn as nn
from torch import Tensor
from torchdiffeq import odeint

from src.baselines.plugin_velocity import exact_plugin_velocity_scaled


@torch.no_grad()
def ode_sample(
    model: nn.Module,
    support: Tensor,
    n_samples: int,
    d: int,
    device: torch.device,
    method: str = "euler",
    n_steps: int = 100,
    t_min: float = 1e-5,
    x_0: Tensor | None = None,
) -> Tensor:
    """Integrate the learned velocity field from t_min to 1, returning (n_samples, d) samples."""
    model.eval()
    if support.dim() == 2:
        support = support.unsqueeze(0)
    support_3d = support.expand(n_samples, -1, -1).to(device)

    def velocity_fn(t_scalar: Tensor, x_flat: Tensor) -> Tensor:
        batch = x_flat.shape[0]
        x_3d = x_flat.unsqueeze(1)
        t_clamped = t_scalar.clamp(min=t_min, max=1.0)
        t_3d = t_clamped.unsqueeze(0).expand(batch).unsqueeze(-1).unsqueeze(-1)
        v_3d = model(x_3d, t_3d, support_3d)
        return v_3d.squeeze(1)

    if x_0 is None:
        x_0 = torch.randn(n_samples, d, device=device)
    else:
        assert x_0.shape == (n_samples, d), f"x_0 shape {x_0.shape} != expected ({n_samples}, {d})"
        x_0 = x_0.to(device)
        assert torch.isfinite(x_0).all(), "x_0 contains non-finite values"
    t_span = torch.linspace(t_min, 1.0, n_steps, device=device)
    trajectory = odeint(velocity_fn, x_0, t_span, method=method)
    return trajectory[-1]


@torch.no_grad()
def ode_sample_exact(
    support: Tensor,
    n_samples: int,
    d: int,
    device: torch.device,
    sigma_min: float,
    method: str = "euler",
    n_steps: int = 100,
    t_min: float = 1e-5,
    x_0: Tensor | None = None,
) -> Tensor:
    """Integrate the exact plug-in velocity field (Prop 4.4) from t_min to 1, returning (n_samples, d) samples."""
    support = support.to(device)
    if support.dim() == 2:
        support_expanded = support.unsqueeze(0).expand(n_samples, -1, -1)
    else:
        support_expanded = support.expand(n_samples, -1, -1)

    def velocity_fn(t_scalar: Tensor, x_flat: Tensor) -> Tensor:
        batch = x_flat.shape[0]
        t_clamped = t_scalar.clamp(min=t_min, max=1.0)
        t_batch = t_clamped.unsqueeze(0).expand(batch).unsqueeze(-1)
        return exact_plugin_velocity_scaled(x_flat, t_batch, support_expanded, sigma_min)

    if x_0 is None:
        x_0 = torch.randn(n_samples, d, device=device)
    else:
        assert x_0.shape == (n_samples, d), f"x_0 shape {x_0.shape} != expected ({n_samples}, {d})"
        x_0 = x_0.to(device)
        assert torch.isfinite(x_0).all(), "x_0 contains non-finite values"
    t_span = torch.linspace(t_min, 1.0, n_steps, device=device)
    trajectory = odeint(velocity_fn, x_0, t_span, method=method)
    return trajectory[-1]
