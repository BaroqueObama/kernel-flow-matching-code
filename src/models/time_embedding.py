"""Sinusoidal time embedding for scalar time input."""

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for scalar time t -> R^d_model."""

    def __init__(self, d_model: int):
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even, got {d_model}"
        half_dim = d_model // 2
        exponent = torch.arange(half_dim, dtype=torch.float32) * -(math.log(10000.0) / half_dim)
        freqs = torch.exp(exponent)
        self.register_buffer("freqs", freqs)

    def forward(self, t: Tensor) -> Tensor:
        """t: (*batch_dims) -> (*batch_dims, d_model). Squeezes trailing dim=1 if present."""
        if t.dim() >= 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        orig_shape = t.shape
        t_flat = t.reshape(-1).float()
        args = t_flat.unsqueeze(-1) * self.freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb.view(*orig_shape, -1)
