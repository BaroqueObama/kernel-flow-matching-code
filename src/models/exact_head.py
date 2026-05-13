"""Frozen exact attention head implementing Thm 5.2 (Attention Realization)."""

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.flow_matching import T_EPS, bandwidth, sigma_t
from src.utils.kernels import nw_local_mean


class ExactAttentionHead(nn.Module):
    """Frozen Gaussian-kernel cross-attention head implementing Thm 5.2.

    Zero learnable parameters. Computes the exact empirical OT-FM velocity
    field for the given support set via:
      1. De-scale: x_tilde = x / t
      2. NW local mean: m_h(x_tilde; S) with h(t) = sigma_t / t
      3. Affine map: velocity = x_tilde + (m_h - x_tilde) / sigma_t

    Accepts (B, n, d) queries with (B, m, d) support. Internally flattens
    to (B*n, d) for kernel functions, then reshapes back.
    """

    def __init__(self, sigma_min: float):
        super().__init__()
        assert 0 < sigma_min <= 1, f"sigma_min must be in (0, 1], got {sigma_min}"
        self.sigma_min = sigma_min

    @torch.no_grad()
    def forward(self, x: Tensor, t: Tensor, support: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Query points (scaled coords), shape (B, n, d).
            t: Time values, shape (B, n, 1).
            support: Support set, shape (B, m, d).

        Returns:
            velocity: Exact plug-in velocity, shape (B, n, d).
            nw_mean: NW local mean (pre-affine), shape (B, n, d).
        """
        assert x.dim() == 3, f"x must be 3D (B, n, d), got {x.dim()}D"
        assert t.dim() == 3 and t.shape[-1] == 1, f"t must be (B, n, 1), got {t.shape}"
        assert support.dim() == 3, f"support must be 3D (B, m, d), got {support.dim()}D"

        B, n, d = x.shape
        m = support.shape[1]
        orig_dtype = x.dtype

        x_flat = x.reshape(B * n, d).float()
        t_flat = t.reshape(B * n, 1).float()
        support_flat = support.float().unsqueeze(1).expand(B, n, m, d).reshape(B * n, m, d)

        t_flat = t_flat.clamp(min=T_EPS)
        x_tilde = x_flat / t_flat
        h_t = bandwidth(t_flat, self.sigma_min)
        sig_t = sigma_t(t_flat, self.sigma_min)

        m_h = nw_local_mean(x_tilde, support_flat, h_t)
        velocity = x_tilde + (m_h - x_tilde) / sig_t

        return velocity.view(B, n, d).to(orig_dtype), m_h.view(B, n, d).to(orig_dtype)
