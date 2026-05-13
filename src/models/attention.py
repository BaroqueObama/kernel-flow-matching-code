"""Learned multi-head cross-attention with scaled dot-product attention (SDPA)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LearnedCrossAttention(nn.Module):
    """Multi-head cross-attention: n query points attend to m support tokens."""

    def __init__(self, d_model: int, n_heads: int, qk_norm: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qk_norm = qk_norm

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        if qk_norm:
            self.kappa = nn.Parameter(torch.full((n_heads,), math.sqrt(self.d_k)))

    def forward(self, query: Tensor, kv: Tensor) -> Tensor:
        """
        Args:
            query: (B, n, d_model) -- multiple query points per task.
            kv: (B, m, d_model) -- support tokens (shared across queries).

        Returns:
            output: (B, n, d_model).
        """
        assert query.dim() == 3, f"query must be 3D (B, n, d_model), got {query.dim()}D"
        assert kv.dim() == 3, f"kv must be 3D (B, m, d_model), got {kv.dim()}D"
        B, n, _ = query.shape
        m = kv.shape[1]

        q = self.w_q(query).view(B, n, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(kv).view(B, m, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(kv).view(B, m, self.n_heads, self.d_k).transpose(1, 2)

        if self.qk_norm:
            q = F.normalize(q, dim=-1) * self.kappa.view(1, self.n_heads, 1, 1)
            k = F.normalize(k, dim=-1)
            attn_out = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v)

        attn_out = attn_out.transpose(1, 2).reshape(B, n, self.d_model)
        return self.w_o(attn_out)
