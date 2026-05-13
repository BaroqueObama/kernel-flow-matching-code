"""ICFM velocity network: support-conditioned flow matching with exact + learned heads."""

import torch
import torch.nn as nn
from torch import Tensor

from src.models.attention import LearnedCrossAttention
from src.models.exact_head import ExactAttentionHead
from src.models.time_embedding import SinusoidalTimeEmbedding


class TransformerLayer(nn.Module):
    """Cross-attention + FFN with pre-norm residuals."""

    def __init__(self, d_model: int, n_heads: int, qk_norm: bool = False):
        super().__init__()
        self.attn = LearnedCrossAttention(d_model, n_heads, qk_norm=qk_norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, query: Tensor, support_tokens: Tensor) -> Tensor:
        """query: (B, n, d_model), support_tokens: (B, m, d_model) -> (B, n, d_model)."""
        query = query + self.attn(self.norm1(query), support_tokens)
        query = query + self.ffn(self.norm2(query))
        return query


class ICFMVelocityNet(nn.Module):
    """Support-conditioned velocity network for In-Context Flow Matching.

    Queries (B, n, d) attend to a shared support set (B, m, d) via learned
    cross-attention layers.  When use_exact_head=True, a parallel frozen head
    computes the Thm 5.2 plug-in velocity and the output MLP fuses both signals.
    """

    def __init__(
        self,
        d_data: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        use_exact_head: bool,
        sigma_min: float,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.d_data = d_data
        self.d_model = d_model
        self.use_exact_head = use_exact_head

        assert n_heads >= 1, f"Need >= 1 attention head, got {n_heads}"
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.query_proj = nn.Linear(d_data + d_model, d_model)
        self.support_proj = nn.Linear(d_data, d_model)

        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, n_heads, qk_norm=qk_norm) for _ in range(n_layers)]
        )

        if use_exact_head:
            self.exact_head = ExactAttentionHead(sigma_min)
            self.exact_proj = nn.Linear(d_data, d_model)
            self.output_mlp = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_data),
            )
        else:
            self.output_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_data),
            )

    def forward(self, x: Tensor, t: Tensor, support: Tensor) -> Tensor:
        """
        Args:
            x: Noisy query points, shape (B, n, d_data).
            t: Time values, shape (B, n, 1).
            support: Support set, shape (B, m, d_data). Shared across n queries.

        Returns:
            velocity: Predicted velocity, shape (B, n, d_data).
        """
        assert x.dim() == 3, f"x must be 3D (B, n, d), got {x.dim()}D"
        assert x.shape[-1] == self.d_data, f"Expected d={self.d_data}, got {x.shape[-1]}"

        t_emb = self.time_emb(t)

        query = self.query_proj(torch.cat([x, t_emb], dim=-1))
        support_tokens = self.support_proj(support)

        for layer in self.layers:
            query = layer(query, support_tokens)

        if self.use_exact_head:
            exact_vel, _ = self.exact_head(x, t, support)
            exact_proj = self.exact_proj(exact_vel.to(query.dtype))
            combined = torch.cat([query, exact_proj], dim=-1)
            velocity = self.output_mlp(combined)
        else:
            velocity = self.output_mlp(query)

        assert velocity.shape == x.shape, f"Output shape {velocity.shape} != input shape {x.shape}"
        return velocity
