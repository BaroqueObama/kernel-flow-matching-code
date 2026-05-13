"""Gaussian kernel utilities: NW weights, local mean, KDE."""

import torch
from torch import Tensor


def gaussian_kernel_logits(x: Tensor, s: Tensor, h: Tensor) -> Tensor:
    """Gaussian kernel logits: -||x - s_i||^2 / (2 h^2).

    Args:
        x: Query points, shape (batch, d).
        s: Support set, shape (batch, m, d) or (m, d).
        h: Bandwidth, shape (batch, 1) or scalar.

    Returns:
        Logits, shape (batch, m).
    """
    if s.dim() == 2:
        s = s.unsqueeze(0).expand(x.shape[0], -1, -1)
    assert s.dim() == 3, f"Support must be 3D (batch, m, d), got {s.dim()}D"
    assert x.dim() == 2, f"Query must be 2D (batch, d), got {x.dim()}D"

    diff = x.unsqueeze(1) - s  # (batch, m, d)
    sq_dist = (diff**2).sum(dim=-1)  # (batch, m)

    if isinstance(h, Tensor):
        if h.dim() == 2:
            h_sq = h**2  # (batch, 1)
        else:
            h_sq = h.unsqueeze(-1) ** 2
    else:
        h_sq = h**2

    return -sq_dist / (2.0 * h_sq)


def nw_weights(x: Tensor, s: Tensor, h: Tensor) -> Tensor:
    """NW kernel weights w_i = phi_h(x - s_i) / sum_j phi_h(x - s_j) via softmax.

    Args:
        x: Query points, shape (batch, d).
        s: Support set, shape (batch, m, d) or (m, d).
        h: Bandwidth, shape (batch, 1) or scalar.

    Returns:
        Weights, shape (batch, m), summing to 1 along m.
    """
    logits = gaussian_kernel_logits(x, s, h)
    weights = torch.softmax(logits, dim=-1)
    return weights


def nw_local_mean(x: Tensor, s: Tensor, h: Tensor) -> Tensor:
    """NW local mean: m_h(x; S) = sum_i w_i * s_i.

    Args:
        x: Query points, shape (batch, d).
        s: Support set, shape (batch, m, d) or (m, d).
        h: Bandwidth, shape (batch, 1) or scalar.

    Returns:
        Local mean, shape (batch, d).
    """
    if s.dim() == 2:
        s = s.unsqueeze(0).expand(x.shape[0], -1, -1)
    weights = nw_weights(x, s, h)  # (batch, m)
    return torch.einsum("bm,bmd->bd", weights, s)
