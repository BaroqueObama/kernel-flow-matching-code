"""OT conditional path utilities: noise schedule, bandwidth, and path sampling."""

from torch import Tensor

T_EPS = 1e-5


def sigma_t(t: Tensor, sigma_min: float) -> Tensor:
    """Noise schedule: sigma_t = 1 - (1 - sigma_min) * t."""
    assert (t >= 0).all() and (t <= 1).all(), (
        f"t must be in [0, 1], got range [{t.min()}, {t.max()}]"
    )
    assert 0 < sigma_min <= 1, f"sigma_min must be in (0, 1], got {sigma_min}"
    return 1.0 - (1.0 - sigma_min) * t


def bandwidth(t: Tensor, sigma_min: float) -> Tensor:
    """De-scaled bandwidth: h(t) = sigma_t / t. Requires t > 0."""
    assert (t > 0).all(), f"bandwidth requires t > 0, got min {t.min()}"
    return sigma_t(t, sigma_min) / t


def sample_ot_conditional_path(
    x_0: Tensor, x_1: Tensor, t: Tensor, sigma_min: float
) -> tuple[Tensor, Tensor]:
    """Sample x_t = t*x_1 + sigma_t*x_0 and regression target y_t = x_1 - (1-sigma_min)*x_0."""
    assert x_0.shape == x_1.shape, f"Shape mismatch: x_0 {x_0.shape} vs x_1 {x_1.shape}"
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    assert t.shape == (x_0.shape[0], 1), f"t shape must be (batch, 1), got {t.shape}"

    sig_t = sigma_t(t, sigma_min)
    x_t = t * x_1 + sig_t * x_0
    y_t = x_1 - (1.0 - sigma_min) * x_0
    return x_t, y_t
