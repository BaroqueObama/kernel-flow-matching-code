"""Exact plug-in OT-FM velocity field (Prop 4.4) and frozen attention realization (Thm 5.2)."""

from torch import Tensor

from src.utils.flow_matching import T_EPS, bandwidth, sigma_t
from src.utils.kernels import nw_local_mean


def exact_plugin_velocity(x_tilde: Tensor, t: Tensor, support: Tensor, sigma_min: float) -> Tensor:
    """Exact empirical OT-FM velocity field (Prop 4.4).

    u_t^S(t * x_tilde) = x_tilde + (m_h(x_tilde; S) - x_tilde) / sigma_t

    Requires t > 0 (the field is undefined at t=0).

    Args:
        x_tilde: De-scaled query points, shape (batch, d).
        t: Time values, shape (batch, 1). Must be > 0.
        support: Support set, shape (batch, m, d) or (m, d).
        sigma_min: Minimum noise level.

    Returns:
        Velocity at x = t * x_tilde, shape (batch, d).
    """
    assert (t > 0).all(), f"Prop 4.4 requires t > 0, got min {t.min()}"
    h_t = bandwidth(t, sigma_min)  # (batch, 1)
    sig_t = sigma_t(t, sigma_min)  # (batch, 1)
    m_h = nw_local_mean(x_tilde, support, h_t)  # (batch, d)
    velocity = x_tilde + (m_h - x_tilde) / sig_t
    return velocity


def exact_plugin_velocity_scaled(x: Tensor, t: Tensor, support: Tensor, sigma_min: float) -> Tensor:
    """Exact plug-in velocity in the original (scaled) coordinates.

    Takes x (not x_tilde) and internally de-scales: x_tilde = x / t.
    At t=0 the de-scaling is singular; t is clamped to T_EPS from below.
    This matches the ODE integration convention of starting from t_min > 0.

    Args:
        x: Query points in scaled coordinates, shape (batch, d).
        t: Time values, shape (batch, 1).
        support: Support set, shape (batch, m, d) or (m, d).
        sigma_min: Minimum noise level.

    Returns:
        Velocity at x, shape (batch, d).
    """
    t = t.clamp(min=T_EPS)
    x_tilde = x / t
    return exact_plugin_velocity(x_tilde, t, support, sigma_min)
