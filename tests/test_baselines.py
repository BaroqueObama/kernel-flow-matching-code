"""Tests for exact plug-in velocity and kernel utilities."""

import pytest
import torch

from src.baselines.plugin_velocity import exact_plugin_velocity, exact_plugin_velocity_scaled
from src.data.meta_distributions import MixedMetaDistribution
from src.utils.flow_matching import bandwidth, sample_ot_conditional_path, sigma_t
from src.utils.kernels import nw_local_mean, nw_weights

SIGMA_MIN = 0.01
ATOL = 1e-5


class TestFlowMatchingUtils:
    def test_sigma_t_boundaries(self):
        t = torch.tensor([1.0])
        assert torch.allclose(sigma_t(t, SIGMA_MIN), torch.tensor([SIGMA_MIN]))

    def test_sigma_t_near_zero(self):
        t = torch.tensor([0.001])
        sig = sigma_t(t, SIGMA_MIN)
        assert sig.item() > 0.99

    def test_bandwidth_decreasing(self):
        t = torch.linspace(0.01, 1.0, 100)
        h = bandwidth(t, SIGMA_MIN)
        assert (h[1:] < h[:-1]).all(), "Bandwidth must be strictly decreasing"

    def test_bandwidth_at_t1(self):
        t = torch.tensor([1.0])
        h = bandwidth(t, SIGMA_MIN)
        assert torch.allclose(h, torch.tensor([SIGMA_MIN]))

    def test_sample_path_shapes(self):
        batch, d = 32, 4
        x_0 = torch.randn(batch, d)
        x_1 = torch.randn(batch, d)
        t = torch.rand(batch, 1) * 0.98 + 0.01  # in (0.01, 0.99)
        x_t, y_t = sample_ot_conditional_path(x_0, x_1, t, SIGMA_MIN)
        assert x_t.shape == (batch, d)
        assert y_t.shape == (batch, d)


class TestKernels:
    def test_nw_weights_sum_to_one(self):
        batch, m, d = 16, 10, 3
        x = torch.randn(batch, d)
        s = torch.randn(batch, m, d)
        h = torch.ones(batch, 1)
        w = nw_weights(x, s, h)
        assert w.shape == (batch, m)
        assert torch.allclose(w.sum(dim=-1), torch.ones(batch), atol=ATOL)

    def test_nw_mean_is_convex_combination(self):
        batch, m, d = 8, 5, 2
        x = torch.randn(batch, d)
        s = torch.randn(batch, m, d)
        h = torch.ones(batch, 1) * 0.5
        mean = nw_local_mean(x, s, h)
        assert mean.shape == (batch, d)
        s_min = s.min(dim=1).values
        s_max = s.max(dim=1).values
        assert (mean >= s_min - ATOL).all() and (mean <= s_max + ATOL).all()


class TestPluginVelocity:
    def test_scaled_unscaled_consistency(self):
        """exact_plugin_velocity_scaled(x, t, S) should match exact_plugin_velocity(x/t, t, S)."""
        batch, m, d = 8, 10, 3
        support = torch.randn(batch, m, d)
        t = torch.rand(batch, 1) * 0.9 + 0.05  # in (0.05, 0.95)
        x_tilde = torch.randn(batch, d)
        x = t * x_tilde

        v1 = exact_plugin_velocity(x_tilde, t, support, SIGMA_MIN)
        v2 = exact_plugin_velocity_scaled(x, t, support, SIGMA_MIN)
        assert torch.allclose(v1, v2, atol=ATOL), f"Max error: {(v1 - v2).abs().max()}"

    def test_output_shape(self):
        batch, m, d = 4, 20, 5
        support = torch.randn(batch, m, d)
        t = torch.ones(batch, 1) * 0.5
        x_tilde = torch.randn(batch, d)
        v = exact_plugin_velocity(x_tilde, t, support, SIGMA_MIN)
        assert v.shape == (batch, d)

    def test_scaled_handles_t_zero(self):
        """exact_plugin_velocity_scaled should handle t=0 by clamping to T_EPS."""
        batch, m, d = 4, 10, 2
        support = torch.randn(batch, m, d)
        t = torch.zeros(batch, 1)
        x = torch.randn(batch, d)
        v = exact_plugin_velocity_scaled(x, t, support, SIGMA_MIN)
        assert v.shape == (batch, d)
        assert torch.isfinite(v).all()

    def test_exact_velocity_rejects_t_zero(self):
        """exact_plugin_velocity (de-scaled API) should reject t=0."""
        batch, m, d = 4, 10, 2
        support = torch.randn(batch, m, d)
        t = torch.zeros(batch, 1)
        x_tilde = torch.randn(batch, d)
        with pytest.raises(AssertionError):
            exact_plugin_velocity(x_tilde, t, support, SIGMA_MIN)


class TestMixedMetaDistribution:
    def test_mixed_with_gmm_kwargs(self):
        """MixedMetaDistribution should not crash when given GMM-specific kwargs."""
        dist = MixedMetaDistribution(
            d=2, min_components=2, max_components=5, mean_scale=3.0, cov_scale=0.5
        )
        assert len(dist.distributions) > 1
        support, target = dist.sample_task(m=20, n=10)
        assert support.shape == (20, 2)
        assert target.shape == (10, 2)

    def test_mixed_higher_dim_falls_back_to_gmm(self):
        dist = MixedMetaDistribution(d=4)
        assert len(dist.distributions) == 1
        support, target = dist.sample_task(m=10, n=5)
        assert support.shape == (10, 4)
