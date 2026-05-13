"""Tests for ICFM model components."""

import math

import torch

from src.baselines.plugin_velocity import exact_plugin_velocity_scaled
from src.models.attention import LearnedCrossAttention
from src.models.exact_head import ExactAttentionHead
from src.models.time_embedding import SinusoidalTimeEmbedding
from src.models.velocity_net import ICFMVelocityNet

SIGMA_MIN = 0.01
ATOL = 1e-5


class TestExactAttentionHead:
    def test_matches_plugin_velocity(self):
        B, n, d, m = 4, 8, 3, 20
        head = ExactAttentionHead(SIGMA_MIN)
        x = torch.randn(B, n, d)
        t = torch.rand(B, n, 1) * 0.9 + 0.05
        s = torch.randn(B, m, d)
        v_head, _ = head(x, t, s)
        x_flat = x.reshape(B * n, d)
        t_flat = t.reshape(B * n, 1)
        s_flat = s.unsqueeze(1).expand(B, n, m, d).reshape(B * n, m, d)
        v_ref = exact_plugin_velocity_scaled(x_flat, t_flat, s_flat, SIGMA_MIN)
        v_ref = v_ref.view(B, n, d)
        assert torch.allclose(v_head, v_ref, atol=ATOL), (
            f"Max error: {(v_head - v_ref).abs().max()}"
        )

    def test_zero_parameters(self):
        head = ExactAttentionHead(SIGMA_MIN)
        assert sum(p.numel() for p in head.parameters()) == 0

    def test_output_shapes(self):
        head = ExactAttentionHead(SIGMA_MIN)
        x = torch.randn(4, 6, 2)
        t = torch.rand(4, 6, 1) * 0.9 + 0.05
        s = torch.randn(4, 10, 2)
        vel, nw_mean = head(x, t, s)
        assert vel.shape == (4, 6, 2)
        assert nw_mean.shape == (4, 6, 2)

    def test_various_time_values(self):
        head = ExactAttentionHead(SIGMA_MIN)
        B, n, d, m = 4, 5, 2, 15
        s = torch.randn(B, m, d)
        for t_val in [0.01, 0.1, 0.5, 0.9, 0.99]:
            x = torch.randn(B, n, d)
            t = torch.full((B, n, 1), t_val)
            v_head, _ = head(x, t, s)
            x_flat = x.reshape(B * n, d)
            t_flat = t.reshape(B * n, 1)
            s_flat = s.unsqueeze(1).expand(B, n, m, d).reshape(B * n, m, d)
            v_ref = exact_plugin_velocity_scaled(x_flat, t_flat, s_flat, SIGMA_MIN)
            assert torch.allclose(v_head, v_ref.view(B, n, d), atol=ATOL), (
                f"Failed at t={t_val}, max error: {(v_head - v_ref.view(B, n, d)).abs().max()}"
            )


class TestSinusoidalTimeEmbedding:
    def test_output_shape_1d(self):
        emb = SinusoidalTimeEmbedding(128)
        t = torch.rand(16)
        out = emb(t)
        assert out.shape == (16, 128)

    def test_output_shape_2d(self):
        emb = SinusoidalTimeEmbedding(64)
        t = torch.rand(4, 8)
        out = emb(t)
        assert out.shape == (4, 8, 64)

    def test_squeezes_trailing_one(self):
        emb = SinusoidalTimeEmbedding(64)
        t = torch.rand(4, 8, 1)
        out = emb(t)
        assert out.shape == (4, 8, 64)

    def test_deterministic(self):
        emb = SinusoidalTimeEmbedding(64)
        t = torch.tensor([0.3, 0.7])
        assert torch.equal(emb(t), emb(t))


class TestLearnedCrossAttention:
    def test_output_shape(self):
        attn = LearnedCrossAttention(128, 4)
        q = torch.randn(8, 12, 128)
        kv = torch.randn(8, 50, 128)
        out = attn(q, kv)
        assert out.shape == (8, 12, 128)

    def test_gradient_flow(self):
        attn = LearnedCrossAttention(64, 2)
        q = torch.randn(4, 6, 64, requires_grad=True)
        kv = torch.randn(4, 10, 64)
        out = attn(q, kv)
        out.sum().backward()
        assert q.grad is not None
        for p in attn.parameters():
            assert p.grad is not None

    def test_qk_norm_output_shape(self):
        attn = LearnedCrossAttention(128, 4, qk_norm=True)
        q = torch.randn(8, 12, 128)
        kv = torch.randn(8, 50, 128)
        out = attn(q, kv)
        assert out.shape == (8, 12, 128)
        assert torch.isfinite(out).all()

    def test_qk_norm_kappa_per_head(self):
        attn = LearnedCrossAttention(128, 4, qk_norm=True)
        assert attn.kappa.shape == (4,), f"Expected per-head kappa (4,), got {attn.kappa.shape}"
        d_k = 128 // 4
        expected = math.sqrt(d_k)
        assert torch.allclose(attn.kappa, torch.full((4,), expected)), (
            f"Expected kappa init sqrt(d_k)={expected}, got {attn.kappa}"
        )

    def test_qk_norm_gradient_flow(self):
        attn = LearnedCrossAttention(64, 2, qk_norm=True)
        q = torch.randn(4, 6, 64, requires_grad=True)
        kv = torch.randn(4, 10, 64)
        out = attn(q, kv)
        out.sum().backward()
        assert q.grad is not None
        assert attn.kappa.grad is not None, "kappa has no gradient"
        for p in attn.parameters():
            assert p.grad is not None

    def test_no_kappa_without_qk_norm(self):
        attn = LearnedCrossAttention(64, 2, qk_norm=False)
        assert not hasattr(attn, "kappa")


class TestICFMVelocityNet:
    def test_forward_shape_with_exact_head(self):
        model = ICFMVelocityNet(2, 64, 4, 2, use_exact_head=True, sigma_min=SIGMA_MIN)
        x = torch.randn(4, 8, 2)
        t = torch.rand(4, 8, 1)
        s = torch.randn(4, 10, 2)
        v = model(x, t, s)
        assert v.shape == (4, 8, 2)

    def test_forward_shape_without_exact_head(self):
        model = ICFMVelocityNet(2, 64, 4, 2, use_exact_head=False, sigma_min=SIGMA_MIN)
        x = torch.randn(4, 8, 2)
        t = torch.rand(4, 8, 1)
        s = torch.randn(4, 10, 2)
        v = model(x, t, s)
        assert v.shape == (4, 8, 2)

    def test_differentiable(self):
        model = ICFMVelocityNet(2, 64, 4, 2, use_exact_head=True, sigma_min=SIGMA_MIN)
        x = torch.randn(2, 4, 2)
        t = torch.rand(2, 4, 1)
        s = torch.randn(2, 10, 2)
        v = model(x, t, s)
        loss = (v**2).sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_higher_dimension(self):
        model = ICFMVelocityNet(4, 64, 4, 2, use_exact_head=True, sigma_min=SIGMA_MIN)
        x = torch.randn(2, 6, 4)
        t = torch.rand(2, 6, 1)
        s = torch.randn(2, 20, 4)
        v = model(x, t, s)
        assert v.shape == (2, 6, 4)

    def test_qk_norm_forward(self):
        model = ICFMVelocityNet(2, 64, 4, 2, use_exact_head=False, sigma_min=SIGMA_MIN, qk_norm=True)
        x = torch.randn(4, 8, 2)
        t = torch.rand(4, 8, 1)
        s = torch.randn(4, 10, 2)
        v = model(x, t, s)
        assert v.shape == (4, 8, 2)
        assert torch.isfinite(v).all()

    def test_qk_norm_differentiable(self):
        model = ICFMVelocityNet(2, 64, 4, 2, use_exact_head=False, sigma_min=SIGMA_MIN, qk_norm=True)
        x = torch.randn(2, 4, 2)
        t = torch.rand(2, 4, 1)
        s = torch.randn(2, 10, 2)
        v = model(x, t, s)
        (v**2).sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_qk_norm_with_exact_head(self):
        model = ICFMVelocityNet(2, 64, 4, 2, use_exact_head=True, sigma_min=SIGMA_MIN, qk_norm=True)
        x = torch.randn(4, 8, 2)
        t = torch.rand(4, 8, 1)
        s = torch.randn(4, 10, 2)
        v = model(x, t, s)
        assert v.shape == (4, 8, 2)
        assert torch.isfinite(v).all()
        (v**2).sum().backward()
        kappa_params = [p for n, p in model.named_parameters() if "kappa" in n]
        assert len(kappa_params) > 0, "No kappa parameters found"
        for p in kappa_params:
            assert p.grad is not None, "kappa has no gradient with exact head"

    def test_single_query_per_task(self):
        model = ICFMVelocityNet(2, 64, 4, 2, use_exact_head=True, sigma_min=SIGMA_MIN)
        x = torch.randn(8, 1, 2)
        t = torch.rand(8, 1, 1)
        s = torch.randn(8, 10, 2)
        v = model(x, t, s)
        assert v.shape == (8, 1, 2)
