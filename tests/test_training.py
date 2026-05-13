"""Tests for config loading, training step, and ODE solver."""

import pytest
import torch

from src.models.velocity_net import ICFMVelocityNet
from src.training.config import (
    DataConfig,
    EvalConfig,
    ICFMConfig,
    ModelConfig,
    TrainingConfig,
    WandbConfig,
    load_config,
    validate_config,
)
from src.training.trainer import ICFMTrainer
from src.utils.ode_solver import ode_sample, ode_sample_exact

SIGMA_MIN = 0.01


def _tiny_config() -> ICFMConfig:
    return ICFMConfig(
        seed=0,
        data=DataConfig(
            d=2,
            support_size=5,
            target_size=8,
            meta_distribution="gaussian_mixture",
            meta_params={
                "min_components": 2,
                "max_components": 3,
                "mean_scale": 2.0,
                "cov_scale": 0.5,
            },
        ),
        model=ModelConfig(
            d_model=16,
            n_heads=2,
            n_layers=1,
            use_exact_head=True,
            time_embedding="sinusoidal",
            sigma_min=SIGMA_MIN,
        ),
        training=TrainingConfig(
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_steps=10,
            warmup_steps=2,
            grad_clip_norm=1.0,
            time_sampling="uniform",
            mixed_precision=False,
            checkpoint_every=100,
            eval_every=100,
            log_every=1,
        ),
        evaluation=EvalConfig(
            n_eval_tasks=2,
            n_generated=20,
            ode_steps=10,
            t_min=1e-5,
            mmd_sigma=None,
            metrics=["velocity_mse", "mmd", "c2st"],
            ode_method="euler",
        ),
        wandb=WandbConfig(project="test", entity=None, tags=[]),
    )


class TestConfig:
    def test_load_base_config(self):
        c = load_config("configs/base.yaml")
        assert c.model.d_model == 128
        assert c.model.sigma_min == SIGMA_MIN
        assert c.data.d == 2

    def test_validation_catches_bad_sigma_min(self):
        c = _tiny_config()
        c.model.sigma_min = 0.0
        with pytest.raises(AssertionError, match="sigma_min must be positive"):
            validate_config(c)

    def test_validation_catches_bad_time_embedding(self):
        c = _tiny_config()
        c.model.time_embedding = "invalid"
        with pytest.raises(AssertionError, match="time_embedding"):
            validate_config(c)

    def test_validation_catches_too_few_heads(self):
        c = _tiny_config()
        c.model.use_exact_head = True
        c.model.n_heads = 1
        with pytest.raises(AssertionError, match="need >= 2 heads"):
            validate_config(c)


class TestTrainer:
    def test_single_step_finite_loss(self):
        config = _tiny_config()
        device = torch.device("cpu")
        model = ICFMVelocityNet(
            d_data=config.data.d,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers,
            use_exact_head=config.model.use_exact_head,
            sigma_min=config.model.sigma_min,
        ).to(device)
        trainer = ICFMTrainer(model, config, device)
        metrics = trainer.train_step()
        loss = metrics["train/loss"]
        assert torch.isfinite(torch.tensor(loss)), f"Loss not finite: {loss}"
        assert metrics["train/loss"] > 0


class TestODESolver:
    def test_exact_solver_finite(self):
        s = torch.randn(10, 2)
        samples = ode_sample_exact(s, 20, 2, torch.device("cpu"), SIGMA_MIN, n_steps=20)
        assert samples.shape == (20, 2)
        assert torch.isfinite(samples).all()

    def test_model_solver_finite(self):
        model = ICFMVelocityNet(2, 16, 2, 1, use_exact_head=False, sigma_min=SIGMA_MIN)
        s = torch.randn(5, 2)
        samples = ode_sample(model, s, 10, 2, torch.device("cpu"), n_steps=10)
        assert samples.shape == (10, 2)
        assert torch.isfinite(samples).all()
