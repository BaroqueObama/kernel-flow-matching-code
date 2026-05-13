"""Train an ICFM velocity network."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from src.models.velocity_net import ICFMVelocityNet
from src.training.config import apply_overrides, load_config
from src.training.trainer import ICFMTrainer


def _build_run_name(config_path: str, overrides: list[str], seed: int) -> str:
    """Build a W&B run name like 'b1_main_r2_m50_s42' from config + overrides."""
    name = Path(config_path).stem
    for ov in overrides:
        key, val = ov.split("=", 1)
        short_key = key.split(".")[-1]
        name += f"_{short_key}{val}"
    name += f"_s{seed}"
    return name


def main():
    parser = argparse.ArgumentParser(description="Train ICFM")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)

    if config.wandb.name is None:
        config.wandb.name = _build_run_name(args.config, args.override, config.seed)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ICFMVelocityNet(
        d_data=config.data.d,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        use_exact_head=config.model.use_exact_head,
        sigma_min=config.model.sigma_min,
        qk_norm=config.model.qk_norm,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    trainer = ICFMTrainer(model, config, device)
    if args.resume:
        step = trainer._load_checkpoint(args.resume)
        print(f"Resumed from step {step}")

    if config.training.compile:
        print("Compiling model with torch.compile...")
        trainer.model = torch.compile(model)
    trainer.train()


if __name__ == "__main__":
    main()
