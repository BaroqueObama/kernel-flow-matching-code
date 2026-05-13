"""Evaluate a trained ICFM model or baseline on held-out tasks."""

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data.meta_distributions import META_DISTRIBUTIONS, MixedMetaDistribution
from src.evaluation.metrics import c2st_accuracy, mmd_squared
from src.models.velocity_net import ICFMVelocityNet
from src.training.config import apply_overrides, load_config
from src.utils.ode_solver import ode_sample, ode_sample_exact

_ARCH_KEYS = ("d", "d_model", "n_heads", "n_layers", "use_exact_head", "sigma_min", "qk_norm")
_ARCH_DEFAULTS = {"qk_norm": False}


def _check_checkpoint_config(state: dict, config) -> None:
    """Warn if CLI config disagrees with checkpoint config on architecture."""
    ckpt_cfg = state.get("config")
    if ckpt_cfg is None:
        print("WARNING: checkpoint has no saved config — cannot verify match", file=sys.stderr)
        return

    mismatches = []
    flat_cli = {"d": config.data.d, **{k: getattr(config.model, k) for k in _ARCH_KEYS if k != "d"}}
    flat_ckpt = {"d": ckpt_cfg.get("data", {}).get("d")}
    for k in _ARCH_KEYS:
        if k != "d":
            flat_ckpt[k] = ckpt_cfg.get("model", {}).get(k)
        if flat_ckpt[k] is None:
            if flat_cli[k] != _ARCH_DEFAULTS.get(k):
                mismatches.append(
                    f"  {k}: cli={flat_cli[k]} vs checkpoint=<absent, default={_ARCH_DEFAULTS.get(k)}>"
                )
        elif flat_cli[k] != flat_ckpt[k]:
            mismatches.append(f"  {k}: cli={flat_cli[k]} vs checkpoint={flat_ckpt[k]}")

    if mismatches:
        print(
            "WARNING: CLI config differs from checkpoint config on architectural fields:\n"
            + "\n".join(mismatches),
            file=sys.stderr,
        )


def _checkpoint_run_id(checkpoint_path: str) -> str:
    """Extract run ID from checkpoint path like 'checkpoints/{run_id}/best.pt'."""
    parts = Path(checkpoint_path).parts
    for i, part in enumerate(parts):
        if part == "checkpoints" and i + 1 < len(parts):
            return parts[i + 1]
    return Path(checkpoint_path).stem


def main():
    parser = argparse.ArgumentParser(description="Evaluate ICFM")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=["exact_plugin"],
    )
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    assert args.checkpoint or args.baseline, "Provide --checkpoint or --baseline"

    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    ckpt_config = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        _check_checkpoint_config(state, config)
        ckpt_config = state.get("config")

        model = ICFMVelocityNet(
            d_data=config.data.d,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers,
            use_exact_head=config.model.use_exact_head,
            sigma_min=config.model.sigma_min,
            qk_norm=config.model.qk_norm,
        ).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        print(f"Loaded checkpoint from {args.checkpoint}")

    eval_params = dict(config.data.meta_params)
    split_overridden = False
    if config.data.meta_distribution == "imagenet":
        split = eval_params.get("split", "train")
        explicitly_set = any(
            ov.split("=", 1)[0] == "data.meta_params.split" for ov in args.override
        )
        if split == "train" and not explicitly_set:
            print(
                "WARNING: Overriding meta_params.split='train' -> 'test' for evaluation. "
                "Use --override data.meta_params.split=train to evaluate on training classes.",
                file=sys.stderr,
            )
            eval_params["split"] = "test"
            split_overridden = True

    if config.data.meta_distribution == "mixed":
        meta_dist = MixedMetaDistribution(d=config.data.d, **eval_params)
    elif config.data.meta_distribution == "imagenet":
        from src.data.imagenet_meta import ImageNetMetaDistribution

        meta_dist = ImageNetMetaDistribution(d=config.data.d, **eval_params)
    else:
        cls = META_DISTRIBUTIONS[config.data.meta_distribution]
        meta_dist = cls(d=config.data.d, **eval_params)

    n_tasks = args.n_tasks or config.evaluation.n_eval_tasks
    mmd_vals, c2st_vals = [], []

    for i in tqdm(range(n_tasks), desc="Evaluating"):
        support, target = meta_dist.sample_task(
            config.data.support_size, config.evaluation.n_generated, device
        )

        if model is not None:
            generated = ode_sample(
                model,
                support,
                config.evaluation.n_generated,
                config.data.d,
                device,
                method=config.evaluation.ode_method,
                n_steps=config.evaluation.ode_steps,
                t_min=config.evaluation.t_min,
            )
        else:
            generated = ode_sample_exact(
                support,
                config.evaluation.n_generated,
                config.data.d,
                device,
                sigma_min=config.model.sigma_min,
                method=config.evaluation.ode_method,
                n_steps=config.evaluation.ode_steps,
                t_min=config.evaluation.t_min,
            )

        mmd_vals.append(mmd_squared(generated, target, config.evaluation.mmd_sigma).item())
        c2st_vals.append(c2st_accuracy(generated, target))

    results = {
        "mmd_mean": float(np.mean(mmd_vals)),
        "mmd_std": float(np.std(mmd_vals)),
        "c2st_mean": float(np.mean(c2st_vals)),
        "c2st_std": float(np.std(c2st_vals)),
        "n_tasks": n_tasks,
        "baseline": args.baseline,
        "checkpoint": args.checkpoint,
        "overrides": args.override,
        "eval_config": dataclasses.asdict(config),
        "eval_meta_params": eval_params,
        "split_overridden": split_overridden,
    }
    if ckpt_config is not None:
        results["checkpoint_config"] = ckpt_config

    print(f"\nResults ({n_tasks} tasks):")
    print(f"  MMD:  {results['mmd_mean']:.4e} +/- {results['mmd_std']:.4e}")
    print(f"  C2ST: {results['c2st_mean']:.3f} +/- {results['c2st_std']:.3f}")

    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    config_name = Path(args.config).stem
    override_tag = ""
    for ov in args.override:
        key, val = ov.split("=", 1)
        short_key = key.split(".")[-1]
        override_tag += f"_{short_key}{val}"
    if args.checkpoint:
        label = _checkpoint_run_id(args.checkpoint)
    else:
        label = args.baseline
    out_path = out_dir / f"{config_name}{override_tag}_{label}.json"
    if out_path.exists():
        print(f"WARNING: overwriting existing results at {out_path}", file=sys.stderr)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
