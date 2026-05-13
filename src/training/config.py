"""Config dataclasses for ICFM experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    d: int
    support_size: int
    target_size: int
    meta_distribution: str
    meta_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    use_exact_head: bool
    time_embedding: str
    sigma_min: float
    qk_norm: bool = False


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_steps: int
    warmup_steps: int
    grad_clip_norm: float
    time_sampling: str
    mixed_precision: bool
    checkpoint_every: int
    eval_every: int
    log_every: int
    compile: bool = False


@dataclass
class EvalConfig:
    n_eval_tasks: int
    n_generated: int
    ode_steps: int
    t_min: float
    mmd_sigma: float | None
    metrics: list[str]
    ode_method: str = "euler"


@dataclass
class WandbConfig:
    project: str
    entity: str | None = None
    tags: list[str] = field(default_factory=list)
    name: str | None = None


@dataclass
class ICFMConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvalConfig
    wandb: WandbConfig


def load_config(path: str | Path) -> ICFMConfig:
    """Load YAML config into typed dataclass."""
    path = Path(path)
    assert path.exists(), f"Config file not found: {path}"

    with open(path) as f:
        raw = yaml.safe_load(f)

    config = ICFMConfig(
        seed=raw["seed"],
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        evaluation=EvalConfig(**raw["evaluation"]),
        wandb=WandbConfig(**raw["wandb"]),
    )
    validate_config(config)
    return config


def validate_config(config: ICFMConfig) -> None:
    """Fail-loud validation of all config fields."""
    m = config.model
    assert m.sigma_min > 0, f"sigma_min must be positive, got {m.sigma_min}"
    assert m.sigma_min <= 1, f"sigma_min must be <= 1, got {m.sigma_min}"
    assert m.d_model > 0, f"d_model must be positive, got {m.d_model}"
    assert m.d_model % 2 == 0, f"d_model must be even for sinusoidal embedding, got {m.d_model}"
    assert m.n_heads >= 1, f"n_heads must be >= 1, got {m.n_heads}"
    assert m.n_layers >= 1, f"n_layers must be >= 1, got {m.n_layers}"
    assert m.time_embedding in (
        "sinusoidal",
        "learned",
    ), f"time_embedding must be sinusoidal|learned, got {m.time_embedding}"
    if m.use_exact_head:
        assert m.n_heads >= 2, (
            f"need >= 2 heads when use_exact_head=True (1 exact + >=1 learned), got {m.n_heads}"
        )

    d = config.data
    assert d.d > 0, f"d must be positive, got {d.d}"
    assert d.support_size > 0, f"support_size must be positive, got {d.support_size}"
    assert d.target_size > 0, f"target_size must be positive, got {d.target_size}"
    valid_dists = (
        "gaussian_mixture",
        "rings",
        "moons",
        "spirals",
        "spherical_shells",
        "mixed",
        "imagenet",
    )
    assert d.meta_distribution in valid_dists, (
        f"meta_distribution must be one of {valid_dists}, got {d.meta_distribution}"
    )

    t = config.training
    assert t.batch_size > 0, f"batch_size must be positive, got {t.batch_size}"
    assert t.learning_rate > 0, f"learning_rate must be positive, got {t.learning_rate}"
    assert t.max_steps > 0, f"max_steps must be positive, got {t.max_steps}"
    assert t.warmup_steps >= 0, f"warmup_steps must be non-negative, got {t.warmup_steps}"
    assert t.grad_clip_norm > 0, f"grad_clip_norm must be positive, got {t.grad_clip_norm}"
    assert t.time_sampling in (
        "uniform",
        "log_uniform_bandwidth",
    ), f"time_sampling must be uniform|log_uniform_bandwidth, got {t.time_sampling}"

    e = config.evaluation
    assert e.t_min > 0, f"t_min must be positive, got {e.t_min}"
    assert e.n_eval_tasks > 0, f"n_eval_tasks must be positive, got {e.n_eval_tasks}"
    assert e.ode_steps > 0, f"ode_steps must be positive, got {e.ode_steps}"
    assert e.ode_method in (
        "euler",
        "dopri5",
    ), f"ode_method must be euler|dopri5, got {e.ode_method}"


def _coerce_value(current, value_str: str, key: str):
    """Coerce a string value to match the type of the current value."""
    if isinstance(current, bool):
        lower = value_str.lower()
        assert lower in ("true", "1", "yes", "false", "0", "no"), (
            f"Invalid boolean value '{value_str}' for {key}. Use true/false/1/0/yes/no."
        )
        return lower in ("true", "1", "yes")
    elif isinstance(current, int):
        return int(value_str)
    elif isinstance(current, float):
        return float(value_str)
    elif current is None:
        try:
            return float(value_str)
        except ValueError:
            return value_str
    else:
        return value_str


def apply_overrides(config: ICFMConfig, overrides: list[str]) -> ICFMConfig:
    """Apply dotted CLI overrides (field / section.field / section.dict.key)."""
    for override in overrides:
        assert "=" in override, f"Override must be key=value, got: {override}"
        key, value_str = override.split("=", 1)
        parts = key.split(".")

        if len(parts) == 1:
            target, field_name = config, parts[0]
        elif len(parts) == 2:
            section_name, field_name = parts
            target = getattr(config, section_name, None)
            assert target is not None, f"Unknown config section: {section_name}"
        elif len(parts) == 3:
            section_name, field_name, subkey = parts
            section = getattr(config, section_name, None)
            assert section is not None, f"Unknown config section: {section_name}"
            container = getattr(section, field_name, None)
            assert isinstance(container, dict), (
                f"{section_name}.{field_name} is not a dict, cannot index with .{subkey}"
            )
            assert subkey in container, (
                f"Unknown key '{subkey}' in {section_name}.{field_name}. "
                f"Existing keys: {list(container.keys())}"
            )
            current = container[subkey]
            container[subkey] = _coerce_value(current, value_str, key)
            continue
        else:
            raise AssertionError(
                f"Override key must be 'field', 'section.field', or "
                f"'section.dict_field.key', got: {key}"
            )

        assert hasattr(target, field_name), f"Unknown field '{field_name}' in {key}"

        current = getattr(target, field_name)
        setattr(target, field_name, _coerce_value(current, value_str, key))

    validate_config(config)
    return config
