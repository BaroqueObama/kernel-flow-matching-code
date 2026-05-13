"""ICFM training loop with W&B logging, mixed precision, and checkpointing."""

import dataclasses
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from src.data.meta_distributions import META_DISTRIBUTIONS, MixedMetaDistribution
from src.evaluation.metrics import c2st_accuracy, mmd_squared, velocity_mse
from src.training.config import ICFMConfig
from src.utils.flow_matching import T_EPS, sample_ot_conditional_path
from src.utils.ode_solver import ode_sample


class ICFMTrainer:
    def __init__(self, model: nn.Module, config: ICFMConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.step = 0
        self.best_val_metric = float("inf")
        self._best_step: int | None = None
        self._ema_mmd: float | None = None
        self._local_run_id = f"local_{torch.randint(0, 2**31, (1,)).item():08x}"

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        warmup = config.training.warmup_steps

        def lr_lambda(step: int) -> float:
            if warmup == 0:
                return 1.0
            if step < warmup:
                return max(step, 1) / warmup
            return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler = torch.amp.GradScaler(enabled=config.training.mixed_precision)
        self.meta_dist = self._build_meta_distribution()
        self.val_meta_dist = self._build_val_meta_distribution()

    def _build_meta_distribution(self):
        name = self.config.data.meta_distribution
        d = self.config.data.d
        params = self.config.data.meta_params
        if name == "mixed":
            return MixedMetaDistribution(d=d, **params)
        if name == "imagenet":
            from src.data.imagenet_meta import ImageNetMetaDistribution

            return ImageNetMetaDistribution(d=d, **params)
        cls = META_DISTRIBUTIONS[name]
        return cls(d=d, **params)

    def _build_val_meta_distribution(self):
        if self.config.data.meta_distribution != "imagenet":
            return None
        from src.data.imagenet_meta import ImageNetMetaDistribution

        params = dict(self.config.data.meta_params)
        params["split"] = "val"
        return ImageNetMetaDistribution(d=self.config.data.d, **params)

    def _sample_time(self, n: int) -> Tensor:
        if self.config.training.time_sampling == "uniform":
            return torch.rand(n, 1, device=self.device).clamp(min=T_EPS)
        elif self.config.training.time_sampling == "log_uniform_bandwidth":
            log_h_min = torch.log(torch.tensor(self.config.model.sigma_min))
            log_h_max = torch.tensor(5.0)
            log_h = log_h_min + (log_h_max - log_h_min) * torch.rand(n, device=self.device)
            h = torch.exp(log_h)
            t = 1.0 / (h + 1.0 - self.config.model.sigma_min)
            return t.unsqueeze(-1).clamp(min=T_EPS, max=1.0)
        else:
            raise ValueError(f"Unknown time_sampling: {self.config.training.time_sampling}")

    def _sample_batch(self) -> tuple[Tensor, Tensor]:
        return self.meta_dist.sample_batch_tasks(
            self.config.training.batch_size,
            self.config.data.support_size,
            self.config.data.target_size,
            self.device,
        )

    def train_step(self) -> dict[str, float]:
        self.model.train()
        support, target = self._sample_batch()
        B, n, d = target.shape

        x_0_flat = torch.randn(B * n, d, device=self.device)
        x_1_flat = target.reshape(B * n, d)
        t_flat = self._sample_time(B * n)

        x_t_flat, y_t = sample_ot_conditional_path(
            x_0_flat, x_1_flat, t_flat, self.config.model.sigma_min
        )

        x_t = x_t_flat.view(B, n, d)
        t_model = t_flat.view(B, n, 1)

        with torch.amp.autocast("cuda", enabled=self.config.training.mixed_precision):
            v_pred = self.model(x_t, t_model, support)
            loss = ((v_pred.reshape(B * n, d) - y_t) ** 2).sum(dim=-1).mean()

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.training.grad_clip_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.scheduler.step()

        return {
            "train/loss": loss.item(),
            "train/grad_norm": grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm,
            "train/lr": self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        cfg = self.config
        eval_dist = self.val_meta_dist if self.val_meta_dist is not None else self.meta_dist
        n_tasks = cfg.evaluation.n_eval_tasks
        mse_vals, mmd_vals, c2st_vals = [], [], []
        exact_vel_norms, exact_vel_norms_high_t, exact_vel_norms_p95 = [], [], []

        raw_model = getattr(self.model, "_orig_mod", self.model)
        has_exact = hasattr(raw_model, "exact_head") and raw_model.use_exact_head

        for _ in range(n_tasks):
            support, target = eval_dist.sample_task(
                cfg.data.support_size, cfg.evaluation.n_generated, self.device
            )

            n_pts = target.shape[0]
            x_0 = torch.randn_like(target)
            t = torch.rand(n_pts, 1, device=self.device).clamp(min=T_EPS)
            x_t, y_t = sample_ot_conditional_path(x_0, target, t, cfg.model.sigma_min)
            x_t_3d = x_t.unsqueeze(0)
            t_3d = t.unsqueeze(0)
            support_3d = support.unsqueeze(0)
            v_pred = self.model(x_t_3d, t_3d, support_3d).squeeze(0)
            mse_vals.append(velocity_mse(v_pred, y_t).item())

            if has_exact:
                exact_vel, _ = raw_model.exact_head(x_t_3d, t_3d, support_3d)
                norms = exact_vel.norm(dim=-1).squeeze(0)
                exact_vel_norms.append(norms.mean().item())
                exact_vel_norms_p95.append(norms.quantile(0.95).item())
                high_t_mask = t.squeeze(-1) > 0.7
                if high_t_mask.any():
                    exact_vel_norms_high_t.append(norms[high_t_mask].mean().item())

            generated = ode_sample(
                self.model,
                support,
                cfg.evaluation.n_generated,
                cfg.data.d,
                self.device,
                method=cfg.evaluation.ode_method,
                n_steps=cfg.evaluation.ode_steps,
                t_min=cfg.evaluation.t_min,
            )
            mmd_vals.append(mmd_squared(generated, target, cfg.evaluation.mmd_sigma).item())
            c2st_vals.append(c2st_accuracy(generated, target))

        result = {
            "val/velocity_mse": sum(mse_vals) / len(mse_vals),
            "val/mmd": sum(mmd_vals) / len(mmd_vals),
            "val/c2st": sum(c2st_vals) / len(c2st_vals),
        }
        if exact_vel_norms:
            result["val/exact_vel_norm"] = sum(exact_vel_norms) / len(exact_vel_norms)
            result["val/exact_vel_norm_p95"] = sum(exact_vel_norms_p95) / len(exact_vel_norms_p95)
        if exact_vel_norms_high_t:
            result["val/exact_vel_norm_high_t"] = sum(exact_vel_norms_high_t) / len(
                exact_vel_norms_high_t
            )
        return result

    def _update_ema_mmd(self, raw_mmd: float, decay: float = 0.8) -> float:
        if self._ema_mmd is None:
            self._ema_mmd = raw_mmd
        else:
            self._ema_mmd = decay * self._ema_mmd + (1 - decay) * raw_mmd
        return self._ema_mmd

    def _save_checkpoint(self, step: int, metrics: dict) -> None:
        run_id = os.environ.get("WANDB_RUN_ID") or self._local_run_id
        ckpt_dir = Path("checkpoints") / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        val_metric = metrics.get("val/mmd", float("inf"))
        is_best = val_metric < self.best_val_metric
        if is_best:
            self.best_val_metric = val_metric

        raw_model = getattr(self.model, "_orig_mod", self.model)
        state = {
            "model": raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": step,
            "best_val_metric": self.best_val_metric,
            "best_step": self._best_step,
            "ema_mmd": self._ema_mmd,
            "config": dataclasses.asdict(self.config),
        }
        path = ckpt_dir / f"step_{step}.pt"
        torch.save(state, path)

        if is_best:
            self._best_step = step
            tmp = ckpt_dir / "best.pt.tmp"
            shutil.copy(path, tmp)
            tmp.rename(ckpt_dir / "best.pt")

        self._cleanup_checkpoints(ckpt_dir, keep=3)

    def _cleanup_checkpoints(self, ckpt_dir: Path, keep: int = 3) -> None:
        best_step = getattr(self, "_best_step", None)
        step_files = sorted(
            ckpt_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        for old in step_files[:-keep]:
            step_num = int(old.stem.split("_")[1])
            if step_num == best_step:
                continue
            old.unlink()

    def _load_checkpoint(self, path: str) -> int:
        state = torch.load(path, map_location=self.device, weights_only=True)
        raw_model = getattr(self.model, "_orig_mod", self.model)
        result = raw_model.load_state_dict(state["model"], strict=False)
        if result.missing_keys:
            tqdm.write(f"Checkpoint missing keys (initialized fresh): {result.missing_keys}")
        if result.unexpected_keys:
            tqdm.write(f"Checkpoint extra keys (ignored): {result.unexpected_keys}")

        n_current = len(list(raw_model.parameters()))
        n_saved = len(state["optimizer"]["param_groups"][0]["params"])
        if n_current == n_saved:
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
        else:
            tqdm.write(
                f"Optimizer param count changed ({n_saved} -> {n_current}), "
                "reinitializing optimizer/scheduler"
            )

        self.scaler.load_state_dict(state["scaler"])
        self.step = state["step"]
        self.best_val_metric = state.get("best_val_metric", float("inf"))
        self._best_step = state.get("best_step")
        self._ema_mmd = state.get("ema_mmd")
        return self.step

    def train(self) -> None:
        try:
            from dotenv import load_dotenv

            import wandb

            load_dotenv()
        except ImportError:
            wandb = None

        use_wandb = wandb is not None and os.environ.get("WANDB_API_KEY")
        if use_wandb:
            wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                name=self.config.wandb.name,
                config=dataclasses.asdict(self.config),
                tags=self.config.wandb.tags,
            )
            os.environ["WANDB_RUN_ID"] = wandb.run.id

        cfg = self.config.training
        pbar = tqdm(
            range(self.step + 1, cfg.max_steps + 1),
            initial=self.step,
            total=cfg.max_steps,
            desc="Training",
        )

        for step in pbar:
            self.step = step
            metrics = self.train_step()

            if step % cfg.log_every == 0:
                pbar.set_postfix(loss=f"{metrics['train/loss']:.4f}")
                if use_wandb:
                    wandb.log(metrics, step=step)

            if step % cfg.eval_every == 0:
                eval_metrics = self.evaluate()
                ema_mmd = self._update_ema_mmd(eval_metrics["val/mmd"])
                eval_metrics["val/mmd_ema"] = ema_mmd
                if use_wandb:
                    wandb.log(eval_metrics, step=step)
                tqdm.write(
                    f"[Step {step}] val_mse={eval_metrics['val/velocity_mse']:.4f} "
                    f"mmd={eval_metrics['val/mmd']:.4e} "
                    f"mmd_ema={ema_mmd:.4e} c2st={eval_metrics['val/c2st']:.3f}"
                )
                self._save_checkpoint(step, eval_metrics)

            elif step % cfg.checkpoint_every == 0:
                self._save_checkpoint(step, metrics)

        final_metrics = self.evaluate()
        ema_mmd = self._update_ema_mmd(final_metrics["val/mmd"])
        final_metrics["val/mmd_ema"] = ema_mmd
        if use_wandb:
            wandb.log(final_metrics, step=self.step)
        tqdm.write(
            f"[Final] val_mse={final_metrics['val/velocity_mse']:.4f} "
            f"mmd={final_metrics['val/mmd']:.4e} c2st={final_metrics['val/c2st']:.3f}"
        )
        self._save_checkpoint(self.step, final_metrics)
        run_id = os.environ.get("WANDB_RUN_ID") or self._local_run_id
        ckpt_dir = Path("checkpoints") / run_id
        final_path = ckpt_dir / f"step_{self.step}.pt"
        if final_path.exists():
            shutil.copy(final_path, ckpt_dir / "final.pt")

        if use_wandb:
            wandb.finish()
