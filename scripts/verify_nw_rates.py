"""Verify NW minimax rates (alpha = 4/(4+d)) on GMM and random Fourier targets."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import mmd_squared
from src.baselines.plugin_velocity import exact_plugin_velocity_scaled


class SimpleGMM:
    """A fixed random GMM for controlled rate experiments."""

    def __init__(self, d: int, k: int = 5, mean_scale: float = 4.0, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.d = d
        self.k = k
        self.weights = torch.from_numpy(np.ones(k) / k).float()
        self.means = torch.from_numpy(rng.randn(k, d) * mean_scale).float()
        factors = rng.randn(k, d, d) * 0.5
        self.covs = torch.from_numpy(
            np.array([f @ f.T + 0.1 * np.eye(d) for f in factors])
        ).float()
        self.chol = torch.linalg.cholesky(self.covs)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        comp = torch.multinomial(self.weights, n, replacement=True)
        z = torch.randn(n, self.d)
        samples = self.means[comp] + torch.einsum("nij,nj->ni", self.chol[comp], z)
        return samples.to(device)


class RandomFourierDensity:
    """Non-parametric target via rejection sampling from a random Fourier potential.

    Amplitude scaled by 1/sqrt(n_modes) to keep rejection sampling feasible.
    """

    def __init__(self, d: int, n_modes: int = 30, freq_scale: float = 1.0,
                 amplitude: float = 0.5, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.d = d
        self.n_modes = n_modes
        self.omega = torch.from_numpy(rng.randn(n_modes, d) * freq_scale).float()
        self.phi = torch.from_numpy(rng.uniform(0, 2 * np.pi, n_modes)).float()
        raw_a = rng.uniform(0.3, 1.0, n_modes)
        self.a = torch.from_numpy(raw_a * amplitude / np.sqrt(n_modes)).float()
        self.bound = 3.5

    @torch.no_grad()
    def sample(self, n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        omega = self.omega.to(device)
        phi = self.phi.to(device)
        a = self.a.to(device)

        samples = []
        remaining = n
        total_proposed = 0
        total_accepted = 0

        for _ in range(1000):
            if remaining <= 0:
                break
            batch = max(remaining * 30, 5000)
            proposal = torch.rand(batch, self.d, device=device) * 2 * self.bound - self.bound
            proj = proposal @ omega.T + phi.unsqueeze(0)
            log_p = -(a.unsqueeze(0) * torch.cos(proj)).sum(dim=-1)
            log_p = log_p - log_p.max()
            accept = torch.rand(batch, device=device) < torch.exp(log_p)
            total_proposed += batch
            total_accepted += accept.sum().item()
            if accept.any():
                samples.append(proposal[accept])
                remaining -= accept.sum().item()

        if len(samples) == 0:
            print(f"  WARNING: Fourier rejection sampling failed at d={self.d}, "
                  f"returning uniform samples")
            return torch.rand(n, self.d, device=device) * 2 * self.bound - self.bound

        result = torch.cat(samples, dim=0)[:n]
        if result.shape[0] < n:
            idx = torch.randint(result.shape[0], (n - result.shape[0],))
            result = torch.cat([result, result[idx]], dim=0)
        return result


@torch.no_grad()
def euler_ode_plugin(support, n_samples, d, sigma_min, n_steps, device):
    """Euler ODE integration of the plug-in velocity field."""
    t_min = 1e-4
    x = torch.randn(n_samples, d, device=device)
    dt = (1.0 - t_min) / n_steps
    support_exp = support.unsqueeze(0).expand(n_samples, -1, -1)

    t_val = t_min
    for _ in range(n_steps):
        t_batch = torch.full((n_samples, 1), t_val, device=device)
        v = exact_plugin_velocity_scaled(x, t_batch, support_exp, sigma_min)
        x = x + v * dt
        t_val += dt

    return x


def run_rate_experiment(
    dist,
    dist_name: str,
    d: int,
    support_sizes: list[int],
    n_tasks: int,
    n_gen: int,
    n_ref: int,
    n_steps: int,
    device: torch.device,
    bandwidth_mode: str = "adapted",
    base_seed: int = 42,
):
    """Run plug-in ODE at each support size, compute MMD."""
    results = []

    for m in support_sizes:
        if bandwidth_mode == "adapted":
            sigma_min = 0.5 * m ** (-1.0 / (4 + d))
            sigma_min = max(sigma_min, 0.002)
        else:
            sigma_min = 0.01

        mmds = []
        n_diverged = 0
        for task_i in range(n_tasks):
            torch.manual_seed(base_seed + task_i * 7919 + m * 31)
            np.random.seed(base_seed + task_i * 7919 + m * 31)

            support = dist.sample(m, device=device)
            reference = dist.sample(n_ref, device=device)

            generated = euler_ode_plugin(
                support=support,
                n_samples=n_gen,
                d=d,
                sigma_min=sigma_min,
                n_steps=n_steps,
                device=device,
            )

            if not torch.isfinite(generated).all():
                n_diverged += 1
                continue
            mmd_val = mmd_squared(generated, reference).sqrt().item()
            if np.isfinite(mmd_val):
                mmds.append(mmd_val)
            else:
                n_diverged += 1

        n_valid = len(mmds)
        if n_valid == 0:
            print(f"  {dist_name} d={d} m={m:4d}  σ_min={sigma_min:.4f}  "
                  f"ALL DIVERGED ({n_diverged}/{n_tasks})")
            continue

        mean_mmd = float(np.mean(mmds))
        std_mmd = float(np.std(mmds, ddof=1)) if n_valid > 1 else 0.0
        results.append({
            "dist": dist_name,
            "d": d,
            "m": m,
            "sigma_min": round(sigma_min, 6),
            "bandwidth_mode": bandwidth_mode,
            "mmd_mean": mean_mmd,
            "mmd_std": std_mmd,
            "n_valid": n_valid,
            "n_diverged": n_diverged,
        })
        div_str = f" ({n_diverged} div)" if n_diverged > 0 else ""
        print(f"  {dist_name} d={d} m={m:4d}  σ_min={sigma_min:.4f}  "
              f"MMD={mean_mmd:.5f} ± {std_mmd:.5f}  [{n_valid}/{n_tasks}]{div_str}")

    return results


def fit_power_law(ms, mmds):
    """Fit MMD ~ m^{-alpha} via log-log OLS. Returns (alpha, R-squared)."""
    log_m = np.log(np.array(ms, dtype=float))
    log_mmd = np.log(np.array(mmds, dtype=float))
    mask = np.isfinite(log_mmd) & np.isfinite(log_m)
    if mask.sum() < 3:
        return float("nan"), 0.0
    A = np.vstack([log_m[mask], np.ones(mask.sum())]).T
    coef, _, _, _ = np.linalg.lstsq(A, log_mmd[mask], rcond=None)
    slope = coef[0]
    residuals = log_mmd[mask] - A @ coef
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_mmd[mask] - log_mmd[mask].mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return -slope, r2


def main():
    parser = argparse.ArgumentParser(description="Verify NW minimax rates for plug-in velocity")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-tasks", type=int, default=30,
                        help="Tasks per (dist, d, m, mode) combination")
    parser.add_argument("--n-gen", type=int, default=500,
                        help="Generated samples per task")
    parser.add_argument("--n-ref", type=int, default=2000,
                        help="Reference samples per task")
    parser.add_argument("--n-steps", type=int, default=100,
                        help="Euler ODE steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)

    dims = [2, 4, 8]
    support_sizes = [10, 25, 50, 100, 250, 500]

    all_results = []
    summary = []

    for d in dims:
        alpha_nw = 4.0 / (4.0 + d)
        print(f"\n{'='*70}")
        print(f"d = {d},  NW minimax rate α_NW = {alpha_nw:.3f}")
        print(f"{'='*70}")

        dists = [
            (SimpleGMM(d=d, k=5, seed=args.seed), "GMM-5"),
            (SimpleGMM(d=d, k=50, mean_scale=5.0, seed=args.seed + 1), "GMM-50"),
            (RandomFourierDensity(d=d, n_modes=max(30, 5 * d),
                                  freq_scale=1.0, amplitude=0.5,
                                  seed=args.seed + 2), "Fourier"),
        ]

        for dist, name in dists:
            for mode in ["adapted", "fixed"]:
                print(f"\n--- {name}, σ_min={mode} ---")
                res = run_rate_experiment(
                    dist=dist,
                    dist_name=name,
                    d=d,
                    support_sizes=support_sizes,
                    n_tasks=args.n_tasks,
                    n_gen=args.n_gen,
                    n_ref=args.n_ref,
                    n_steps=args.n_steps,
                    device=device,
                    bandwidth_mode=mode,
                    base_seed=args.seed,
                )
                all_results.extend(res)

                if len(res) >= 3:
                    ms = [r["m"] for r in res]
                    mmds = [r["mmd_mean"] for r in res]
                    alpha, r2 = fit_power_law(ms, mmds)
                else:
                    alpha, r2 = float("nan"), 0.0

                ratio = alpha / alpha_nw if np.isfinite(alpha) and alpha_nw > 0 else float("nan")
                row = {
                    "dist": name, "d": d, "bandwidth": mode,
                    "alpha": round(alpha, 3) if np.isfinite(alpha) else None,
                    "r2": round(r2, 3),
                    "alpha_nw": round(alpha_nw, 3),
                    "n_points": len(res),
                }
                summary.append(row)
                if np.isfinite(alpha):
                    print(f"  → α = {alpha:.3f} (R² = {r2:.3f}), "
                          f"α_NW = {alpha_nw:.3f}, ratio = {ratio:.2f}×")
                else:
                    print(f"  → insufficient data for fit")

    # Save
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "nw_rate_verification.json", "w") as f:
        json.dump({"results": all_results, "summary": summary}, f, indent=2)
    print(f"\nSaved to {out_dir / 'nw_rate_verification.json'}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Plug-in scaling rate α vs NW minimax rate α_NW = 4/(4+d)")
    print(f"{'='*70}")
    print(f"{'Dist':<10} {'d':>3} {'σ_min':<10} {'α_obs':>7} {'α_NW':>7} {'Ratio':>7} {'R²':>6} {'pts':>4}")
    print("-" * 62)
    for s in summary:
        alpha = s["alpha"]
        if alpha is not None:
            ratio = alpha / s["alpha_nw"]
            print(f"{s['dist']:<10} {s['d']:>3} {s['bandwidth']:<10} "
                  f"{alpha:>7.3f} {s['alpha_nw']:>7.3f} {ratio:>6.2f}× {s['r2']:>6.3f} {s['n_points']:>4}")
        else:
            print(f"{s['dist']:<10} {s['d']:>3} {s['bandwidth']:<10} "
                  f"{'N/A':>7} {s['alpha_nw']:>7.3f} {'---':>7} {'---':>6} {s['n_points']:>4}")


if __name__ == "__main__":
    main()
