"""Verify NW velocity MSE scaling via direct measurement at adapted and fixed bandwidths."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.kernels import nw_local_mean, nw_weights


class SimpleGMM:
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
    def __init__(self, d: int, n_modes: int = 30, freq_scale: float = 1.0,
                 amplitude: float = 0.5, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.d = d
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
        for _ in range(1000):
            if remaining <= 0:
                break
            batch = max(remaining * 30, 5000)
            proposal = torch.rand(batch, self.d, device=device) * 2 * self.bound - self.bound
            proj = proposal @ omega.T + phi.unsqueeze(0)
            log_p = -(a.unsqueeze(0) * torch.cos(proj)).sum(dim=-1)
            log_p = log_p - log_p.max()
            accept = torch.rand(batch, device=device) < torch.exp(log_p)
            if accept.any():
                samples.append(proposal[accept])
                remaining -= accept.sum().item()
        if len(samples) == 0:
            return torch.rand(n, self.d, device=device) * 2 * self.bound - self.bound
        result = torch.cat(samples, dim=0)[:n]
        if result.shape[0] < n:
            idx = torch.randint(result.shape[0], (n - result.shape[0],))
            result = torch.cat([result, result[idx]], dim=0)
        return result


def h_to_t(h: float, sigma_min: float = 0.01) -> float:
    """Solve h(t) = sigma_t / t for t. Returns t in (0, 1]."""
    # h = (1 - (1-sigma_min)*t) / t  =>  t = 1 / (h + 1 - sigma_min)
    t = 1.0 / (h + 1.0 - sigma_min)
    return min(max(t, 1e-5), 1.0)


def sigma_t_val(t: float, sigma_min: float = 0.01) -> float:
    return 1.0 - (1.0 - sigma_min) * t


@torch.no_grad()
def measure_nw_mse(
    dist,
    d: int,
    m: int,
    h: float,
    n_tasks: int,
    n_query: int,
    m_large: int,
    device: torch.device,
    base_seed: int = 42,
):
    """Measure NW estimator MSE at bandwidth h with m support points."""
    nw_mses = []
    neffs = []

    for task_i in range(n_tasks):
        torch.manual_seed(base_seed + task_i * 7919 + m * 31 + int(h * 1e6))

        S = dist.sample(m, device=device)
        S_large = dist.sample(m_large, device=device)

        # Query from de-scaled marginal: x_tilde = X_1 + h*Z
        X1 = dist.sample(n_query, device=device)
        Z = torch.randn(n_query, d, device=device)
        x_tilde = X1 + h * Z

        h_tensor = torch.tensor(h, device=device)
        m_h_S = nw_local_mean(x_tilde, S, h_tensor)
        m_h_large = nw_local_mean(x_tilde, S_large, h_tensor)

        sq_err = ((m_h_S - m_h_large) ** 2).sum(dim=-1)
        mse = sq_err.mean().item()

        if not np.isfinite(mse):
            continue

        w = nw_weights(x_tilde, S, h_tensor)
        n_eff = (1.0 / (w ** 2).sum(dim=-1)).mean().item()

        nw_mses.append(mse)
        neffs.append(n_eff)

    if len(nw_mses) == 0:
        return None

    return {
        "nw_mse_mean": float(np.mean(nw_mses)),
        "nw_mse_std": float(np.std(nw_mses, ddof=1)) if len(nw_mses) > 1 else 0.0,
        "neff_mean": float(np.mean(neffs)),
        "n_valid": len(nw_mses),
    }


def fit_power_law(ms, vals):
    log_m = np.log(np.array(ms, dtype=float))
    log_v = np.log(np.array(vals, dtype=float))
    mask = np.isfinite(log_v) & np.isfinite(log_m)
    if mask.sum() < 3:
        return float("nan"), 0.0
    A = np.vstack([log_m[mask], np.ones(mask.sum())]).T
    coef, _, _, _ = np.linalg.lstsq(A, log_v[mask], rcond=None)
    slope = coef[0]
    residuals = log_v[mask] - A @ coef
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_v[mask] - log_v[mask].mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return -slope, r2


def main():
    parser = argparse.ArgumentParser(description="Verify NW minimax rates via velocity MSE")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-tasks", type=int, default=30)
    parser.add_argument("--n-query", type=int, default=1000)
    parser.add_argument("--m-large", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma-min", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device(args.device)

    dims = [2, 4, 8, 16]
    support_sizes = [10, 25, 50, 100, 250, 500, 1000]
    bw_constant = 1.0

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
        ]
        if d <= 8:
            dists.append((
                RandomFourierDensity(d=d, n_modes=max(30, 5*d), seed=args.seed + 2),
                "Fourier",
            ))

        for dist, name in dists:
            for mode in ["adapted", "fixed"]:
                print(f"\n--- {name}, bandwidth={mode} ---")
                ms_used = []
                nw_mses = []
                vel_mses = []
                neffs_list = []

                for m in support_sizes:
                    if mode == "adapted":
                        h = bw_constant * m ** (-1.0 / (4 + d))
                    else:
                        h = 0.3

                    t = h_to_t(h, args.sigma_min)
                    sig_t = sigma_t_val(t, args.sigma_min)

                    result = measure_nw_mse(
                        dist=dist, d=d, m=m, h=h,
                        n_tasks=args.n_tasks,
                        n_query=args.n_query,
                        m_large=args.m_large,
                        device=device,
                        base_seed=args.seed,
                    )

                    if result is None:
                        print(f"  {name} d={d} m={m:5d}  h={h:.4f}  ALL FAILED")
                        continue

                    nw_mse = result["nw_mse_mean"]
                    vel_mse = nw_mse / (sig_t ** 2)
                    n_eff = result["neff_mean"]

                    ms_used.append(m)
                    nw_mses.append(nw_mse)
                    vel_mses.append(vel_mse)
                    neffs_list.append(n_eff)

                    all_results.append({
                        "dist": name, "d": d, "m": m,
                        "bandwidth_mode": mode, "h": round(h, 6),
                        "t": round(t, 6), "sigma_t": round(sig_t, 6),
                        "nw_mse_mean": result["nw_mse_mean"],
                        "nw_mse_std": result["nw_mse_std"],
                        "vel_mse": vel_mse,
                        "neff_mean": n_eff,
                        "n_valid": result["n_valid"],
                    })

                    print(f"  m={m:5d}  h={h:.4f}  t={t:.4f}  σ_t={sig_t:.4f}  "
                          f"NW_MSE={nw_mse:.6f}  vel_MSE={vel_mse:.4f}  "
                          f"n_eff={n_eff:.1f}  [{result['n_valid']}/{args.n_tasks}]")

                if len(ms_used) >= 3:
                    alpha_nw_fit, r2_nw = fit_power_law(ms_used, nw_mses)
                    alpha_vel, r2_vel = fit_power_law(ms_used, vel_mses)
                else:
                    alpha_nw_fit, r2_nw = float("nan"), 0.0
                    alpha_vel, r2_vel = float("nan"), 0.0

                if mode == "adapted":
                    alpha_nw_theory = alpha_nw  # 4/(4+d)
                    alpha_vel_theory = 2.0 / (4.0 + d)  # 2/(4+d), sigma_t ~ h ~ m^{-1/(4+d)}
                else:
                    alpha_nw_theory = 1.0  # variance-dominated at fixed h
                    alpha_vel_theory = 1.0  # sigma_t constant at fixed h

                row = {
                    "dist": name, "d": d, "bandwidth": mode,
                    "alpha_nw_mse": round(alpha_nw_fit, 4) if np.isfinite(alpha_nw_fit) else None,
                    "r2_nw_mse": round(r2_nw, 4),
                    "alpha_vel_mse": round(alpha_vel, 4) if np.isfinite(alpha_vel) else None,
                    "r2_vel_mse": round(r2_vel, 4),
                    "alpha_nw_theory": round(alpha_nw_theory, 4),
                    "alpha_vel_theory": round(alpha_vel_theory, 4),
                    "n_points": len(ms_used),
                }
                summary.append(row)

                if np.isfinite(alpha_nw_fit):
                    ratio_nw = alpha_nw_fit / alpha_nw_theory if alpha_nw_theory > 0 else float("nan")
                    ratio_vel = alpha_vel / alpha_vel_theory if alpha_vel_theory > 0 else float("nan")
                    print(f"  → NW MSE:  α = {alpha_nw_fit:.3f} (R² = {r2_nw:.3f}), "
                          f"theory = {alpha_nw_theory:.3f}, ratio = {ratio_nw:.2f}×")
                    print(f"  → Vel MSE: α = {alpha_vel:.3f} (R² = {r2_vel:.3f}), "
                          f"theory = {alpha_vel_theory:.3f}, ratio = {ratio_vel:.2f}×")
                else:
                    print(f"  → insufficient data for fit")

    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nw_velocity_mse_verification.json"
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "summary": summary}, f, indent=2)
    print(f"\nSaved to {out_path}")

    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"{'Dist':<10} {'d':>3} {'BW':<8} "
          f"{'α_NW':>7} {'th':>6} {'ratio':>6} {'R²':>5}   "
          f"{'α_vel':>7} {'th':>6} {'ratio':>6} {'R²':>5}")
    print("-" * 85)
    for s in summary:
        a_nw = s["alpha_nw_mse"]
        a_vel = s["alpha_vel_mse"]
        a_nw_th = s["alpha_nw_theory"]
        a_vel_th = s["alpha_vel_theory"]
        if a_nw is not None and a_vel is not None:
            r_nw = a_nw / a_nw_th if a_nw_th > 0 else float("nan")
            r_vel = a_vel / a_vel_th if a_vel_th > 0 else float("nan")
            print(f"{s['dist']:<10} {s['d']:>3} {s['bandwidth']:<8} "
                  f"{a_nw:>7.3f} {a_nw_th:>6.3f} {r_nw:>5.2f}× {s['r2_nw_mse']:>5.3f}   "
                  f"{a_vel:>7.3f} {a_vel_th:>6.3f} {r_vel:>5.2f}× {s['r2_vel_mse']:>5.3f}")
        else:
            print(f"{s['dist']:<10} {s['d']:>3} {s['bandwidth']:<8}   N/A")


if __name__ == "__main__":
    main()
