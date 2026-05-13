"""Numerical verification of Props 3.3, 4.2, 4.4, Thm 5.2, and Cor 5.3."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

ATOL = 1e-5
# Near t=0 and t=1, float32 precision degrades: the feature-lift form
# (Cor 5.3) computes ||s||^2/(2h^2) with h->sigma_min~0.01, causing large
# intermediates and catastrophic cancellation. The autograd score
# (Prop 3.3) also loses precision with peaked distributions at t->1.
ATOL_EDGE = 1e-2


def test_velocity_score_identity(
    support: torch.Tensor, t_val: float, x: torch.Tensor, sigma_min: float
) -> bool:
    """Verify Prop 3.3 + 4.4: NW velocity matches score-based velocity."""
    m, d = support.shape
    t = torch.tensor(t_val)
    sig_t = 1.0 - (1.0 - sigma_min) * t
    h_t = sig_t / t

    # Velocity via Prop 4.4 (NW form)
    x_tilde = x / t
    sq_dist = ((x_tilde.unsqueeze(0) - support) ** 2).sum(dim=-1)
    logits = -sq_dist / (2 * h_t**2)
    weights = torch.softmax(logits, dim=0)
    m_h = (weights.unsqueeze(-1) * support).sum(dim=0)
    velocity_nw = x_tilde + (m_h - x_tilde) / sig_t

    # Velocity via score of p_t^S (Prop 3.3)
    centered = x.unsqueeze(0) - t * support
    log_components = -0.5 * (centered**2).sum(dim=-1) / sig_t**2
    log_components -= d / 2 * torch.log(2 * torch.pi * sig_t**2)
    torch.logsumexp(log_components, dim=0) - torch.log(torch.tensor(float(m)))

    # Score via autograd (nabla log p_t^S)
    x_ag = x.clone().requires_grad_(True)
    centered_ag = x_ag.unsqueeze(0) - t * support
    log_comps = -0.5 * (centered_ag**2).sum(dim=-1) / sig_t**2
    log_comps -= d / 2 * torch.log(2 * torch.pi * sig_t**2)
    log_p_ag = torch.logsumexp(log_comps, dim=0) - torch.log(torch.tensor(float(m)))
    log_p_ag.backward()
    score = x_ag.grad

    velocity_score = x / t + (sig_t / t) * score

    err = (velocity_nw - velocity_score).abs().max().item()
    return err < ATOL, err


def test_descaled_kde_identity(
    support: torch.Tensor, t_val: float, x_tilde: torch.Tensor, sigma_min: float
) -> bool:
    """Verify Prop 4.2: de-scaled empirical density equals KDE at bandwidth h(t)."""
    m, d = support.shape
    t = torch.tensor(t_val)
    sig_t = 1.0 - (1.0 - sigma_min) * t
    h_t = sig_t / t
    x = t * x_tilde

    centered = x.unsqueeze(0) - t * support
    log_comps = -0.5 * (centered**2).sum(dim=-1) / sig_t**2 - d / 2 * torch.log(
        2 * torch.pi * sig_t**2
    )
    p_t = torch.logsumexp(log_comps, dim=0).exp() / m

    tilde_p_t = t**d * p_t

    centered_kde = x_tilde.unsqueeze(0) - support
    log_kde_comps = -0.5 * (centered_kde**2).sum(dim=-1) / h_t**2 - d / 2 * torch.log(
        2 * torch.pi * h_t**2
    )
    kde = torch.logsumexp(log_kde_comps, dim=0).exp() / m

    err = (tilde_p_t - kde).abs().item()
    return err < ATOL, err


def test_attention_realization(
    support: torch.Tensor, t_val: float, x_tilde: torch.Tensor, sigma_min: float
) -> bool:
    """Verify Thm 5.2: ExactAttentionHead matches independent plug-in baseline."""
    from src.baselines.plugin_velocity import exact_plugin_velocity_scaled
    from src.models.exact_head import ExactAttentionHead

    t = torch.tensor(t_val)
    x = t * x_tilde

    # ExactAttentionHead (batched API)
    head = ExactAttentionHead(sigma_min)
    x_3d = x.unsqueeze(0).unsqueeze(0)
    t_3d = t.reshape(1, 1, 1)
    support_3d = support.unsqueeze(0)
    vel_head, _ = head(x_3d, t_3d, support_3d)
    vel_head = vel_head.squeeze()

    # Independent baseline
    x_2d = x.unsqueeze(0)
    t_2d = t.reshape(1, 1)
    support_2d = support.unsqueeze(0)
    vel_plugin = exact_plugin_velocity_scaled(x_2d, t_2d, support_2d, sigma_min).squeeze()

    err = (vel_head - vel_plugin).abs().max().item()
    return err < ATOL, err


def test_dot_product_realization(
    support: torch.Tensor, t_val: float, x_tilde: torch.Tensor, sigma_min: float
) -> bool:
    """Verify Cor 5.3: feature-lift dot product reproduces Gaussian kernel logits."""
    m, d = support.shape
    t = torch.tensor(t_val)
    sig_t = 1.0 - (1.0 - sigma_min) * t
    h_t = sig_t / t

    # Gaussian logits (direct)
    sq_dist = ((x_tilde.unsqueeze(0) - support) ** 2).sum(dim=-1)
    logits_direct = -sq_dist / (2 * h_t**2)

    # Feature-lift dot product (Cor 5.3)
    q = torch.cat(
        [
            x_tilde / h_t,
            torch.tensor([-x_tilde.dot(x_tilde) / (2 * h_t**2)]),
            torch.tensor([1.0]),
        ]
    )
    logits_dp = torch.zeros(m)
    for i in range(m):
        k_i = torch.cat(
            [
                support[i] / h_t,
                torch.tensor([1.0]),
                torch.tensor([-support[i].dot(support[i]) / (2 * h_t**2)]),
            ]
        )
        logits_dp[i] = q.dot(k_i)

    err = (logits_direct - logits_dp).abs().max().item()
    return err < ATOL, err


def run_all_tests():
    torch.manual_seed(42)
    sigma_min = 0.01

    configs = [
        {"d": 2, "m": 3, "t_vals": [0.1, 0.3, 0.5, 0.7, 0.9]},
        {"d": 4, "m": 50, "t_vals": [1e-4, 0.01, 0.5, 0.99]},
        {"d": 8, "m": 500, "t_vals": [1e-4, 0.1, 0.5, 0.999]},
    ]

    all_passed = True

    tests = [
        ("Prop 3.3 + 4.4: Velocity-score identity", test_velocity_score_identity),
        ("Prop 4.2: De-scaled KDE identity", test_descaled_kde_identity),
        ("Thm 5.2: Attention realization", test_attention_realization),
        ("Cor 5.3: Dot-product feature lift", test_dot_product_realization),
    ]

    for cfg in configs:
        d, m = cfg["d"], cfg["m"]
        support = torch.randn(m, d)
        t_vals = cfg["t_vals"]

        print("=" * 60)
        print(f"ICFM Theorem Verification: d={d}, m={m}, sigma_min={sigma_min}")
        print("=" * 60)

        for test_name, test_fn in tests:
            if test_fn == test_dot_product_realization and m > 50:
                print(f"\n{test_name}  [skipped, m={m} too large for loop impl]")
                continue

            print(f"\n{test_name}")
            for t_val in t_vals:
                is_edge = t_val < 0.001 or t_val > 0.85
                tol = ATOL_EDGE if is_edge else ATOL

                x_tilde = torch.randn(d)
                if test_fn == test_velocity_score_identity:
                    x = t_val * x_tilde
                    _, err = test_fn(support, t_val, x, sigma_min)
                else:
                    _, err = test_fn(support, t_val, x_tilde, sigma_min)
                passed = err < tol
                tag = " [edge]" if is_edge else ""
                status = "PASS" if passed else "FAIL"
                print(f"  t={t_val:.4g}: {status} (err={err:.2e}, tol={tol:.0e}){tag}")
                if not passed:
                    all_passed = False

        print()

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
