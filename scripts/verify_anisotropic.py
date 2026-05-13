"""Numerical verification of anisotropic (Mahalanobis) kernel realization.

Checks that Mahalanobis-kernel NW velocity matches isotropic NW on
L-transformed coordinates (where M = L^T L) at multiple (d, t, M) settings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

ATOL = 1e-5
ATOL_EDGE = 1e-2
SIGMA_MIN = 0.01


def random_psd_matrix(d):
    L = torch.randn(d, d, dtype=torch.float64)
    M = L.T @ L + 0.1 * torch.eye(d, dtype=torch.float64)
    L_chol = torch.linalg.cholesky(M)
    return M, L_chol


def mahalanobis_nw_velocity(x_tilde, support, M, h, sigma_t):
    diff = x_tilde.unsqueeze(0) - support
    mahal_sq = (diff @ M * diff).sum(dim=-1)
    logits = -mahal_sq / (2 * h**2)
    weights = torch.softmax(logits, dim=0)
    m_M = (weights.unsqueeze(-1) * support).sum(dim=0)
    velocity = x_tilde + (m_M - x_tilde) / sigma_t
    return velocity, weights


def transformed_isotropic_velocity(x_tilde, support, L, h, sigma_t):
    # ||x-s||_M^2 = ||L^T(x-s)||^2, so transform by L^T
    Lt = L.T
    y_tilde = Lt @ x_tilde
    support_y = (Lt @ support.T).T

    diff = y_tilde.unsqueeze(0) - support_y
    sq_dist = (diff**2).sum(dim=-1)
    logits = -sq_dist / (2 * h**2)
    weights = torch.softmax(logits, dim=0)

    m_h = (weights.unsqueeze(-1) * support).sum(dim=0)
    velocity = x_tilde + (m_h - x_tilde) / sigma_t
    return velocity, weights


def test_all_equivalences(d, m, t_val):
    """Test weight, velocity, and feature-lift equivalence on shared random data."""
    M, L = random_psd_matrix(d)
    support = torch.randn(m, d, dtype=torch.float64)
    x_tilde = torch.randn(d, dtype=torch.float64)

    sig_t = 1.0 - (1.0 - SIGMA_MIN) * t_val
    h = sig_t / t_val

    v_direct, w_direct = mahalanobis_nw_velocity(x_tilde, support, M, h, sig_t)
    v_transformed, w_transformed = transformed_isotropic_velocity(x_tilde, support, L, h, sig_t)

    w_err = (w_direct - w_transformed).abs().max().item()
    v_err = (v_direct - v_transformed).abs().max().item()

    # Feature-lift equivalence (Thm: anisotropic realization)
    Lt = L.T
    diff = x_tilde.unsqueeze(0) - support
    mahal_sq = (diff @ M * diff).sum(dim=-1)
    logits_direct = -mahal_sq / (2 * h**2)

    Lx = Lt @ x_tilde
    Ls = (Lt @ support.T).T
    x_mahal_sq = (x_tilde @ M * x_tilde).sum()
    s_mahal_sq = (support @ M * support).sum(dim=-1)

    q = torch.cat([Lx / h, (-x_mahal_sq / (2 * h**2)).unsqueeze(0), torch.ones(1, dtype=torch.float64)])
    k = torch.zeros(m, d + 2, dtype=torch.float64)
    k[:, :d] = Ls / h
    k[:, d] = 1.0
    k[:, d + 1] = -s_mahal_sq / (2 * h**2)
    logits_lift = q @ k.T

    fl_err = (logits_direct - logits_lift).abs().max().item()

    return w_err, v_err, fl_err


def main():
    torch.manual_seed(42)
    print("Anisotropic (Mahalanobis) Kernel Realization Verification")
    print("=" * 70)

    configs = [
        {"d": 2, "m": 3, "t_vals": [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]},
        {"d": 4, "m": 50, "t_vals": [1e-4, 0.01, 0.1, 0.5, 0.9, 0.99]},
        {"d": 8, "m": 50, "t_vals": [0.01, 0.1, 0.5, 0.9]},
        {"d": 16, "m": 50, "t_vals": [0.1, 0.5, 0.9]},
        {"d": 32, "m": 50, "t_vals": [0.1, 0.5]},
    ]

    all_passed = True
    total_tests = 0
    n_trials = 3

    for cfg in configs:
        d, m = cfg["d"], cfg["m"]
        print(f"\n--- d={d}, m={m} ---")

        for trial in range(n_trials):
            for t_val in cfg["t_vals"]:
                is_edge = t_val < 0.001 or t_val > 0.85
                tol = ATOL_EDGE if is_edge else ATOL

                w_err, v_err, fl_err = test_all_equivalences(d, m, t_val)
                err = max(w_err, v_err, fl_err)

                passed = err < tol
                status = "PASS" if passed else "FAIL"
                tag = " [edge]" if is_edge else ""
                print(
                    f"  trial={trial} t={t_val:.4g}: {status}"
                    f" (w={w_err:.2e}, v={v_err:.2e}, lift={fl_err:.2e}, tol={tol:.0e}){tag}"
                )

                if not passed:
                    all_passed = False
                total_tests += 1

    print(f"\n{'=' * 70}")
    print("Float32 verification")
    print("=" * 70)

    ATOL_F32 = 1e-3
    ATOL_F32_EDGE = 5e-2
    f32_passed = True
    f32_tests = 0

    for cfg in configs:
        d, m = cfg["d"], cfg["m"]
        for trial in range(2):
            for t_val in cfg["t_vals"]:
                is_edge = t_val < 0.001 or t_val > 0.85
                tol = ATOL_F32_EDGE if is_edge else ATOL_F32

                M32, L32 = random_psd_matrix(d)
                M32, L32 = M32.float(), L32.float()
                support32 = torch.randn(m, d, dtype=torch.float32)
                x32 = torch.randn(d, dtype=torch.float32)
                sig_t = 1.0 - (1.0 - SIGMA_MIN) * t_val
                h = sig_t / t_val

                v1, _ = mahalanobis_nw_velocity(x32, support32, M32, h, sig_t)
                v2, _ = transformed_isotropic_velocity(x32, support32, L32, h, sig_t)
                err = (v1 - v2).abs().max().item()

                passed = err < tol
                if not passed:
                    f32_passed = False
                    print(f"  FAIL d={d} t={t_val:.4g}: err={err:.2e} tol={tol:.0e}")
                f32_tests += 1

    if f32_passed:
        print(f"  All {f32_tests} float32 tests PASSED")
    else:
        print(f"  Some float32 tests FAILED (see above)")
        print(f"  Note: float32 errors at extreme t expected due to 1/sigma_t amplification")

    all_passed = all_passed and f32_passed
    total_tests += f32_tests

    print(f"\n{'=' * 70}")
    print(f"Total tests: {total_tests} (float64 + float32)")
    if all_passed:
        print("ALL ANISOTROPIC VERIFICATION TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
