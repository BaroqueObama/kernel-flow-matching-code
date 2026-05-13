"""Evaluation metrics: velocity MSE, MMD, C2ST."""

import torch
from torch import Tensor


def velocity_mse(predicted: Tensor, target: Tensor) -> Tensor:
    """Mean squared error between predicted and target velocity fields."""
    assert predicted.shape == target.shape, f"Shape mismatch: {predicted.shape} vs {target.shape}"
    return ((predicted - target) ** 2).sum(dim=-1).mean()


def gaussian_kernel_matrix(x: Tensor, y: Tensor, sigma: float) -> Tensor:
    """Gaussian kernel matrix between point sets x and y."""
    x_sq = (x**2).sum(dim=-1, keepdim=True)
    y_sq = (y**2).sum(dim=-1, keepdim=True)
    dist_sq = x_sq + y_sq.T - 2 * x @ y.T
    return torch.exp(-dist_sq / (2 * sigma**2))


def mmd_squared(samples: Tensor, reference: Tensor, sigma: float | None = None) -> Tensor:
    """Maximum Mean Discrepancy (squared) with Gaussian kernel.

    Uses the unbiased U-statistic estimator.

    Args:
        samples: Generated samples, shape (n, d).
        reference: Reference samples, shape (n_ref, d).
        sigma: Kernel bandwidth. If None, uses median heuristic.

    Returns:
        MMD^2 (scalar).
    """
    assert samples.dim() == 2 and reference.dim() == 2
    assert samples.shape[1] == reference.shape[1]

    if sigma is None:
        with torch.no_grad():
            all_pts = torch.cat([samples, reference], dim=0)
            dists = torch.cdist(all_pts, all_pts)
            sigma = dists.median().item()
            if sigma < 1e-8:
                sigma = 1.0

    k_xx = gaussian_kernel_matrix(samples, samples, sigma)
    k_yy = gaussian_kernel_matrix(reference, reference, sigma)
    k_xy = gaussian_kernel_matrix(samples, reference, sigma)

    n = samples.shape[0]
    m = reference.shape[0]

    # Unbiased estimator: exclude diagonal for k_xx and k_yy
    mmd2 = (
        (k_xx.sum() - k_xx.trace()) / (n * (n - 1))
        + (k_yy.sum() - k_yy.trace()) / (m * (m - 1))
        - 2 * k_xy.mean()
    )
    return mmd2


def energy_distance(samples: Tensor, reference: Tensor) -> Tensor:
    """Energy distance between two point sets.

    Uses L2 distances (not squared). Scale-sensitive alternative to MMD
    that does not require a bandwidth parameter.
    """
    assert samples.dim() == 2 and reference.dim() == 2
    assert samples.shape[1] == reference.shape[1]

    cross_dists = torch.cdist(samples, reference, p=2)
    cross_term = cross_dists.mean()

    n = samples.shape[0]
    if n > 1:
        self_dists_x = torch.cdist(samples, samples, p=2)
        self_term_x = self_dists_x.sum() / (n * (n - 1))
    else:
        self_term_x = torch.tensor(0.0, device=samples.device)

    m_ref = reference.shape[0]
    if m_ref > 1:
        self_dists_y = torch.cdist(reference, reference, p=2)
        self_term_y = self_dists_y.sum() / (m_ref * (m_ref - 1))
    else:
        self_term_y = torch.tensor(0.0, device=reference.device)

    return 2 * cross_term - self_term_x - self_term_y


def c2st_accuracy(samples: Tensor, reference: Tensor, n_folds: int = 5) -> float:
    """Classifier 2-Sample Test: train a classifier to distinguish two sets.

    Returns accuracy (0.5 = indistinguishable, 1.0 = perfectly separable).

    Uses a simple logistic regression for speed.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    x = torch.cat([samples, reference], dim=0).detach().cpu().numpy()
    y = np.concatenate([np.ones(len(samples)), np.zeros(len(reference))])

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    scores = cross_val_score(clf, x, y, cv=n_folds, scoring="accuracy")
    return float(scores.mean())
