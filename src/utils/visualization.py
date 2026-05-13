"""Minimal visualization utilities for 2D ICFM experiments."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import Tensor


def plot_task_samples(
    support: Tensor,
    generated: Tensor,
    target: Tensor,
    save_path: str | Path | None = None,
    title: str = "",
) -> None:
    """2D scatter plot of support, generated, and true target samples."""
    assert support.dim() == 2 and support.shape[1] == 2, "Only 2D supported"
    s = support.detach().cpu()
    g = generated.detach().cpu()
    t = target.detach().cpu()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(t[:, 0], t[:, 1], c="gray", s=8, alpha=0.3, label=f"True (n={len(t)})")
    ax.scatter(g[:, 0], g[:, 1], c="red", s=8, alpha=0.5, label=f"Generated (n={len(g)})")
    ax.scatter(
        s[:, 0], s[:, 1], c="blue", s=30, alpha=0.8, marker="x", label=f"Support (m={len(s)})"
    )
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
