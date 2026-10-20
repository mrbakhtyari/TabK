import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _setup_plot_style() -> None:
    """Configure matplotlib to use Cairo backend and serif fonts."""
    try:
        import matplotlib

        matplotlib.use("cairo")
    except ImportError:
        pass

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
            "font.size": 12,
        }
    )


def plot_distributions(
    k: np.ndarray,
    n: np.ndarray,
    d: np.ndarray,
    save_dir: Path | None = None,
    show: bool = False,
) -> None:
    """
    Plot histograms for Number of Clusters (k), Number of Instances (n), and Number of Features (d).
    """
    _setup_plot_style()

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    label_fs = 14
    tick_fs = 10
    title_fs = 14

    # k (Clusters)
    k_min, k_max = int(k.min()), int(k.max())
    bins_k = np.arange(k_min - 0.5, k_max + 1.5, 1)
    axs[0].hist(k, bins=bins_k, edgecolor="black", alpha=0.7)
    axs[0].set_title("Clusters (k)", fontsize=title_fs)
    axs[0].set_xlabel("Number of Clusters", fontsize=label_fs)
    axs[0].set_ylabel("Count", fontsize=label_fs)
    axs[0].tick_params(axis="both", labelsize=tick_fs)

    # n (Instances)
    axs[1].hist(n, bins=30, edgecolor="black", alpha=0.7)
    axs[1].set_title("Instances (n)", fontsize=title_fs)
    axs[1].set_xlabel("Number of Instances", fontsize=label_fs)
    axs[1].set_ylabel("Count", fontsize=label_fs)
    axs[1].tick_params(axis="both", labelsize=tick_fs)

    # d (Features)
    d_min, d_max = int(d.min()), int(d.max())
    bins_d = np.arange(d_min - 0.5, d_max + 1.5, 1)
    axs[2].hist(d, bins=bins_d, edgecolor="black", alpha=0.7)
    axs[2].set_title("Features (d)", fontsize=title_fs)
    axs[2].set_xlabel("Number of Features", fontsize=label_fs)
    axs[2].set_ylabel("Count", fontsize=label_fs)
    axs[2].tick_params(axis="both", labelsize=tick_fs)

    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "config_distributions.pdf"
        fig.savefig(save_path, dpi=600, format="pdf", bbox_inches="tight")
        logger.info(f"Saved distribution plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_coverage(
    k: np.ndarray,
    n: np.ndarray,
    d: np.ndarray,
    save_dir: Path | None = None,
    show: bool = False,
    seed: int = 42,
) -> None:
    """Plot (n, d), (k, n) and (k, d) coverage scatter plots in one figure."""
    _setup_plot_style()

    rng = np.random.default_rng(seed)
    k_jittered = k + rng.normal(0, 0.05, size=len(k))

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    plots = [
        (n, d, "n_samples", "d_features", "Coverage (n vs d)"),
        (k_jittered, n, "k_clusters (jittered)", "n_samples", "Coverage (k vs n)"),
        (k_jittered, d, "k_clusters (jittered)", "d_features", "Coverage (k vs d)"),
    ]

    for ax, (x_data, y_data, x_lbl, y_lbl, title) in zip(axs, plots):
        ax.scatter(x_data, y_data, s=15, alpha=0.6, edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel(x_lbl)
        ax.set_ylabel(y_lbl)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "config_coverage.pdf"
        fig.savefig(save_path, dpi=600, format="pdf", bbox_inches="tight")
        logger.info(f"Saved coverage plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
