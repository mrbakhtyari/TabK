from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS: OrderedDict[str, dict] = OrderedDict(
    [
        (
            "TabK",
            {"label": "TabK", "marker": "D", "color": "#E63946"},
        ),
        ("tabclustpfn", {"label": "TCP", "marker": "s", "color": "#457B9D"}),
        ("x_means", {"label": "X-Means", "marker": "^", "color": "#2A9D8F"}),
        ("silhouette", {"label": "Sil", "marker": "o", "color": "#264653"}),
        (
            "calinski_harabasz",
            {"label": "CH", "marker": "v", "color": "#E9C46A"},
        ),
        (
            "davies_bouldin",
            {"label": "DB", "marker": "<", "color": "#F4A261"},
        ),
        ("dunn", {"label": "Dunn", "marker": ">", "color": "#6A4C93"}),
        ("cnak", {"label": "CNAK", "marker": "P", "color": "#1D3557"}),
        ("gap_statistic", {"label": "Gap", "marker": "X", "color": "#A8DADC"}),
        ("gmm_bic", {"label": "GMM-BIC", "marker": "h", "color": "#606C38"}),
        ("imwkmeans", {"label": "IMWk", "marker": "p", "color": "#BC6C25"}),
        ("unseen_dcn", {"label": "UNSEEN", "marker": "H", "color": "#ADB5BD"}),
    ]
)


def method_label(csv_prefix: str) -> str:
    """Return the short paper-ready label for a CSV method prefix."""
    return METHODS[csv_prefix]["label"] if csv_prefix in METHODS else csv_prefix


def method_labels() -> list[str]:
    """Return all short labels in registry order."""
    return [m["label"] for m in METHODS.values()]


def method_prefixes() -> list[str]:
    """Return all CSV prefixes in registry order."""
    return list(METHODS.keys())


def set_style():
    # Total text width is 5.5 inches
    # For two subfigures, each should be ~2.65 inches
    SUB_WIDTH = 2.65

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Computer Modern Roman", "serif"],  #
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,  # [cite: 80]
            "ps.fonttype": 42,
            "figure.figsize": (SUB_WIDTH, 2.1),
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def clean_axes(ax=None):
    """
    Cleans up a single matplotlib axis by removing the top and right spines.
    Standard requirement for modern clean publication figures.
    """
    if ax is None:
        ax = plt.gca()

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def get_gradient_colors(cmap_name: str = "coolwarm", num_colors: int = 5) -> list:
    """
    Generates a spaced array of hex/RGB colors from a distinct matplotlib colormap.
    Helpful for coloring boxplots uniformly based on method ranks.
    """
    cmap = plt.get_cmap(cmap_name)
    # Exclude the absolute extreme edges (0.0 and 1.0) so colors aren't entirely invisible/white
    return [cmap(x) for x in np.linspace(0.05, 0.95, num_colors)]


def load_results(
    csv_path: str | Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Load the final results CSV and return (df, active_method_prefixes).

    Parameters
    ----------
    csv_path : path to the final CSV (e.g. ``results/real_world/final_50.csv``).

    Returns
    -------
    df : the raw DataFrame
    active : list of method prefixes present in the CSV, in registry order
    """
    df = pd.read_csv(csv_path)

    # Determine which methods are actually present
    csv_cols = set(df.columns)
    active: list[str] = []
    for prefix in METHODS:
        if f"{prefix}__pred_k" in csv_cols:
            active.append(prefix)

    return df, active


def extract_metric(
    df: pd.DataFrame,
    active: list[str],
    metric: str = "pred_k",
) -> pd.DataFrame:
    """Extract a per-method metric into a tidy DataFrame.

    Parameters
    ----------
    df : raw results DataFrame
    active : list of method prefixes
    metric : one of ``pred_k``, ``ari``, ``time_ms``, ``time_ms_cuda``

    Returns
    -------
    DataFrame with columns = short labels, index = dataset rows
    """
    cols = {}
    for prefix in active:
        col = f"{prefix}__{metric}"
        if col in df.columns:
            cols[method_label(prefix)] = df[col].values
    return pd.DataFrame(cols, index=df.index)
