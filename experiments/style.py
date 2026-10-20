"""The paper's figure style: NeurIPS dimensions, LaTeX typography and method colors."""

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle

from experiments import FIGURES

logging.getLogger("fontTools").setLevel(logging.ERROR)

# Two subfigures side by side in the 5.5-inch NeurIPS text width
SUB_WIDTH = 2.65
ERROR_LABEL = r"Mean absolute error $|\bar{k}-k^*|$"
# Method label -> (figure label, marker, color)
METHOD_STYLES: dict[str, tuple[str, str | MarkerStyle, str]] = {
    "TabK": ("TabK", "D", "#E63946"),
    "TCP": ("TCP", "s", "#457B9D"),
    "X-Means": ("X-Means", "^", "#2A9D8F"),
    "Sil": ("Sil", "o", "#264653"),
    "CH": ("CH", "v", "#E9C46A"),
    "DB": ("DB", "<", "#F4A261"),
    "Dunn": ("Dunn", ">", "#6A4C93"),
    "CNAK": ("CNAK", "P", "#1D3557"),
    "Gap": ("Gap", "X", "#A8DADC"),
    "GMM-BIC": ("GMM-BIC", "h", "#606C38"),
    "IMWk": ("IMWk", "p", "#BC6C25"),
    "UNSEEN": ("UNSEEN", "H", "#ADB5BD"),
    "SCARF+CH": ("SCARF", "8", "#C51B7D"),
    "SubTab+Sil": ("SubTab", MarkerStyle((6, 1, 0)), "#31A354"),
    "VIME+Sil": ("VIME", MarkerStyle((4, 1, 0)), "#8073AC"),
}


def set_style() -> None:
    """Configure LaTeX-rendered NeurIPS figures."""
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Computer Modern Roman", "serif"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
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


def figure_label(method: str) -> str:
    """Short label of a method in figures (encoders by name)."""
    return METHOD_STYLES[method][0]


def marker(method: str) -> str | MarkerStyle:
    """Marker of a method in figures."""
    return METHOD_STYLES[method][1]


def color(method: str) -> str:
    """Color of a method in figures."""
    return METHOD_STYLES[method][2]


def clean_axes(ax: Axes) -> None:
    """Remove the top and right spines."""
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def gradient_colors(num_colors: int, cmap_name: str = "coolwarm") -> list:
    """Evenly spaced colors from a colormap, excluding its extreme ends."""
    cmap = plt.get_cmap(cmap_name)
    return [cmap(x) for x in np.linspace(0.05, 0.95, num_colors)]


def save_figure(fig: Figure, name: str, **kwargs) -> None:
    """Save a figure as PDF and release it."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None}, **kwargs
    )
    plt.close(fig)
