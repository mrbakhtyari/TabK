"""Section 4.3: Figure 4 and the in-text numbers on uncertainty and scalability."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax

from experiments import (
    PROB_COLUMNS,
    aligned,
    ensemble_log_probs,
    nll,
    predictions,
    write_section,
)
from experiments.style import clean_axes, save_figure, set_style

N_THRESHOLDS = 50
RISK_STYLES = {
    "TabK": {"color": "#E63946", "linewidth": 2.5, "linestyle": "-"},
    "TCP": {"color": "#457B9D", "linewidth": 2.0, "linestyle": "--"},
}
# Red shades for TabK and blue shades for TCP, as in the other figures; darker for GPU
SCALING_SERIES = [
    ("TabK (GPU)", "TabK (CUDA)", "-", "o", "#9C2F2F"),
    ("TCP (GPU)", "TCP (CUDA)", "--", "s", "#1D4E89"),
    ("TabK", "TabK (CPU)", "-.", "^", "#E63946"),
    ("TCP", "TCP (CPU)", (0, (1.2, 1.2)), "D", "#4EA8DE"),
]


def risk_coverage(frame: pd.DataFrame, mean_log: np.ndarray, tcp: np.ndarray) -> None:
    """Figure 4a: MAE of the predictions retained above each confidence threshold."""
    true_k = frame["true_k"].to_numpy(int)
    confidences = {"TabK": np.exp(mean_log.max(axis=1)), "TCP": tcp.max(axis=1)}
    fig, ax = plt.subplots()
    for method, confidence in confidences.items():
        err = np.abs(frame[method].to_numpy(int) - true_k)
        thresholds = np.linspace(0.0, confidence.max(), N_THRESHOLDS)
        maes = [
            err[confidence >= t].mean() if (confidence >= t).sum() > 0 else np.nan
            for t in thresholds
        ]
        ax.plot(thresholds, maes, label=method, **RISK_STYLES[method])
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("MAE on retained subset")
    ax.set_ylim(bottom=0)
    ax.legend(
        loc="upper right",
        framealpha=0.9,
        fontsize=7,
        handlelength=1.5,
        labelspacing=0.3,
        borderpad=0.3,
    )
    clean_axes(ax)
    save_figure(fig, "figure4a_risk_coverage")


def fit_loglog(x: np.ndarray, y: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, ...]:
    """Log-log least-squares fit with a one residual standard deviation band."""
    lx, ly = np.log10(x), np.log10(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    resid_std = np.std(ly - (slope * lx + intercept))
    ly_eval = slope * np.log10(x_eval) + intercept
    return 10**ly_eval, 10 ** (ly_eval - resid_std), 10 ** (ly_eval + resid_std)


def scalability(frame: pd.DataFrame, latency: pd.DataFrame) -> None:
    """Figure 4b: latency against the number of objects for TabK and TCP on CPU and GPU."""
    order = np.argsort(frame["n_objects"].to_numpy())
    x = frame["n_objects"].to_numpy(float)[order]
    x_eval = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    fig, ax = plt.subplots()
    for column, label, linestyle, point, line_color in SCALING_SERIES:
        y = latency[column].to_numpy(float)[order] * 1000
        y_fit, y_lo, y_hi = fit_loglog(x, y, x_eval)
        ax.fill_between(x_eval, y_lo, y_hi, color=line_color, alpha=0.18, linewidth=0, zorder=1)
        ax.plot(x_eval, y_fit, linestyle=linestyle, color=line_color, label=label, zorder=3)
        ax.plot(
            x,
            y,
            linestyle="None",
            marker=point,
            markerfacecolor="white",
            markeredgecolor=line_color,
            markeredgewidth=0.9,
            zorder=4,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of objects ($n$)")
    ax.set_ylabel("Inference time (ms)")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=5)
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, "figure4b_scalability")


def main() -> None:
    """Write Figure 4 and the Section 4.3 numbers."""
    set_style()
    frame = predictions()
    latency = aligned("latency", frame)
    mean_log = ensemble_log_probs(frame)
    tcp = aligned("tcp_probs", frame)[PROB_COLUMNS].to_numpy(float)
    true_k = frame["true_k"].to_numpy()
    risk_coverage(frame, mean_log, tcp)
    scalability(frame, latency)
    write_section(
        "section4.3",
        {
            "TabK_NLL": nll(softmax(mean_log, axis=1), true_k),
            "TCP_NLL": nll(tcp, true_k),
            "TabK_CPU_faster_than_TCP_GPU_percent": (latency["TabK"] < latency["TCP (GPU)"]).mean()
            * 100,
        },
    )


if __name__ == "__main__":
    main()
