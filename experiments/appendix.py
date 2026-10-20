"""Appendices D and F: Tables 5, 7, 9, 10 and 11, Figures 6 and 7 and failure confidences."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments import (
    FIGURE_METHODS,
    METHODS,
    PROB_COLUMNS,
    aligned,
    ari_at,
    brier,
    ensemble_log_probs,
    error_metrics,
    errors,
    load,
    nll,
    predictions,
    write_section,
)
from experiments.latex import (
    escape,
    ranked_rows,
    recovery,
    row,
    seconds,
    thousands,
    write_tabular,
)
from experiments.style import ERROR_LABEL, clean_axes, figure_label, save_figure, set_style

STRATA = {
    "true_k": [(r"$ k \leq 8$", 2, 8), (r"$ k > 8$", 8, 15)],
    "n_objects": [(r"$n \leq 1000$", 0, 1000), (r"$n > 1000$", 1001, 10000)],
    "n_features": [(r"$m \leq 64$", 2, 64), (r"$m > 64$", 65, 500)],
}
TABLE5_COLUMNS = [
    "openml_id",
    "dataset",
    "n_objects",
    "n_features",
    "true_k",
    "n_missing",
    "n_categorical",
]
CC18_CELL_THRESHOLD = 500_000
HOME_MAX_OBJECTS = 1000
HOME_MAX_FEATURES = 64
HOME_MAX_K = 10
FAILURE_IDS = {4153: "Smartphone", 54: "vehicle", 40691: "wine-quality-red"}
ROBUSTNESS = [
    ("noisy_features", "Noisy features", r"+{:.0f}\% of $m$"),
    ("anomalies", "Diffuse outliers", r"{:.0f}\% of $n$"),
]


def table5(frame: pd.DataFrame) -> None:
    """Table 5: the 50 benchmark datasets."""
    body = [
        row([escape(str(r[c])) for c in TABLE5_COLUMNS])
        for r in frame.sort_index().reset_index().to_dict("records")
    ]
    header = [
        r"\textbf{ID} & \textbf{Name} & \textbf{Objects ($n$)} & \textbf{Features ($m$)} & "
        r"\textbf{$k$} & \textbf{\#Missing} & \textbf{\#Categorical} \\"
    ]
    write_tabular("table5_datasets", "rlrrrr r", header, body)


def table7(frame: pd.DataFrame) -> None:
    """Table 7: TabK and TabClustPFN within TabClustPFN's training-scale bounds."""
    subset = frame.loc[
        (frame["n_objects"] <= HOME_MAX_OBJECTS)
        & (frame["n_features_ohe"] <= HOME_MAX_FEATURES)
        & (frame["true_k"] <= HOME_MAX_K)
    ]
    latency = aligned("latency", subset)
    true_k = subset["true_k"].to_numpy()
    methods = {"TabK": r"\textbf{TabK}", "TCP": "TabClustPFN"}
    metrics = pd.DataFrame(
        {
            m: {
                **error_metrics(true_k, subset[m].to_numpy()),
                "ARI": ari_at(subset, subset[m].to_numpy(), "k_means").mean(),
                "latency": latency[m].mean(),
            }
            for m in methods
        }
    ).T
    cells = ranked_rows(
        metrics,
        [
            ("MAE", 2, True),
            ("ARI", 3, False),
            ("EM", 1, False),
            ("tol1", 1, False),
            ("latency", seconds, True),
        ],
        underline_second=False,
    )
    body = [row([label, *method_cells]) for label, method_cells in zip(methods.values(), cells)]
    header = [
        r"Method &  MAE $\downarrow$ & ARI ($k$-means) $\uparrow$ & EM (\%) $\uparrow$ & "
        r"$\tau\leq1$ (\%) $\uparrow$ & Latency (s) $\downarrow$ \\"
    ]
    write_tabular("table7_home_turf", "l c c c c c", header, body)


def table9(frame: pd.DataFrame) -> None:
    """Table 9: MAE and agreement with the clean prediction, averaged over seeds."""
    data = load("robustness")
    clean = data.loc[data["condition"] == "clean"].set_index("openml_id")["pred_k"]
    data["error"] = (data["pred_k"] - frame["true_k"].reindex(data["openml_id"]).to_numpy()).abs()
    data["agreement"] = (data["pred_k"] == clean.reindex(data["openml_id"]).to_numpy()) * 100
    per_seed = pd.DataFrame(
        data.groupby(["condition", "level", "seed"])[["error", "agreement"]].mean()
    )
    table = pd.DataFrame(per_seed.groupby(["condition", "level"]).mean())
    clean_row = table.loc["clean"].iloc[0]
    body = [
        row(["None (clean)", "--", f"{clean_row['error']:.2f}", f"{clean_row['agreement']:.1f}"])
    ]
    for condition, label, level_format in ROBUSTNESS:
        body.append(r"\midrule")
        for i, (level, values) in enumerate(table.loc[condition].iterrows()):
            body.append(
                row(
                    [
                        label if i == 0 else "",
                        level_format.format(float(level) * 100),
                        f"{values['error']:.2f}",
                        f"{values['agreement']:.1f}",
                    ]
                )
            )
    header = [r"Perturbation & Level & MAE $\downarrow$ & Agreement (\%) $\uparrow$ \\"]
    write_tabular("table9_robustness", "l l c c", header, body)


def cc18() -> pd.DataFrame:
    """Read the 25 CC18 datasets with their table sizes."""
    data = load("cc18_25")
    data["cells"] = data["n_objects"] * data["n_features"]
    return data


def table10(data: pd.DataFrame) -> None:
    """Table 10: TabK and TabClustPFN on all CC18 datasets and the >500k-cell subset."""
    methods = {"TabK": r"\textbf{TabK}", "TCP": "TCP"}
    body = []
    for subset, mask in [
        ("CC18, all", np.ones(len(data), dtype=bool)),
        (r"CC18, $>$500k cells", data["cells"] > CC18_CELL_THRESHOLD),
    ]:
        metrics = {}
        for method in methods:
            part = data.loc[mask & (data["method"] == method)]
            true_k = part["true_k"].to_numpy(int)
            probs = part.reindex(columns=PROB_COLUMNS).fillna(0).to_numpy(float)
            metrics[method] = {
                **error_metrics(true_k, part["pred_k"].to_numpy(int)),
                "Brier": brier(probs, true_k),
                "NLL": nll(probs, true_k),
                "latency": part["gpu_seconds"].mean(),
                "n": len(part),
            }
        table = pd.DataFrame(metrics).T
        cells = ranked_rows(
            table,
            [
                ("MAE", 2, True),
                ("EM", 1, False),
                ("tol1", 1, False),
                ("Brier", 3, True),
                ("NLL", 3, True),
                ("latency", 2, True),
            ],
            underline_second=False,
        )
        if body:
            body.append(r"\midrule")
        for i, (label, method_cells) in enumerate(zip(methods.values(), cells)):
            name = f"{subset} ({int(table['n'].iloc[0])})" if i == 0 else ""
            body.append(row([name, label, *method_cells]))
    header = [
        r"Subset & Method & MAE $\downarrow$ & EM (\%) $\uparrow$ & "
        r"$\tau \leq 1$ (\%) $\uparrow$ & Brier $\downarrow$ & NLL $\downarrow$ & "
        r"Latency (s) $\downarrow$ \\"
    ]
    write_tabular("table10_cc18", "l l c c c c c c", header, body)


def table11(data: pd.DataFrame) -> None:
    """Table 11: per-dataset CC18 predictions ordered by table size."""
    keys = ["openml_id", "dataset", "n_objects", "n_features", "cells", "true_k"]
    table = data.pivot(index=keys, columns="method", values="pred_k").reset_index()
    body = []
    above = False
    for r in table.sort_values("cells").to_dict("records"):
        if not above and r["cells"] > CC18_CELL_THRESHOLD:
            body.append(r"\midrule")
            above = True
        body.append(
            row(
                [
                    str(r["openml_id"]),
                    escape(r["dataset"]),
                    thousands(r["n_objects"]),
                    thousands(r["n_features"]),
                    thousands(r["cells"]),
                    str(r["true_k"]),
                    recovery(r["TabK"], r["true_k"]),
                    recovery(r["TCP"], r["true_k"]),
                ]
            )
        )
    header = [r"ID & Dataset & $n$ & $m$ & Cells & $k^*$ & TabK & TCP \\"]
    write_tabular("table11_cc18_datasets", "r l r r r c c c", header, body)


def failure_confidences(frame: pd.DataFrame) -> None:
    """Appendix F.7: ensemble confidence for the discussed failure cases."""
    confidence = np.exp(ensemble_log_probs(frame).max(axis=1)) * 100
    positions = frame.index.get_indexer(list(FAILURE_IDS))
    write_section(
        "appendixF.7",
        {
            f"{name}_confidence_percent_at_k{int(frame['TabK'].iloc[i])}": confidence[i]
            for name, i in zip(FAILURE_IDS.values(), positions)
        },
    )


def structure(frame: pd.DataFrame) -> None:
    """Figure 6: MAE by number of clusters, objects and features."""
    methods = [m for m in METHODS if m in FIGURE_METHODS]
    labels = [figure_label(m) for m in methods]
    err = errors(frame, methods)
    matrices = {}
    for feature, bins in STRATA.items():
        values = frame[feature].to_numpy(float)
        matrix = np.full((len(methods), len(bins)), np.nan)
        for j, (_, lo, hi) in enumerate(bins):
            mask = (values >= lo) & (values <= hi)
            matrix[:, j] = err.loc[mask].mean().to_numpy()
        matrices[feature] = matrix
    all_values = np.concatenate([m.ravel() for m in matrices.values()])
    norm = mcolors.Normalize(vmin=np.nanmin(all_values), vmax=np.nanmax(all_values))
    cmap = sns.color_palette("rocket", as_cmap=True).reversed()
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(STRATA) + 1,
        figsize=(5.5, 0.22 * len(methods) + 0.6),
        gridspec_kw={"width_ratios": [len(b) for b in STRATA.values()] + [0.3], "wspace": 0.12},
    )
    images = []
    for idx, (feature, matrix) in enumerate(matrices.items()):
        ax = axes[idx]
        images.append(ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm))
        ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels([b[0] for b in STRATA[feature]], rotation=0)
        if idx == 0:
            ax.set_yticks(np.arange(len(methods)))
            ax.set_yticklabels(labels)
        else:
            ax.set_yticks([])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if norm(value) > 0.55 else "black",
                    fontweight="bold" if labels[i] == "TabK" else "normal",
                )
        for spine in ax.spines.values():
            spine.set_visible(False)
    colorbar = fig.colorbar(images[-1], cax=axes[-1])
    for spine in colorbar.ax.spines.values():
        spine.set_visible(False)
    colorbar.ax.tick_params(labelsize=6, length=2)
    colorbar.set_label(ERROR_LABEL, fontsize=7, labelpad=2)
    save_figure(fig, "figure6_structure", dpi=300)


def composition(frame: pd.DataFrame) -> None:
    """Figure 7: MAE by missing values and categorical features."""
    missing, categorical = frame["n_missing"] > 0, frame["n_categorical"] > 0
    groups = {
        "Complete Numerical": ~missing & ~categorical,
        "Missing Only": missing & ~categorical,
        "Categorical Only": ~missing & categorical,
        "Missing + Categorical": missing & categorical,
    }
    err = errors(frame, FIGURE_METHODS)
    order = np.argsort([err[m].mean() for m in FIGURE_METHODS])
    methods = [FIGURE_METHODS[i] for i in order]
    bar_width = 0.18
    x = np.arange(len(methods))
    group_colors = ["#E63946", "#2A9D8F", "#457B9D", "#F4A261"]
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    for idx, (group, mask) in enumerate(groups.items()):
        ax.bar(
            x + (idx - (len(groups) - 1) / 2) * bar_width,
            err.loc[mask, methods].mean().to_numpy(),
            width=bar_width,
            label=group,
            color=group_colors[idx],
            edgecolor="#222222",
            linewidth=0.4,
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([figure_label(m) for m in methods], rotation=45, ha="right")
    ax.set_ylabel(ERROR_LABEL)
    ax.legend(fontsize=6, ncol=2, loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    clean_axes(ax)
    save_figure(fig, "figure7_composition")


def main() -> None:
    """Write the appendix tables, Figures 6 and 7 and the Appendix F.7 numbers."""
    set_style()
    frame = predictions()
    data = cc18()
    table5(frame)
    table7(frame)
    table9(frame)
    table10(data)
    table11(data)
    structure(frame)
    composition(frame)
    failure_confidences(frame)


if __name__ == "__main__":
    main()
