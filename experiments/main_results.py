"""Section 4.2: Tables 1 and 12, Figures 2 and 3 and the in-text numbers."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import PchipInterpolator

from experiments import (
    FIGURE_METHODS,
    K_VALUES,
    METHODS,
    PROB_COLUMNS,
    aligned,
    ari_at,
    error_metrics,
    errors,
    load,
    predictions,
    write_section,
)
from experiments.latex import (
    fmt,
    ranked,
    ranked_rows,
    recovery,
    row,
    scientific,
    seconds,
    write_tabular,
)
from experiments.style import (
    ERROR_LABEL,
    clean_axes,
    color,
    figure_label,
    gradient_colors,
    marker,
    save_figure,
    set_style,
)

ORACLE_THRESHOLDS = np.arange(10, 31) / 100
ORACLE_BINS = np.arange(0.0, 0.9, 0.1)
TOLERANCES = np.arange(0, 6)
NEMENYI_ALPHA = 0.05
# Annotation placement around the points of Figure 2d and the ranks of Figure 2b
PARETO_PLACEMENT = {
    "TCP": "center-right",
    "CH": "center-left",
    "IMWk": "center-right",
    "DB": "center-right",
    "Dunn": "center-left",
    "GMM-BIC": "center-right",
    "X-Means": "top-left",
    "UNSEEN": "center-left",
    "SCARF": "center-right",
    "SubTab": "bottom-center",
    "VIME": "center-left",
}
PARETO_NUDGE = {"VIME": (0, -3), "SubTab": (2, 0)}
RANK_PLACEMENT = {"CNAK": "top-center", "DB": "top-center", "GMM-BIC": "top-center"}


def table1(frame: pd.DataFrame, latency: pd.DataFrame) -> None:
    """Table 1 with paired one-sided signed-rank tests against TabK."""
    err = errors(frame)
    true_k = frame["true_k"].to_numpy()
    metrics = pd.DataFrame(
        {
            m: {
                **error_metrics(true_k, frame[m].to_numpy()),
                "ARI1": ari_at(frame, frame[m].to_numpy(), "k_means").mean(),
                "ARI2": ari_at(frame, frame[m].to_numpy(), "gmm").mean(),
                "latency": latency[m].mean(),
            }
            for m in METHODS
        }
    ).T
    cells = ranked_rows(
        metrics,
        [
            ("MAE", 2, True),
            ("ARI1", 3, False),
            ("ARI2", 3, False),
            ("EM", 1, False),
            ("tol1", 1, False),
            ("latency", seconds, True),
        ],
    )
    body = []
    for method, method_cells in zip(METHODS, cells):
        if method == "TabK":
            p_value = "--"
        else:
            test = stats.wilcoxon(err["TabK"], err[method], alternative="less")
            p_value = scientific(float(test.pvalue))  # pyright: ignore[reportAttributeAccessIssue]
        name = r"\textbf{TabK}" if method == "TabK" else method
        body.append(row([name, *method_cells, p_value]))
    header = [
        r"& \multicolumn{6}{c}{\textbf{Performance metrics}} & "
        r"\multicolumn{1}{c}{\textbf{Wilcoxon test}} \\",
        r"\cmidrule(lr){2-7} \cmidrule(lr){8-8}",
        r"Method & MAE $\downarrow$ & $\text{ARI}_{1}$ $\uparrow$ & $\text{ARI}_{2}$ $\uparrow$ & "
        r"EM (\%) $\uparrow$ & $\tau \leq 1$ (\%) $\uparrow$ & Latency (s) $\downarrow$ & "
        r"$p$-value \\",
    ]
    write_tabular("table1_main", "l c c c c c c c", header, body)


def table12(frame: pd.DataFrame) -> None:
    """Table 12: per-dataset predictions with the MAE row."""
    err = errors(frame)
    header = [row(["ID", "$k^*$", *[rf"\rotatebox{{90}}{{{m}}}" for m in METHODS]])]
    body = [
        row(
            [str(r["openml_id"]), str(r["true_k"]), *[recovery(r[m], r["true_k"]) for m in METHODS]]
        )
        for r in frame.sort_index().reset_index().to_dict("records")
    ]
    mae = err.mean()
    body += [
        r"\midrule",
        row([r"\multicolumn{2}{l}{\textbf{MAE}}", *ranked(mae, fmt(mae, 2), lower_is_better=True)]),
    ]
    write_tabular("table12_per_dataset", "l c " + "c " * len(METHODS), header, body)


def section_numbers(frame: pd.DataFrame) -> None:
    """In-text numbers of Section 4.2."""
    err = errors(frame)
    values: dict[str, float] = {}
    for algorithm, label in [("k_means", "kmeans"), ("gmm", "gmm")]:
        oracle = ari_at(frame, frame["true_k"].to_numpy(), algorithm).mean()
        tabk = ari_at(frame, frame["TabK"].to_numpy(), algorithm).mean()
        values[f"oracle_ARI_{label}"] = oracle
        values[f"TabK_oracle_recovery_percent_{label}"] = 100 * tabk / oracle
    oracle = ari_at(frame, frame["true_k"].to_numpy(), "k_means")
    subset = err.loc[oracle >= 0.30].mean().sort_values()
    values["oracle_ARI_ge_0.30_datasets"] = int((oracle >= 0.30).sum())
    values["oracle_ARI_ge_0.30_TabK_MAE"] = subset["TabK"]
    values[f"oracle_ARI_ge_0.30_next_best_MAE ({subset.index[1]})"] = subset.iloc[1]
    first = [err.loc[oracle >= t].mean().idxmin() == "TabK" for t in ORACLE_THRESHOLDS]
    values["TabK_first_at_every_threshold_0.10_to_0.30"] = int(all(first))
    members = load("tabk_log_probs").set_index("openml_id").loc[frame.index]
    member_pred = members[PROB_COLUMNS].to_numpy().argmax(axis=1) + K_VALUES[0]
    member_err = np.abs(member_pred - frame["true_k"].reindex(members.index).to_numpy())
    member_ids = members["member"].to_numpy()
    member_mae = [member_err[member_ids == m].mean() for m in np.unique(member_ids)]
    values["single_member_mean_MAE"] = float(np.mean(member_mae))
    write_section("section4.2", values)


def annotation_offset(place: str, step: int = 4) -> tuple[int, int, str, str]:
    """Offset and alignment of a point annotation, e.g. "top-right" or "center-left"."""
    tokens = set(place.lower().strip().replace("_", "-").split("-"))
    center = "center" in tokens
    x, y, ha, va = 0, 0, "center", "center"
    if "left" in tokens:
        x, ha = -step, "right"
    elif "right" in tokens:
        x, ha = step, "left"
    if "top" in tokens:
        y, va = step, "bottom"
    elif "bottom" in tokens:
        y, va = -step, "top"
    if y == 0 and "top" not in tokens and "bottom" not in tokens and not center:
        y, va = step, "bottom"
    if x == 0 and "left" not in tokens and "right" not in tokens and not center:
        x, ha = step, "left"
    return x, y, ha, va


def boxplot(err: pd.DataFrame) -> None:
    """Figure 2a: per-dataset error distributions ordered by median."""
    medians = {m: np.nanmedian(err[m].to_numpy(float)) for m in FIGURE_METHODS}
    order = sorted(medians, key=medians.__getitem__)
    fig, ax = plt.subplots()
    boxes = ax.boxplot(
        [err[m].to_numpy(float) for m in order],
        vert=True,
        tick_labels=[figure_label(m) for m in order],
        patch_artist=True,
        showfliers=True,
        widths=0.6,
        medianprops=dict(color="white", linewidth=1.0),
        boxprops=dict(linewidth=0.6, edgecolor="#222222"),
        whiskerprops=dict(linewidth=0.6, color="#444444"),
        capprops=dict(linewidth=0.6, color="#444444"),
        flierprops=dict(
            marker="d",
            markersize=2.5,
            markerfacecolor="#888888",
            markeredgecolor="none",
            alpha=0.4,
        ),
    )
    for patch, fill in zip(boxes["boxes"], gradient_colors(len(order))):
        patch.set_facecolor(fill)
        patch.set_alpha(0.9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel(ERROR_LABEL)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    clean_axes(ax)
    save_figure(fig, "figure2a_boxplot")


def ranking(err: pd.DataFrame) -> None:
    """Figure 2b: average Friedman ranks with the Nemenyi critical difference."""
    matrix = pd.DataFrame(
        err[FIGURE_METHODS].to_numpy(), columns=[figure_label(m) for m in FIGURE_METHODS]
    )
    mean_ranks = matrix.rank(axis=1, method="average").mean().sort_values()
    n_methods = len(mean_ranks)
    q_alpha = stats.studentized_range.ppf(1 - NEMENYI_ALPHA, n_methods, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * len(err)))
    labels = [str(label) for label in mean_ranks.index]
    ranks = mean_ranks.to_numpy()
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.axhspan(
        ranks[0],
        ranks[0] + cd,
        color="#F2AEB5",
        alpha=0.45,
        label=f"Within CD of {labels[0]}",
        zorder=1,
    )
    ax.scatter(x, ranks, s=35, color="navy", edgecolors="none", zorder=3)
    for xi, yi, label in zip(x, ranks, labels):
        dx, dy, ha, va = annotation_offset(RANK_PLACEMENT.get(label, "top-center"))
        ax.annotate(
            f"{yi:.2f}",
            (float(xi), float(yi)),
            textcoords="offset points",
            xytext=(dx, dy),
            va=va,
            ha=ha,
            fontsize=7,
            rotation=90,
            color="#2f2f2f",
            zorder=4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Average rank")
    ax.invert_yaxis()
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=7)
    clean_axes(ax)
    save_figure(fig, "figure2b_ranking")


def tolerance(err: pd.DataFrame) -> None:
    """Figure 2c: cumulative accuracy at integer error tolerances."""
    fig, ax = plt.subplots()
    for method in FIGURE_METHODS:
        accuracy = [np.mean(err[method].to_numpy(float) <= t) * 100 for t in TOLERANCES]
        ax.plot(
            TOLERANCES,
            accuracy,
            marker=marker(method),
            color=color(method),
            label=figure_label(method),
            linewidth=1.0,
            markersize=3.5,
            alpha=0.85,
        )
    ax.set_xlabel(r"Tolerance $\tau$")
    ax.set_ylabel(r"Accuracy (\%)")
    ax.set_xticks(TOLERANCES)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=6, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    clean_axes(ax)
    save_figure(fig, "figure2c_tolerance")


def pareto(err: pd.DataFrame, latency: pd.DataFrame) -> None:
    """Figure 2d: mean CPU latency against MAE with the Pareto frontier."""
    fig, ax = plt.subplots()
    points = []
    for method in FIGURE_METHODS:
        mae = np.mean(err[method].to_numpy(float))
        mean_time = np.mean(latency[method].to_numpy(float) * 1000)
        ax.scatter(
            mean_time,
            mae,
            marker=marker(method),
            color=color(method),
            s=35,
            zorder=5,
            edgecolors="black",
            linewidths=0.4,
            alpha=0.9,
        )
        points.append((mae, mean_time, figure_label(method)))
    frontier = []
    best = float("inf")
    for mae, mean_time, _ in sorted(points, key=lambda p: p[1]):
        if mae < best:
            frontier.append((mean_time, mae))
            best = mae
    if len(frontier) > 1:
        ax.plot(*zip(*frontier), "--", color="gray", alpha=0.7, linewidth=0.8)
    for mae, mean_time, label in points:
        dx, dy, ha, va = annotation_offset(PARETO_PLACEMENT.get(label, "top-right"))
        nudge_x, nudge_y = PARETO_NUDGE.get(label, (0, 0))
        ax.annotate(
            label,
            (mean_time, mae),
            textcoords="offset points",
            xytext=(dx + nudge_x, dy + nudge_y),
            fontsize=7,
            alpha=0.9,
            ha=ha,
            va=va,
        )
    ax.set_xlabel("Mean time (ms) - log scale")
    ax.set_ylabel(ERROR_LABEL)
    ax.set_xscale("log")
    clean_axes(ax)
    save_figure(fig, "figure2d_pareto")


def oracle_strata(err: pd.DataFrame, oracle: np.ndarray) -> None:
    """Figure 3: mean error per oracle-ARI bin, with ARI below 0.1 in the first bin."""
    bins = np.clip(np.digitize(oracle, ORACLE_BINS) - 1, 0, len(ORACLE_BINS) - 1)
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    for method in FIGURE_METHODS:
        errors_ = err[method].to_numpy(float)
        means = np.array(
            [
                np.nanmean(errors_[bins == b]) if (bins == b).sum() > 0 else np.nan
                for b in range(len(ORACLE_BINS))
            ]
        )
        ours = method == "TabK"
        valid = ~np.isnan(means)
        x_valid, y_valid = ORACLE_BINS[valid], means[valid]
        if len(x_valid) > 3:
            x_smooth = np.linspace(x_valid.min(), x_valid.max(), 300)
            y_smooth = np.clip(PchipInterpolator(x_valid, y_valid)(x_smooth), 0, None)
        else:
            x_smooth, y_smooth = x_valid, y_valid
        ax.plot(
            x_smooth,
            y_smooth,
            color=color(method),
            linewidth=2.5 if ours else 1.0,
            linestyle="-" if ours else "--",
            alpha=1.0 if ours else 0.75,
            zorder=10 if ours else 5,
        )
        ax.plot(
            x_valid,
            y_valid,
            marker=marker(method),
            color=color(method),
            linestyle="None",
            markersize=6 if ours else 3.5,
            alpha=1.0 if ours else 0.75,
            zorder=11 if ours else 6,
        )
        ax.plot(
            [],
            [],
            color=color(method),
            marker=marker(method),
            label=figure_label(method),
            linewidth=2.5 if ours else 1.0,
            linestyle="-" if ours else "--",
            markersize=6 if ours else 3.5,
        )
    ax.set_xlabel("Oracle ARI bin")
    ax.set_ylabel(ERROR_LABEL)
    ax.set_xticks(ORACLE_BINS)
    ax.set_xticklabels([f"{e:.1f}" for e in ORACLE_BINS])
    ax.legend(
        fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, framealpha=0.9
    )
    clean_axes(ax)
    save_figure(fig, "figure3_oracle_ari")


def main() -> None:
    """Write Tables 1 and 12, Figures 2 and 3 and the Section 4.2 numbers."""
    set_style()
    frame = predictions()
    latency = aligned("latency", frame)
    err = errors(frame)
    oracle = ari_at(frame, frame["true_k"].to_numpy(), "k_means")
    table1(frame, latency)
    table12(frame)
    section_numbers(frame)
    boxplot(err)
    ranking(err)
    tolerance(err)
    pareto(err, latency)
    oracle_strata(err, oracle)


if __name__ == "__main__":
    main()
