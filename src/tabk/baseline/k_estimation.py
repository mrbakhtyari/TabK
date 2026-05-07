from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm_fn

from .cnak_k import predict_k_with_cnak
from .cvi_methods import CVI_DIRECTIONS, list_cvi_methods, score_cvi_method
from .gap_statistic import score_from_cached_inertia
from .gmm_bic import score_from_cached_bic
from .imwkmeans_k import predict_k_with_imwkmeans
from .unseen_dcn_k import predict_k_with_unseen_dcn
from .x_means import predict_k_with_xmeans

METHOD_DIRECTIONS: dict[str, str] = {
    **CVI_DIRECTIONS,
    "gap_statistic": "max",
    "gmm_bic": "min",
    "x_means": "max",
    "imwkmeans": "max",
    "cnak": "min",
    "unseen_dcn": "max",
}


def list_k_selection_methods() -> list[str]:
    return sorted(METHOD_DIRECTIONS)


def _label_dataset_name(k_group: h5py.Group) -> str:
    if "predicted_labels" in k_group:
        return "predicted_labels"
    if "labels" in k_group:
        return "labels"
    raise KeyError("No predicted labels dataset found. Expected 'predicted_labels' or 'labels'.")


def _resolve_method_list(methods: list[str] | None) -> list[str]:
    selected = methods or list_k_selection_methods()
    valid = set(list_k_selection_methods())
    unknown = [m for m in selected if m not in valid]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}")
    return selected


def _compute_run_tag(base_h5_path: Path) -> str:
    return base_h5_path.stem


def _get_algorithm_group(ds_group: h5py.Group, algorithm_name: str) -> h5py.Group | None:
    algorithms_group = ds_group.get("algorithms")
    if algorithms_group is None:
        return None

    if algorithm_name in algorithms_group:
        return algorithms_group[algorithm_name]

    for alg_key in algorithms_group.keys():
        alg_group = algorithms_group[alg_key]
        attr_name = str(alg_group.attrs.get("algorithm_name", alg_key))
        if attr_name == algorithm_name:
            return alg_group

    return None


def _read_attr_by_k(
    algo_group: h5py.Group | None,
    k_values: list[int],
    attr_name: str,
) -> dict[int, float | None]:
    values: dict[int, float | None] = {k: None for k in k_values}
    if algo_group is None or "num_clusters" not in algo_group:
        return values

    clusters_group = algo_group["num_clusters"]
    for k in k_values:
        k_key = str(int(k))
        if k_key not in clusters_group:
            continue
        raw_val = float(clusters_group[k_key].attrs.get(attr_name, np.nan))
        values[k] = None if np.isnan(raw_val) else raw_val

    return values


def _select_best_k(method_name: str, score_by_k: dict[int, float]) -> int | None:
    valid_scores = {
        int(k): float(v) for k, v in score_by_k.items() if v is not None and np.isfinite(float(v))
    }
    if not valid_scores:
        return None

    direction = METHOD_DIRECTIONS[method_name]
    if direction == "max":
        return max(valid_scores, key=valid_scores.get)
    return min(valid_scores, key=valid_scores.get)


def _select_gap_statistic_k(
    score_by_k: dict[int, float],
    gap_se_by_k: dict[int, float],
) -> int | None:
    """Select k with the Gap 1-SE rule: smallest k where Gap(k) >= Gap(k+1) - s(k+1)."""
    valid_k = sorted(
        int(k) for k, v in score_by_k.items() if v is not None and np.isfinite(float(v))
    )
    if not valid_k:
        return None
    if len(valid_k) == 1:
        return valid_k[0]

    for idx in range(len(valid_k) - 1):
        k = valid_k[idx]
        next_k = valid_k[idx + 1]

        gap_k = float(score_by_k[k])
        gap_next = float(score_by_k[next_k])

        next_se_raw = gap_se_by_k.get(next_k, np.nan)
        next_se = 0.0
        if next_se_raw is not None and np.isfinite(float(next_se_raw)):
            next_se = float(next_se_raw)

        if gap_k >= (gap_next - next_se):
            return k

    return valid_k[-1]


def _score_cvi_by_k(
    method_name: str,
    X: np.ndarray,
    labels_by_k: dict[int, np.ndarray],
    k_values: list[int],
) -> tuple[dict[int, float], dict[int, float]]:
    score_by_k: dict[int, float] = {}
    elapsed_ms_by_k: dict[int, float] = {}

    for k in k_values:
        start_k = perf_counter()
        cvi_score = score_cvi_method(method_name=method_name, X=X, labels=labels_by_k[k])
        elapsed_ms_by_k[k] = (perf_counter() - start_k) * 1000.0
        score_by_k[k] = cvi_score

    return score_by_k, elapsed_ms_by_k


def _resolve_k_selection_h5_path(base_h5_path: Path, output_h5_path: str | Path | None) -> Path:
    if output_h5_path is not None:
        return Path(output_h5_path)
    return base_h5_path.parent / f"k_selection_{base_h5_path.stem}.h5"


def _write_method_result_group(
    method_group: h5py.Group,
    dataset_name: str,
    method_name: str,
    k_values: list[int],
    score_by_k: dict[int, float],
    base_ari_by_k: dict[int, float | None],
    base_elapsed_ms_by_k: dict[int, float],
    method_elapsed_ms_by_k: dict[int, float],
    pred_k: int | None,
    ari_at_pred_k: float,
    total_time_ms: float,
) -> None:
    for key in list(method_group.keys()):
        del method_group[key]

    k_arr = np.asarray(k_values, dtype=np.int64)
    score_arr = np.asarray([float(score_by_k[k]) for k in k_values], dtype=np.float64)
    ari_arr = np.asarray(
        [np.nan if base_ari_by_k[k] is None else float(base_ari_by_k[k]) for k in k_values],
        dtype=np.float64,
    )
    base_elapsed_arr = np.asarray(
        [float(base_elapsed_ms_by_k[k]) for k in k_values],
        dtype=np.float64,
    )
    method_elapsed_arr = np.asarray(
        [float(method_elapsed_ms_by_k[k]) for k in k_values],
        dtype=np.float64,
    )

    method_group.create_dataset("k_values", data=k_arr)
    method_group.create_dataset("scores", data=score_arr)
    method_group.create_dataset("ari_by_k", data=ari_arr)
    method_group.create_dataset("base_elapsed_ms_by_k", data=base_elapsed_arr)
    method_group.create_dataset("method_elapsed_ms_by_k", data=method_elapsed_arr)

    method_group.attrs["dataset_name"] = str(dataset_name)
    method_group.attrs["method_name"] = str(method_name)
    method_group.attrs["direction"] = str(METHOD_DIRECTIONS[method_name])
    method_group.attrs["pred_k"] = -1 if pred_k is None else int(pred_k)
    method_group.attrs["ari_at_pred_k"] = float(ari_at_pred_k)
    method_group.attrs["total_time_ms"] = float(total_time_ms)


def compute_k_selection_h5_from_base_h5(
    base_h5_path: str | Path,
    output_h5_path: str | Path | None = None,
    methods: list[str] | None = None,
    random_state: int = 42,
    gap_n_refs: int = 8,
    base_algorithm_name: str = "k_means",
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, object]:
    """Compute k-selection results and store incrementally in HDF5.

    overwrite=False skips existing dataset/method results.
    overwrite=True recomputes and rewrites existing dataset/method results.
    """
    base_path = Path(base_h5_path)
    if not base_path.exists():
        raise FileNotFoundError(f"H5 file not found: {base_path}")

    selected_methods = _resolve_method_list(methods)
    run_tag = _compute_run_tag(base_path)
    cvi_methods = set(list_cvi_methods())

    out_h5_path = _resolve_k_selection_h5_path(base_path, output_h5_path)
    out_h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(base_path), "r") as base_h5f, h5py.File(str(out_h5_path), "a") as out_h5f:
        if "datasets" not in base_h5f:
            raise KeyError("Invalid base H5 file: root group 'datasets' is missing")

        out_h5f.attrs["source_base_h5"] = str(base_path)
        out_h5f.attrs["run_tag"] = str(run_tag)
        out_h5f.attrs["base_algorithm_name"] = str(base_algorithm_name)

        out_datasets_group = out_h5f.require_group("datasets")
        base_datasets_group = base_h5f["datasets"]

        for dataset_key in sorted(base_datasets_group.keys()):
            ds_group = base_datasets_group[dataset_key]
            dataset_name = str(ds_group.attrs.get("dataset_name", dataset_key))
            if "X" not in ds_group or "algorithms" not in ds_group:
                continue

            X = np.asarray(ds_group["X"])
            base_algo_group = _get_algorithm_group(ds_group, base_algorithm_name)
            if base_algo_group is None or "num_clusters" not in base_algo_group:
                continue

            base_clusters_group = base_algo_group["num_clusters"]
            k_values = sorted(int(k) for k in base_clusters_group.keys())
            if not k_values:
                continue

            base_elapsed_ms_by_k: dict[int, float] = {}
            base_ari_by_k: dict[int, float | None] = {}
            labels_by_k: dict[int, np.ndarray] = {}
            base_time_total_ms = 0.0

            for k in k_values:
                k_group = base_clusters_group[str(k)]
                label_name = _label_dataset_name(k_group)
                labels_by_k[k] = np.asarray(k_group[label_name])

                elapsed_ms = float(k_group.attrs.get("elapsed_seconds", np.nan)) * 1000.0
                base_elapsed_ms_by_k[k] = elapsed_ms
                if np.isfinite(elapsed_ms):
                    base_time_total_ms += elapsed_ms

                ari_raw = float(k_group.attrs.get("ari", np.nan))
                base_ari_by_k[k] = None if np.isnan(ari_raw) else ari_raw

            kmeans_group = _get_algorithm_group(ds_group, "k_means")
            gmm_group = _get_algorithm_group(ds_group, "gmm")
            inertia_by_k = _read_attr_by_k(kmeans_group, k_values, "inertia")
            bic_by_k = _read_attr_by_k(gmm_group, k_values, "bic")

            out_dataset_group = out_datasets_group.require_group(str(dataset_key))
            out_dataset_group.attrs["dataset_name"] = str(dataset_name)
            methods_group = out_dataset_group.require_group("methods")

            method_iterator = (
                tqdm_fn(selected_methods, desc=f"{dataset_name}: methods", leave=False)
                if show_progress and tqdm_fn
                else selected_methods
            )

            for method_name in method_iterator:
                method_group = methods_group.require_group(method_name)
                if not overwrite and "k_values" in method_group:
                    continue

                gap_se_by_k: dict[int, float] = {}

                if method_name in cvi_methods:
                    score_by_k, method_elapsed_ms_by_k = _score_cvi_by_k(
                        method_name=method_name,
                        X=X,
                        labels_by_k=labels_by_k,
                        k_values=k_values,
                    )
                elif method_name == "gap_statistic":
                    score_by_k, method_elapsed_ms_by_k, gap_se_by_k = score_from_cached_inertia(
                        X=X,
                        inertia_by_k=inertia_by_k,
                        k_values=k_values,
                        random_state=random_state,
                        n_refs=gap_n_refs,
                    )
                elif method_name == "gmm_bic":
                    score_by_k, method_elapsed_ms_by_k = score_from_cached_bic(
                        bic_by_k=bic_by_k,
                        k_values=k_values,
                    )
                elif method_name == "x_means":
                    predicted_k, elapsed_ms = predict_k_with_xmeans(
                        X=X,
                        k_values=k_values,
                        random_state=random_state,
                    )
                    score_by_k = {k: np.nan for k in k_values}
                    method_elapsed_ms_by_k = {k: 0.0 for k in k_values}
                    if predicted_k is not None:
                        score_by_k[predicted_k] = 1.0
                        method_elapsed_ms_by_k[predicted_k] = float(elapsed_ms)
                elif method_name == "imwkmeans":
                    predicted_k, elapsed_ms = predict_k_with_imwkmeans(
                        X=X,
                        k_values=k_values,
                        random_state=random_state,
                    )
                    score_by_k = {k: np.nan for k in k_values}
                    method_elapsed_ms_by_k = {k: 0.0 for k in k_values}
                    if predicted_k is not None:
                        score_by_k[predicted_k] = 1.0
                        method_elapsed_ms_by_k[predicted_k] = float(elapsed_ms)
                elif method_name == "cnak":
                    predicted_k, score_by_k, method_elapsed_ms_by_k = predict_k_with_cnak(
                        X=X,
                        k_values=k_values,
                        random_state=random_state,
                    )
                    for k in k_values:
                        if k not in score_by_k:
                            score_by_k[k] = np.nan
                        if k not in method_elapsed_ms_by_k:
                            method_elapsed_ms_by_k[k] = 0.0
                elif method_name == "unseen_dcn":
                    predicted_k, elapsed_ms = predict_k_with_unseen_dcn(
                        X=X,
                        random_state=random_state,
                    )
                    score_by_k = {k: np.nan for k in k_values}
                    method_elapsed_ms_by_k = {k: 0.0 for k in k_values}
                    if predicted_k is not None:
                        score_by_k[predicted_k] = 1.0
                        method_elapsed_ms_by_k[predicted_k] = float(elapsed_ms)
                        if predicted_k not in base_ari_by_k:
                            base_ari_by_k[predicted_k] = None
                else:
                    raise ValueError(f"Unsupported method: {method_name}")

                if method_name == "gap_statistic":
                    pred_k = _select_gap_statistic_k(
                        score_by_k=score_by_k,
                        gap_se_by_k=gap_se_by_k,
                    )
                else:
                    pred_k = _select_best_k(method_name, score_by_k)
                ari_at_pred_k = (
                    0.0
                    if pred_k is None
                    else (np.nan if base_ari_by_k[pred_k] is None else float(base_ari_by_k[pred_k]))
                )
                total_time_ms = float(base_time_total_ms + sum(method_elapsed_ms_by_k.values()))

                _write_method_result_group(
                    method_group=method_group,
                    dataset_name=dataset_name,
                    method_name=method_name,
                    k_values=k_values,
                    score_by_k=score_by_k,
                    base_ari_by_k=base_ari_by_k,
                    base_elapsed_ms_by_k=base_elapsed_ms_by_k,
                    method_elapsed_ms_by_k=method_elapsed_ms_by_k,
                    pred_k=pred_k,
                    ari_at_pred_k=ari_at_pred_k,
                    total_time_ms=total_time_ms,
                )

    return {
        "run_tag": run_tag,
        "k_selection_h5": str(out_h5_path),
    }


def export_k_selection_csv_from_h5(
    k_selection_h5_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Export detailed per-method CSVs and wide summary CSV from k-selection HDF5."""
    ksel_h5_path = Path(k_selection_h5_path)
    if not ksel_h5_path.exists():
        raise FileNotFoundError(f"K-selection H5 file not found: {ksel_h5_path}")

    run_tag = str(ksel_h5_path.stem)
    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else ksel_h5_path.parent / "detailed_baseline_reports"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    detailed_rows_by_method: dict[str, list[dict[str, object]]] = {}
    summary_rows: list[dict[str, object]] = []

    with h5py.File(str(ksel_h5_path), "r") as h5f:
        if "datasets" not in h5f:
            raise KeyError("Invalid k-selection H5 file: root group 'datasets' is missing")

        datasets_group = h5f["datasets"]
        for dataset_key in sorted(datasets_group.keys()):
            ds_group = datasets_group[dataset_key]
            dataset_name = str(ds_group.attrs.get("dataset_name", dataset_key))
            methods_group = ds_group.get("methods")
            if methods_group is None:
                continue

            for method_name in sorted(methods_group.keys()):
                method_group = methods_group[method_name]
                if "k_values" not in method_group or "scores" not in method_group:
                    continue

                k_values = np.asarray(method_group["k_values"], dtype=np.int64)
                scores = np.asarray(method_group["scores"], dtype=np.float64)
                ari_by_k = np.asarray(method_group["ari_by_k"], dtype=np.float64)
                base_elapsed = np.asarray(method_group["base_elapsed_ms_by_k"], dtype=np.float64)
                method_elapsed = np.asarray(
                    method_group["method_elapsed_ms_by_k"],
                    dtype=np.float64,
                )

                pred_k = int(method_group.attrs.get("pred_k", -1))
                detailed_row: dict[str, object] = {
                    "dataset": dataset_name,
                    "pred_k": np.nan if pred_k < 0 else pred_k,
                    "ari_at_pred_k": float(method_group.attrs.get("ari_at_pred_k", np.nan)),
                    "total_time_ms": float(method_group.attrs.get("total_time_ms", np.nan)),
                }
                for i, k in enumerate(k_values.tolist()):
                    detailed_row[f"{k}_score"] = (
                        float(scores[i]) if np.isfinite(scores[i]) else np.nan
                    )
                    detailed_row[f"{k}_ari"] = (
                        float(ari_by_k[i]) if np.isfinite(ari_by_k[i]) else np.nan
                    )
                    detailed_row[f"{k}_base_elapsed_ms"] = float(base_elapsed[i])
                    detailed_row[f"{k}_method_elapsed_ms"] = float(method_elapsed[i])

                detailed_rows_by_method.setdefault(method_name, []).append(detailed_row)
                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "method": method_name,
                        "pred_k": pred_k,
                        "ari_at_pred_k": float(method_group.attrs.get("ari_at_pred_k", np.nan)),
                        "total_time_ms": float(method_group.attrs.get("total_time_ms", np.nan)),
                    }
                )

    detail_paths_by_method: dict[str, str] = {}
    for method_name, rows in detailed_rows_by_method.items():
        method_df = pd.DataFrame(rows).sort_values(["dataset"]).reset_index(drop=True)

        ordered_columns = ["dataset", "pred_k", "ari_at_pred_k", "total_time_ms"]
        if not method_df.empty:
            k_values_seen: set[int] = set()
            for col in method_df.columns:
                if col in {"dataset", "pred_k", "ari_at_pred_k", "total_time_ms"}:
                    continue
                if "_" not in col:
                    continue
                k_str = col.split("_", maxsplit=1)[0]
                if k_str.isdigit():
                    k_values_seen.add(int(k_str))

            for k in sorted(k_values_seen):
                ordered_columns.extend(
                    [
                        f"{k}_score",
                        f"{k}_ari",
                        f"{k}_base_elapsed_ms",
                        f"{k}_method_elapsed_ms",
                    ]
                )
            method_df = method_df[[col for col in ordered_columns if col in method_df.columns]]

        method_path = out_dir / f"{method_name}_{run_tag}.csv"
        method_df.to_csv(method_path, index=False)
        detail_paths_by_method[method_name] = str(method_path)

    summary_df = (
        pd.DataFrame(summary_rows).sort_values(["dataset", "method"]).reset_index(drop=True)
    )
    summary_wide_path = out_dir / f"k_selection_summary_{run_tag}.csv"

    if summary_df.empty:
        pd.DataFrame().to_csv(summary_wide_path, index=False)
    else:
        index_cols = ["dataset"]
        wide_df = summary_df[["dataset"]].drop_duplicates(subset=index_cols).set_index(index_cols)

        metric_cols = ["pred_k", "total_time_ms", "ari_at_pred_k"]
        for metric_name in metric_cols:
            pivot = summary_df.pivot_table(
                index=index_cols,
                columns="method",
                values=metric_name,
                aggfunc="first",
            )
            if not pivot.empty:
                pivot.columns = [f"{str(method)}_{metric_name}" for method in pivot.columns]
                wide_df = wide_df.join(pivot, how="left")

        wide_out = wide_df.reset_index()
        present_methods = sorted(summary_df["method"].unique())

        ordered_cols = ["dataset"]
        for method_name in present_methods:
            ordered_cols.extend(
                [
                    f"{method_name}_pred_k",
                    f"{method_name}_total_time_ms",
                    f"{method_name}_ari_at_pred_k",
                ]
            )

        for col in ordered_cols:
            if col not in wide_out.columns:
                wide_out[col] = np.nan

        wide_out = wide_out[ordered_cols]
        wide_out.to_csv(summary_wide_path, index=False)

    return {
        "run_tag": run_tag,
        "summary_wide_csv": str(summary_wide_path),
        "detailed_csv_by_method": detail_paths_by_method,
    }


def compute_k_selection_reports_from_h5(
    base_h5_path: str | Path,
    output_dir: str | Path | None = None,
    methods: list[str] | None = None,
    random_state: int = 42,
    gap_n_refs: int = 8,
    base_algorithm_name: str = "k_means",
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, object]:
    """Backward-compatible wrapper: compute k-selection H5 then export CSV reports."""
    h5_info = compute_k_selection_h5_from_base_h5(
        base_h5_path=base_h5_path,
        output_h5_path=None,
        methods=methods,
        random_state=random_state,
        gap_n_refs=gap_n_refs,
        base_algorithm_name=base_algorithm_name,
        overwrite=overwrite,
        show_progress=show_progress,
    )

    report_paths = export_k_selection_csv_from_h5(
        k_selection_h5_path=h5_info["k_selection_h5"],
        output_dir=output_dir,
    )
    return {
        **h5_info,
        **report_paths,
    }
