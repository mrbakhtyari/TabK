from time import perf_counter

import numpy as np

from .kmeans_grid import fit_kmeans


def gap_scores(
    X: np.ndarray, inertia_by_k: dict[int, float | None], k_values: list[int]
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute Gap scores and standard errors from eight uniform references."""
    X = np.asarray(X)
    rng = np.random.default_rng(42)
    feature_min = np.min(X, axis=0)
    feature_max = np.max(X, axis=0)
    scores: dict[int, float] = {}
    standard_errors: dict[int, float] = {}
    n_refs = 8
    for k in k_values:
        observed_inertia = inertia_by_k.get(k)
        if observed_inertia is None or not np.isfinite(float(observed_inertia)):
            scores[k] = float("nan")
            standard_errors[k] = float("nan")
            continue
        observed_log_wk = float(np.log(max(float(observed_inertia), 1e-12)))
        reference_log_wk: list[float] = []
        for ref_idx in range(n_refs):
            X_ref = rng.uniform(feature_min, feature_max, size=X.shape).astype(np.float32)
            _, inertia = fit_kmeans(X_ref, int(k), n_init=10, seed=42 + ref_idx + 1)
            reference_log_wk.append(float(np.log(max(inertia, 1e-12))))
        reference_log_wk_arr = np.asarray(reference_log_wk, dtype=float)
        scores[k] = float(np.mean(reference_log_wk_arr) - observed_log_wk)
        std_ddof = 1 if reference_log_wk_arr.size > 1 else 0
        std_ref = float(np.std(reference_log_wk_arr, ddof=std_ddof))
        standard_errors[k] = float(std_ref * np.sqrt(1.0 + (1.0 / max(int(n_refs), 1))))
    return scores, standard_errors


def select_gap_k(scores: dict[int, float], standard_errors: dict[int, float]) -> int | None:
    """Apply the 1-SE rule, falling back to the largest valid k."""
    valid_k = sorted(int(k) for k, v in scores.items() if v is not None and np.isfinite(v))
    if not valid_k:
        return None
    for k, next_k in zip(valid_k, valid_k[1:]):
        next_se_raw = standard_errors.get(next_k, np.nan)
        next_se = 0.0
        if next_se_raw is not None and np.isfinite(float(next_se_raw)):
            next_se = float(next_se_raw)
        if float(scores[k]) >= float(scores[next_k]) - next_se:
            return k
    return valid_k[-1]


def predict_k(
    X: np.ndarray, inertia_by_k: dict[int, float | None], k_values: list[int]
) -> tuple[int | None, float]:
    """Select k by Gap using shared observed inertia."""
    start = perf_counter()
    scores, standard_errors = gap_scores(X, inertia_by_k, k_values)
    return select_gap_k(scores, standard_errors), perf_counter() - start
