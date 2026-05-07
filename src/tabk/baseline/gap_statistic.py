from time import perf_counter

import numpy as np
from sklearn.cluster import KMeans

METHOD_NAME = "gap_statistic"
DIRECTION = "max"


def score_from_cached_inertia(
    X: np.ndarray,
    inertia_by_k: dict[int, float | None],
    k_values: list[int],
    random_state: int,
    n_refs: int,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """Compute Gap scores and per-k standard errors using cached observed inertia."""
    X_arr = np.asarray(X)
    rng = np.random.default_rng(random_state)
    feature_min = np.min(X_arr, axis=0)
    feature_max = np.max(X_arr, axis=0)

    score_by_k: dict[int, float] = {}
    elapsed_ms_by_k: dict[int, float] = {}
    gap_se_by_k: dict[int, float] = {}

    for k in k_values:
        start_k = perf_counter()

        observed_inertia = inertia_by_k.get(k)
        if observed_inertia is None or not np.isfinite(float(observed_inertia)):
            score_by_k[k] = float("nan")
            elapsed_ms_by_k[k] = (perf_counter() - start_k) * 1000.0
            gap_se_by_k[k] = float("nan")
            continue

        observed_log_wk = float(np.log(max(float(observed_inertia), 1e-12)))

        reference_log_wk: list[float] = []
        for ref_idx in range(n_refs):
            X_ref = rng.uniform(feature_min, feature_max, size=X_arr.shape).astype(np.float32)
            model = KMeans(n_clusters=int(k), n_init=10, random_state=random_state + ref_idx + 1)
            model.fit(X_ref)
            reference_log_wk.append(float(np.log(max(float(model.inertia_), 1e-12))))

        reference_log_wk_arr = np.asarray(reference_log_wk, dtype=float)
        score_by_k[k] = float(np.mean(reference_log_wk_arr) - observed_log_wk)

        # Tibshirani et al. Gap Statistic uncertainty term for the 1-SE selection rule.
        std_ddof = 1 if reference_log_wk_arr.size > 1 else 0
        std_ref = float(np.std(reference_log_wk_arr, ddof=std_ddof))
        gap_se_by_k[k] = float(std_ref * np.sqrt(1.0 + (1.0 / max(int(n_refs), 1))))

        elapsed_ms_by_k[k] = (perf_counter() - start_k) * 1000.0

    return score_by_k, elapsed_ms_by_k, gap_se_by_k
