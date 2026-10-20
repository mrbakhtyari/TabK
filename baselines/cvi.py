from collections.abc import Callable
from time import perf_counter

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)


def dunn_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Return minimum cluster separation divided by maximum diameter."""
    distances = pairwise_distances(X, metric="euclidean")
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters < 2:
        raise ValueError("Dunn index is undefined for fewer than 2 clusters")
    if distances.shape[0] != len(labels):
        raise ValueError("Labels and distance matrix shape mismatch")

    cluster_indices = {c: np.where(labels == c)[0] for c in unique_labels}
    min_intercluster = np.inf
    for i in range(n_clusters):
        idx_i = cluster_indices[unique_labels[i]]
        for j in range(i + 1, n_clusters):
            idx_j = cluster_indices[unique_labels[j]]
            inter_dists = distances[np.ix_(idx_i, idx_j)]
            min_intercluster = min(min_intercluster, float(inter_dists.min()))

    max_diameter = 0.0
    for c in unique_labels:
        idx = cluster_indices[c]
        if len(idx) < 2:
            continue
        intra_dists = distances[np.ix_(idx, idx)]
        max_diameter = max(max_diameter, float(intra_dists.max()))
    if max_diameter == 0:
        raise ValueError("Maximum cluster diameter is zero")
    return min_intercluster / max_diameter


SCORERS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "silhouette": silhouette_score,
    "calinski_harabasz": calinski_harabasz_score,
    "davies_bouldin": davies_bouldin_score,
    "dunn": dunn_score,
}


def select_best_k(scores: dict[int, float], *, minimize: bool = False) -> int | None:
    """Select the first finite optimum in iteration order."""
    valid = {int(k): float(v) for k, v in scores.items() if v is not None and np.isfinite(v)}
    if not valid:
        return None
    return min(valid, key=valid.__getitem__) if minimize else max(valid, key=valid.__getitem__)


def score_cvi(method: str, X: np.ndarray, labels: np.ndarray) -> float:
    """Score labels, returning NaN for undefined or failed scores."""
    scorer = SCORERS[method]
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if np.unique(labels).size < 2:
        return float("nan")
    try:
        score = float(scorer(X, labels))
    except Exception:
        return float("nan")
    return score if np.isfinite(score) else float("nan")


def predict_k(
    method: str, X: np.ndarray, labels_by_k: dict[int, np.ndarray], k_values: list[int]
) -> tuple[int | None, float]:
    """Select k using a CVI on shared k-means labels."""
    start = perf_counter()
    scores = {k: score_cvi(method, X, labels_by_k[k]) for k in k_values}
    predicted_k = select_best_k(scores, minimize=method == "davies_bouldin")
    return predicted_k, perf_counter() - start
