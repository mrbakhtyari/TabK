"""Classical cluster cardinality estimators."""

import numpy as np

from . import cvi, gap, gmm_bic, imwk, x_means
from .kmeans_grid import fit_kmeans_grid

METHODS = (*cvi.SCORERS, "gap_statistic", "gmm_bic", "x_means", "imwkmeans")


def estimate_k(
    X: np.ndarray, k_values: list[int], methods: list[str] | None = None
) -> dict[str, tuple[int | None, float]]:
    """Return predictions and seconds per method, sharing the k-means grid.

    X must already be preprocessed. Each CVI and Gap time includes the shared
    grid fit; other methods include their own fitting time.
    """
    selected = list(METHODS) if methods is None else methods
    unknown = sorted(set(selected) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    k_values = sorted(set(k_values))
    if not k_values:
        return {method: (None, 0.0) for method in selected}
    if k_values[0] < 2:
        raise ValueError("k values must be >= 2")
    X = np.asarray(X)
    labels: dict[int, np.ndarray] = {}
    inertia: dict[int, float | None] = {}
    grid_seconds = 0.0
    if any(method in cvi.SCORERS or method == "gap_statistic" for method in selected):
        labels, fitted_inertia, grid_seconds = fit_kmeans_grid(X, k_values)
        inertia.update(fitted_inertia)

    results: dict[str, tuple[int | None, float]] = {}
    for method in selected:
        if method in cvi.SCORERS:
            predicted_k, seconds = cvi.predict_k(method, X, labels, k_values)
            results[method] = predicted_k, grid_seconds + seconds
        elif method == "gap_statistic":
            predicted_k, seconds = gap.predict_k(X, inertia, k_values)
            results[method] = predicted_k, grid_seconds + seconds
        elif method == "gmm_bic":
            results[method] = gmm_bic.predict_k(X, k_values)
        elif method == "x_means":
            results[method] = x_means.predict_k(X, k_values)
        elif method == "imwkmeans":
            results[method] = imwk.predict_k(X, k_values)
    return results
