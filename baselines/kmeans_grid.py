from time import perf_counter

import numpy as np
from sklearn.cluster import KMeans


def fit_kmeans(X: np.ndarray, k: int, n_init: int, seed: int) -> tuple[np.ndarray, float]:
    """Fit k-means with k-means++ initialization and return its labels and inertia."""
    # sklearn's untyped defaults make pyright infer n_init as str and inertia_ as None
    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed)  # pyright: ignore[reportArgumentType]
    labels = np.asarray(model.fit_predict(X))
    return labels, float(model.inertia_)  # pyright: ignore[reportArgumentType]


def fit_kmeans_grid(
    X: np.ndarray, k_values: list[int]
) -> tuple[dict[int, np.ndarray], dict[int, float], float]:
    """Fit shared k-means labels and inertia, returning elapsed seconds."""
    labels: dict[int, np.ndarray] = {}
    inertia: dict[int, float] = {}
    start = perf_counter()
    for k in k_values:
        labels[k], inertia[k] = fit_kmeans(X, k, n_init=100, seed=42)
    return labels, inertia, perf_counter() - start
