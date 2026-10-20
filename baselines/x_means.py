import warnings
from time import perf_counter
from typing import cast

import numpy as np
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import splitting_type, xmeans

# pyclustering accesses the warnings module through NumPy.
if not hasattr(np, "warnings"):
    setattr(np, "warnings", warnings)


def predict_k(X: np.ndarray, k_values: list[int]) -> tuple[int | None, float]:
    """Predict k with X-Means, returning None on failure or outside the range."""
    if not k_values:
        return None, 0.0
    initial_k = max(2, int(min(k_values)))
    k_max = int(max(k_values))
    X = np.asarray(X, dtype=np.float64)
    start = perf_counter()
    try:
        initializer = kmeans_plusplus_initializer(
            X.tolist(), amount_centers=initial_k, random_state=42
        )
        initial_centers = initializer.initialize()
        model = xmeans(
            data=X.tolist(),
            initial_centers=initial_centers,
            kmax=k_max,
            criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION,
            ccore=True,
        )
        model.process()
        clusters = cast(list[list[int]], model.get_clusters())
        predicted_k = int(len(clusters))
    except Exception:
        predicted_k = None
    elapsed = perf_counter() - start
    if predicted_k is None or predicted_k not in set(k_values):
        return None, elapsed
    return predicted_k, elapsed
