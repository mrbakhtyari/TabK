import warnings
from time import perf_counter

import numpy as np
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import splitting_type, xmeans

np.warnings = warnings  # Fix for the AttributeError


METHOD_NAME = "x_means"
DIRECTION = "max"


def predict_k_with_xmeans(
    X: np.ndarray,
    k_values: list[int],
    random_state: int,
) -> tuple[int | None, float]:
    """Predict the number of clusters with X-Means within the provided k range."""

    if not k_values:
        return None, 0.0

    k_min = int(min(k_values))
    k_max = int(max(k_values))
    initial_k = max(2, k_min)

    X_arr = np.asarray(X, dtype=np.float64)
    start = perf_counter()

    try:
        initializer = kmeans_plusplus_initializer(
            X_arr.tolist(),
            amount_centers=initial_k,
            random_state=random_state,
        )
        initial_centers = initializer.initialize()

        model = xmeans(
            data=X_arr.tolist(),
            initial_centers=initial_centers,
            kmax=k_max,
            criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION,
            ccore=True,
        )
        model.process()
        predicted_k = int(len(model.get_clusters()))
    except Exception:
        predicted_k = None

    elapsed_ms = (perf_counter() - start) * 1000.0

    if predicted_k is None or predicted_k not in set(k_values):
        return None, elapsed_ms
    return predicted_k, elapsed_ms
