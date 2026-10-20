from time import perf_counter

import numpy as np
from sklearn.mixture import GaussianMixture

from .cvi import select_best_k


def predict_k(X: np.ndarray, k_values: list[int]) -> tuple[int | None, float]:
    """Select k by minimum Gaussian mixture BIC."""
    start = perf_counter()
    scores: dict[int, float] = {}
    for k in k_values:
        model = GaussianMixture(n_components=k, n_init=100, random_state=42)
        model.fit_predict(X)
        scores[k] = float(model.bic(X))
    return select_best_k(scores, minimize=True), perf_counter() - start
