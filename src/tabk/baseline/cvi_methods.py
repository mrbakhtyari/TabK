from collections.abc import Callable
from typing import Literal

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from tabk.utils import dunn_score

CVIDirection = Literal["min", "max"]

CVI_SCORERS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "davies_bouldin": davies_bouldin_score,
    "calinski_harabasz": calinski_harabasz_score,
    "silhouette": silhouette_score,
    "dunn": dunn_score,
}

CVI_DIRECTIONS: dict[str, CVIDirection] = {
    "davies_bouldin": "min",
    "calinski_harabasz": "max",
    "silhouette": "max",
    "dunn": "max",
}


def list_cvi_methods() -> list[str]:
    return sorted(CVI_SCORERS)


def score_cvi_method(
    method_name: str,
    X: np.ndarray,
    labels: np.ndarray,
) -> float:
    if method_name not in CVI_SCORERS:
        supported = ", ".join(list_cvi_methods())
        raise ValueError(f"Unsupported CVI method: {method_name}. Supported: {supported}")

    labels_arr = np.asarray(labels)
    if labels_arr.ndim != 1:
        raise ValueError("labels must be a 1D array")

    unique_count = int(np.unique(labels_arr).size)
    if unique_count < 2:
        return float("nan")

    try:
        score_val = float(CVI_SCORERS[method_name](X, labels_arr))
    except Exception:
        return float("nan")

    return score_val if np.isfinite(score_val) else float("nan")
