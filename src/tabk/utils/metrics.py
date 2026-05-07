from collections.abc import Callable

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def dunn_score(data: np.ndarray, labels: np.ndarray, metric: str | Callable = "euclidean") -> float:
    """Dunn index: min(inter-cluster distance) / max(intra-cluster diameter).

    Higher values indicate better-separated, compact clusters.
    """
    distances = pairwise_distances(data, metric=metric)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        raise ValueError("Dunn Index is undefined for fewer than 2 clusters.")
    if distances.shape[0] != len(labels):
        raise ValueError("Labels and distance matrix shape mismatch.")

    # Precompute cluster member indices
    cluster_indices = {c: np.where(labels == c)[0] for c in unique_labels}

    # Min inter-cluster distance (vectorized over cluster pairs)
    min_intercluster = np.inf
    for i in range(n_clusters):
        idx_i = cluster_indices[unique_labels[i]]
        for j in range(i + 1, n_clusters):
            idx_j = cluster_indices[unique_labels[j]]
            inter_dists = distances[np.ix_(idx_i, idx_j)]
            min_intercluster = min(min_intercluster, float(inter_dists.min()))

    # Max intra-cluster diameter
    max_diameter = 0.0
    for c in unique_labels:
        idx = cluster_indices[c]
        if len(idx) < 2:
            continue
        intra_dists = distances[np.ix_(idx, idx)]
        max_diameter = max(max_diameter, float(intra_dists.max()))

    if max_diameter == 0:
        raise ValueError("Max diameter is zero. Likely all clusters are singletons.")

    return min_intercluster / max_diameter


# Example usage:
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]])
    labels = np.array([0, 0, 0, 1, 1, 1])
    score = dunn_score(X, labels)
    print(score)  # Should print a Dunn index value, 9.51
