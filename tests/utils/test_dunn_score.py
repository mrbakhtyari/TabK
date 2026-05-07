import numpy as np
import pytest

from tabk.utils import dunn_score


def test_two_well_separated_clusters():
    # Two clear clusters in 2D
    cluster1 = np.random.randn(10, 2) + [5, 5]
    cluster2 = np.random.randn(10, 2) + [-5, -5]
    data = np.vstack([cluster1, cluster2])
    labels = np.array([0] * 10 + [1] * 10)
    score = dunn_score(data, labels)
    assert score > 1.0


def test_single_cluster_error():
    data = np.random.randn(10, 2)
    labels = np.zeros(10)
    with pytest.raises(ValueError):
        dunn_score(data, labels)


def test_zero_diameter_error():
    data = np.array([[0, 0], [0, 0], [10, 10]])
    labels = np.array([0, 0, 1])  # cluster 0 has only identical points
    with pytest.raises(ValueError, match="Max diameter is zero"):
        dunn_score(data, labels)


def test_three_clusters_simple():
    data = np.array(
        [
            [0, 0],
            [1, 1],  # Cluster 0
            [10, 10],
            [11, 11],  # Cluster 1
            [20, 0],
            [21, 1],  # Cluster 2
        ]
    )
    labels = np.array([0, 0, 1, 1, 2, 2])
    score = dunn_score(data, labels)
    assert 0 < score < 10
