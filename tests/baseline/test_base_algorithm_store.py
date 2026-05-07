import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

from tabk.baseline import (
    ALGORITHMS,
    compute_k_values,
    evaluate_and_save,
    load_labels,
    load_run_metrics,
    register_algorithm,
)


def test_compute_k_values_is_inclusive():
    assert compute_k_values(2, 5) == [2, 3, 4, 5]


def test_evaluate_and_save_persists_nested_results(tmp_path: Path):
    X, y = make_blobs(n_samples=120, centers=3, random_state=7)
    h5_path = tmp_path / "cluster_results.h5"

    evaluate_and_save(
        X=X,
        y=y,
        dataset_name="toy/set 1",
        k_min=2,
        k_max=4,
        h5_path=h5_path,
        algorithms=["k_means"],
        algorithm_kwargs={"k_means": {"random_state": 0, "n_init": 10}},
        show_progress=True,
    )

    labels_k3 = load_labels(h5_path, "toy/set 1", "k_means", 3)
    run_k3 = load_run_metrics(h5_path, "toy/set 1", "k_means", 3)

    assert labels_k3.shape == (X.shape[0],)
    assert run_k3["ari"] is not None
    assert -1.0 <= float(run_k3["ari"]) <= 1.0
    assert float(run_k3["elapsed_seconds"]) >= 0.0

    with h5py.File(str(h5_path), "r") as h5f:
        dataset_group = h5f["datasets"]["toy%2Fset%201"]
        np.testing.assert_allclose(dataset_group["X"][:], X)
        np.testing.assert_array_equal(dataset_group["y"][:], y)
        assert (
            json.loads(dataset_group["algorithms"]["k_means"].attrs["algorithm_params_json"])[
                "random_state"
            ]
            == 0
        )


def test_evaluate_and_save_without_ground_truth_sets_ari_to_none(tmp_path: Path):
    X, _ = make_blobs(n_samples=64, centers=2, random_state=11)
    h5_path = tmp_path / "cluster_results.h5"

    evaluate_and_save(
        X=X,
        y=None,
        dataset_name="no-labels",
        k_min=2,
        k_max=2,
        h5_path=h5_path,
        algorithms=["k_means"],
    )

    run = load_run_metrics(h5_path, "no-labels", "k_means", 2)
    assert run["ari"] is None


def test_custom_algorithm_registration(tmp_path: Path):
    name = "agglomerative"

    def _make_agglomerative(n_clusters: int, **kwargs):
        return AgglomerativeClustering(n_clusters=n_clusters, **kwargs)

    previous = ALGORITHMS.get(name)
    register_algorithm(name, _make_agglomerative)

    X, _ = make_blobs(n_samples=80, centers=3, random_state=2)
    results = evaluate_and_save(
        X=X,
        y=None,
        dataset_name="agglo_test",
        k_min=3,
        k_max=3,
        h5_path=tmp_path / "cluster_results.h5",
        algorithms=[name],
    )
    labels = results[name][3]["predicted_labels"]
    assert labels.shape == (80,)

    if previous is None:
        ALGORITHMS.pop(name, None)
    else:
        ALGORITHMS[name] = previous
