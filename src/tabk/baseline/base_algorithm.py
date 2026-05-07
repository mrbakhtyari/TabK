import json
import logging
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import quote

import h5py
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm as tqdm_fn

logger = logging.getLogger(__name__)

AlgorithmFactory = Callable[..., Any]

# --- Algorithm Registry ---


def _make_kmeans(n_clusters: int, **kwargs: Any) -> Any:
    return KMeans(n_clusters=n_clusters, **kwargs)


def _make_gmm(n_clusters: int, **kwargs: Any) -> Any:
    # GaussianMixture uses n_components instead of n_clusters
    return GaussianMixture(n_components=n_clusters, **kwargs)


def _make_spectral(n_clusters: int, **kwargs: Any) -> Any:
    return SpectralClustering(n_clusters=n_clusters, **kwargs)


ALGORITHMS: dict[str, AlgorithmFactory] = {
    "k_means": _make_kmeans,
    "gmm": _make_gmm,
    "spectral": _make_spectral,
}


def register_algorithm(name: str, factory: AlgorithmFactory) -> None:
    """Register a new method."""
    if not name:
        raise ValueError("Algorithm name must be a non-empty string")
    if not callable(factory):
        raise TypeError("Algorithm factory must be callable")
    ALGORITHMS[name] = factory


# --- Internal Helpers ---


def compute_k_values(k_min: int, k_max: int) -> list[int]:
    if k_min < 2:
        raise ValueError("k_min must be >= 2")
    if k_max < k_min:
        raise ValueError("k_max must be >= k_min")
    return list(range(int(k_min), int(k_max) + 1))


def _to_h5_key(name: str) -> str:
    return quote(name, safe="._-")


def _fit_predict(
    X: np.ndarray, algorithm_name: str, n_clusters: int, **kwargs: Any
) -> tuple[np.ndarray, dict[str, float]]:
    """Fits model and returns labels plus internal metrics (BIC, Inertia)."""
    if algorithm_name not in ALGORITHMS:
        supported = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unsupported algorithm: {algorithm_name}. Supported: {supported}")

    estimator = ALGORITHMS[algorithm_name](n_clusters=n_clusters, **kwargs)

    # Standard fit/predict
    if hasattr(estimator, "fit_predict"):
        labels = np.asarray(estimator.fit_predict(X))
    else:
        estimator.fit(X)
        labels = np.asarray(
            estimator.labels_ if hasattr(estimator, "labels_") else estimator.predict(X)
        )

    # Collect internal metrics for K-selection (BIC for GMM, Inertia for KMeans)
    metrics = {}
    if hasattr(estimator, "bic"):
        metrics["bic"] = float(estimator.bic(X))
    if hasattr(estimator, "inertia_"):
        metrics["inertia"] = float(estimator.inertia_)

    return labels, metrics


# --- Main Execution Engine ---


def evaluate_and_save(
    X: np.ndarray,
    y: np.ndarray | None,
    dataset_name: str,
    k_min: int,
    k_max: int,
    h5_path: str | Path,
    algorithms: list[str] | None = None,
    algorithm_kwargs: dict[str, dict[str, Any]] | None = None,
    overwrite: bool = False,
) -> None:
    """
    Runs clustering and saves to HDF5.
    Skips existing entries unless overwrite=True.
    """
    X_arr = np.asarray(X)
    y_arr = None if y is None else np.asarray(y)
    k_values = compute_k_values(k_min=k_min, k_max=k_max)
    selected_algorithms = algorithms or list(ALGORITHMS)
    kwargs_by_algorithm = algorithm_kwargs or {}

    h5_file = Path(h5_path)
    h5_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(h5_file), "a") as h5f:
        # 1. Ensure Dataset Group exists
        ds_key = _to_h5_key(dataset_name)
        dataset_group = h5f.require_group("datasets").require_group(ds_key)

        # Save X and y if not present
        if "X" not in dataset_group:
            dataset_group.create_dataset("X", data=X_arr)
        if y_arr is not None and "y" not in dataset_group:
            dataset_group.create_dataset("y", data=y_arr)

        algorithms_group = dataset_group.require_group("algorithms")

        for alg_name in selected_algorithms:
            run_kwargs = kwargs_by_algorithm.get(alg_name, {})
            alg_group = algorithms_group.require_group(_to_h5_key(alg_name))
            clusters_group = alg_group.require_group("num_clusters")

            # Persist algorithm metadata for downstream readers/reports.
            alg_group.attrs["algorithm_name"] = str(alg_name)
            alg_group.attrs["algorithm_params_json"] = json.dumps(
                run_kwargs,
                sort_keys=True,
                default=str,
            )

            k_iterator = (
                tqdm_fn(k_values, desc=f"{dataset_name}:{alg_name}") if tqdm_fn else k_values
            )

            for k in k_iterator:
                cluster_key = str(int(k))

                # INCREMENTAL CHECK: Skip if result exists
                if cluster_key in clusters_group and not overwrite:
                    continue

                start = perf_counter()
                labels, internal_metrics = _fit_predict(X_arr, alg_name, n_clusters=k, **run_kwargs)
                elapsed = perf_counter() - start
                ari = float(adjusted_rand_score(y_arr, labels)) if y_arr is not None else np.nan

                # Save labels
                if cluster_key in clusters_group:
                    del clusters_group[cluster_key]
                k_grp = clusters_group.create_group(cluster_key)
                k_grp.create_dataset("predicted_labels", data=labels)

                # Save Metadata and Metrics
                k_grp.attrs["n_clusters"] = int(k)
                k_grp.attrs["elapsed_seconds"] = float(elapsed)
                k_grp.attrs["ari"] = ari
                for m_name, m_val in internal_metrics.items():
                    k_grp.attrs[m_name] = m_val
