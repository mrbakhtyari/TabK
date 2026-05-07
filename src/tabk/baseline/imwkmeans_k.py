from time import perf_counter

import numpy as np


def _mink_dist(
    x: np.ndarray,
    y: np.ndarray,
    p: float,
    w: np.ndarray,
) -> np.ndarray:
    return np.sum((np.abs(x - y) ** p) * w, axis=1) ** (1.0 / p)


def _new_cmt(data: np.ndarray, p: float) -> np.ndarray:
    n_rows, n_cols = data.shape
    if p == 1:
        return np.median(data, axis=0)
    if p == 2:
        return np.mean(data, axis=0)
    if n_rows == 1:
        return data[0].copy()

    gradient = np.full((n_cols,), 0.001, dtype=float)
    center = np.mean(data, axis=0)

    def _objective(c: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(data - c[None, :]) ** p, axis=0)

    dist_center = _objective(center)
    new_center = center + gradient
    dist_new = _objective(new_center)
    gradient[dist_center < dist_new] *= -1.0

    while True:
        new_center = center + gradient
        dist_new = _objective(new_center)
        not_improved = dist_new >= dist_center
        gradient[not_improved] *= 0.9

        improved = dist_new < dist_center
        center[improved] = new_center[improved]
        dist_center[improved] = dist_new[improved]

        if np.all(np.abs(gradient) < 1e-4):
            break

    return center


def _mwk_get_new_u(
    data: np.ndarray,
    z: np.ndarray,
    w_pow: np.ndarray,
    p: float,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.zeros((data.shape[0], z.shape[0]), dtype=float)
    for c in range(z.shape[0]):
        distances[:, c] = _mink_dist(data, z[c][None, :].repeat(data.shape[0], axis=0), p, w_pow[c])
    u = np.argmin(distances, axis=1)
    u_dist_to_z = np.min(distances, axis=1)
    return u, u_dist_to_z


def _mwk_get_new_z(
    data: np.ndarray,
    u: np.ndarray,
    k: int,
    p: float,
    old_z: np.ndarray,
) -> np.ndarray:
    z = old_z.copy()
    for cluster_idx in range(k):
        mask = u == cluster_idx
        if np.sum(mask) > 1:
            z[cluster_idx, :] = _new_cmt(data[mask], p)
    return z


def _mwk_get_new_w(data: np.ndarray, u: np.ndarray, z: np.ndarray, p: float) -> np.ndarray:
    k, n_features = z.shape
    d = np.zeros((k, n_features), dtype=float)
    w = np.zeros((k, n_features), dtype=float)

    for cluster_idx in range(k):
        mask = u == cluster_idx
        if np.any(mask):
            diff = np.abs(data[mask] - z[cluster_idx][None, :]) ** p
            d[cluster_idx, :] = np.sum(diff, axis=0)

    d = d + float(np.mean(d))

    if p != 1:
        exp = 1.0 / (p - 1.0)
        for cluster_idx in range(k):
            for feature_idx in range(n_features):
                tmp = d[cluster_idx, feature_idx]
                denom = np.sum((tmp / d[cluster_idx, :]) ** exp)
                w[cluster_idx, feature_idx] = 1.0 / denom if denom > 0 else (1.0 / n_features)
    else:
        for cluster_idx in range(k):
            min_idx = int(np.argmin(d[cluster_idx, :]))
            w[cluster_idx, min_idx] = 1.0

    return w


def _mwkmeans(
    data: np.ndarray,
    k: int,
    p: float,
    initial_centroids: np.ndarray,
    initial_w: np.ndarray,
    max_loops: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    u = np.zeros((data.shape[0],), dtype=int)
    old_u = u.copy()
    z = initial_centroids.copy()
    w = initial_w.copy()

    loop_count = 0
    while loop_count <= max_loops:
        new_u, u_dist_to_z = _mwk_get_new_u(data, z, w**p, p)
        if np.array_equal(new_u, u):
            break
        if np.array_equal(new_u, old_u):
            break

        old_u = u.copy()
        u = new_u

        z_next = _mwk_get_new_z(data, u, k, p, z)
        if np.array_equal(z_next, z):
            break
        z = z_next

        w = _mwk_get_new_w(data, u, z, p)
        loop_count += 1

    return u, w, z, u_dist_to_z, loop_count


def _intelligent_mwk_initialization(
    data: np.ndarray,
    ik_threshold: int,
    p: float,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_features = data.shape
    equal_w = np.full((n_features,), 1.0 / n_features, dtype=float)

    mink_center = _new_cmt(data, p)
    dist_to_center = _mink_dist(
        data,
        mink_center[None, :].repeat(n_rows, axis=0),
        p,
        equal_w,
    )
    data_work = data[np.argsort(dist_to_center)].copy()

    centroids: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    counts: list[int] = []

    while data_work.shape[0] > 0:
        data_size = data_work.shape[0]
        tent_centroid = data_work[-1].copy()
        prev_belongs: np.ndarray | None = None
        prev_prev_belongs: np.ndarray | None = None
        tent_w = equal_w.copy()

        belongs = np.zeros((data_size,), dtype=bool)
        new_centroid = tent_centroid.copy()
        for _ in range(200):
            dist_tent = _mink_dist(
                data_work,
                tent_centroid[None, :].repeat(data_size, axis=0),
                p,
                tent_w**p,
            )
            dist_center = _mink_dist(
                data_work,
                mink_center[None, :].repeat(data_size, axis=0),
                p,
                tent_w**p,
            )
            belongs = dist_tent < dist_center
            if np.sum(belongs) == 0:
                belongs[-1] = True

            new_centroid = _new_cmt(data_work[belongs], p)
            if np.array_equal(tent_centroid, new_centroid):
                break
            if prev_belongs is not None and np.array_equal(belongs, prev_belongs):
                break
            if prev_prev_belongs is not None and np.array_equal(belongs, prev_prev_belongs):
                break

            tent_centroid = new_centroid.copy()
            prev_prev_belongs = None if prev_belongs is None else prev_belongs.copy()
            prev_belongs = belongs.copy()

            tent_w = _mwk_get_new_w(
                data_work[belongs],
                np.zeros(np.sum(belongs), dtype=int),
                new_centroid[None, :],
                p,
            )[0]

        cluster_size = int(np.sum(belongs))
        if cluster_size > ik_threshold:
            centroids.append(new_centroid.copy())
            weights.append(tent_w.copy())
            counts.append(cluster_size)

        data_work = data_work[~belongs]

    if len(centroids) == 0:
        centroids = [mink_center.copy()]
        weights = [equal_w.copy()]
        counts = [data.shape[0]]

    if len(centroids) > max_k:
        order = np.argsort(np.asarray(counts))[::-1][:max_k]
        centroids = [centroids[idx] for idx in order]
        weights = [weights[idx] for idx in order]

    return np.vstack(centroids), np.vstack(weights)


def predict_k_with_imwkmeans(
    X: np.ndarray,
    k_values: list[int],
    random_state: int = 42,
    p: float = 1.4,
    ik_threshold: int | None = None,
) -> tuple[int | None, float]:
    """Predict k with Intelligent MWK-Means, bounded to the provided k range."""
    del random_state  # Deterministic algorithm for fixed input.

    if not k_values:
        return None, 0.0

    start = perf_counter()
    x_eval = np.asarray(X, dtype=np.float64)
    max_k = int(max(k_values))
    min_k = int(min(k_values))

    if ik_threshold is None:
        ik_threshold = max(0, int(round(0.01 * x_eval.shape[0])))

    try:
        init_z, init_w = _intelligent_mwk_initialization(
            data=x_eval,
            ik_threshold=int(ik_threshold),
            p=float(p),
            max_k=max_k,
        )

        k_found = int(init_z.shape[0])
        k_run = int(np.clip(k_found, min_k, max_k))
        if k_run != k_found:
            init_z = init_z[:k_run]
            init_w = init_w[:k_run]

        u, _, _, _, _ = _mwkmeans(
            data=x_eval,
            k=k_run,
            p=float(p),
            initial_centroids=init_z,
            initial_w=init_w,
            max_loops=500,
        )

        predicted_k = int(np.unique(u).size)
    except Exception:
        predicted_k = None

    elapsed_ms = (perf_counter() - start) * 1000.0

    if predicted_k is None or predicted_k not in set(k_values):
        return None, elapsed_ms
    return predicted_k, elapsed_ms
