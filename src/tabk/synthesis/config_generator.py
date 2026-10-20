import numpy as np

from .core import ClusterConfig
from .distributions import (
    loguniform_quantile,
    make_truncated_zipf,
    zipf_quantile,
)
from .sampling import lhs


def generate_configs(
    n_configs: int,
    k_min: int = 2,
    k_max: int = 15,
    n_low: int = 100,
    n_high: int = 2500,
    d_low: int = 2,
    d_high: int = 200,
    zipf_alpha: float = 1.6,
    seed: int = 42,
):
    """
    Map LHS unit-hypercube samples through distribution PPFs to get (k, n, d).
    """
    rng = np.random.default_rng(seed)

    U = lhs(n_configs, 3, rng)  # columns: u_k, u_n, u_d_bucket, u_d_within
    u_k, u_n, u_d = U[:, 0], U[:, 1], U[:, 2]

    # Truncated Zipf via rv_discrete (use PPF to transform uniform to discrete quantiles)
    zipf_dist = make_truncated_zipf(k_min=k_min, k_max=k_max, alpha=zipf_alpha)
    num_clusters = zipf_quantile(u_k, zipf_dist)

    # Log-uniform via SciPy; prefer PPF to keep mapping deterministic w.r.t. U
    num_samples = loguniform_quantile(u_n, n_low, n_high)

    # Log-uniform via SciPy; prefer PPF to keep mapping deterministic w.r.t. U
    num_dimensions = loguniform_quantile(u_d, d_low, d_high)

    return [
        ClusterConfig(int(k), int(n), int(d))
        for k, n, d in zip(num_clusters, num_samples, num_dimensions)
    ]
