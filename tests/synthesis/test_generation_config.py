import numpy as np
import pytest

from tabk.synthesis import ClusterConfig, generate_configs


def extract_arrays(configs):
    k = np.array([c.num_clusters for c in configs], dtype=int)
    n = np.array([c.num_samples for c in configs], dtype=int)
    d = np.array([c.num_dimensions for c in configs], dtype=int)
    return k, n, d


@pytest.mark.parametrize("k_min,k_max", [(2, 2), (3, 7), (10, 15)])
def test_k_range_respected(k_min, k_max):
    configs = generate_configs(
        n_configs=25,
        k_min=k_min,
        k_max=k_max,
        n_low=100,
        n_high=1000,
        d_low=2,
        d_high=50,
        seed=123,
    )
    k, n, d = extract_arrays(configs)
    assert (k >= k_min).all() and (k <= k_max).all()
    if k_min == k_max:
        # Degenerate case: all equal
        assert np.unique(k).size == 1


@pytest.mark.parametrize("n_low,n_high", [(50, 51), (100, 2500)])
def test_n_range_and_integrality(n_low, n_high):
    configs = generate_configs(
        n_configs=40,
        n_low=n_low,
        n_high=n_high,
        d_low=2,
        d_high=10,
        k_min=2,
        k_max=8,
        seed=777,
    )
    k, n, d = extract_arrays(configs)
    assert (n >= n_low).all() and (n <= n_high).all()
    assert issubclass(n.dtype.type, np.integer)  # ints after casting
    if n_low == n_high:
        assert np.unique(n).size == 1


@pytest.mark.parametrize("d_low,d_high", [(2, 5), (10, 200)])
def test_d_range(d_low, d_high):
    configs = generate_configs(
        n_configs=30,
        n_low=100,
        n_high=500,
        d_low=d_low,
        d_high=d_high,
        k_min=2,
        k_max=10,
        seed=99,
    )
    k, n, d = extract_arrays(configs)
    assert (d >= d_low).all() and (d <= d_high).all()
    if d_low == d_high:
        assert np.unique(d).size == 1


def test_reproducibility_same_seed():
    """
    Ensures that generating configs with the same seed produces identical results.
    This serves as the primary verification for deterministic generation.
    """
    n_configs = 50
    seed = 42

    cfgs_a = generate_configs(n_configs=n_configs, seed=seed)
    cfgs_b = generate_configs(n_configs=n_configs, seed=seed)

    # Dataclasses support direct equality checks; this verifies all fields match
    assert cfgs_a == cfgs_b


def test_global_seed_does_not_affect_generation():
    """
    Ensure that setting np.random.seed() globally does not alter the
    generation, which should rely on its own local RNG.
    """
    seed = 12345
    n_configs = 10

    # Case 1: Global seed set to 0
    np.random.seed(0)
    cfgs_a = generate_configs(n_configs=n_configs, seed=seed)

    # Case 2: Global seed set to 999
    np.random.seed(999)
    cfgs_b = generate_configs(n_configs=n_configs, seed=seed)

    assert cfgs_a == cfgs_b


def test_different_seed_changes_at_least_one_config():
    cfgs_a = generate_configs(n_configs=20, seed=111)
    cfgs_b = generate_configs(n_configs=20, seed=222)
    triplets_a = {(c.num_clusters, c.num_samples, c.num_dimensions) for c in cfgs_a}
    triplets_b = {(c.num_clusters, c.num_samples, c.num_dimensions) for c in cfgs_b}
    # Not guaranteed but overwhelmingly likely; fallback assert difference
    assert triplets_a != triplets_b


def test_all_instances_are_clusterconfig():
    configs = generate_configs(n_configs=10, seed=5)
    assert all(isinstance(c, ClusterConfig) for c in configs)


def test_zipf_alpha_extremes():
    # Lower alpha -> heavier tail; higher alpha -> more mass near k_min
    cfg_low_alpha = generate_configs(n_configs=200, zipf_alpha=1.01, seed=1)
    cfg_high_alpha = generate_configs(n_configs=200, zipf_alpha=5.0, seed=1)

    k_low, _, _ = extract_arrays(cfg_low_alpha)
    k_high, _, _ = extract_arrays(cfg_high_alpha)

    # Expect median for high alpha closer to k_min than low-alpha median
    assert np.median(k_high) <= np.median(k_low)


def test_invalid_k_range_raises():
    # k_min > k_max should fail inside truncated zipf construction
    with pytest.raises((ValueError, AssertionError)):
        generate_configs(n_configs=5, k_min=10, k_max=5)


def test_large_batch_shapes():
    N = 250
    configs = generate_configs(n_configs=N, seed=321)
    assert len(configs) == N
    k, n, d = extract_arrays(configs)
    assert k.shape == (N,) and n.shape == (N,) and d.shape == (N,)


def test_monotonic_effect_of_parameter_bounds():
    # Tight upper bounds produce smaller maxima
    cfg_tight = generate_configs(n_configs=50, n_low=100, n_high=200, seed=10)
    cfg_wide = generate_configs(n_configs=50, n_low=100, n_high=2000, seed=10)
    _, n_tight, _ = extract_arrays(cfg_tight)
    _, n_wide, _ = extract_arrays(cfg_wide)
    assert n_tight.max() <= n_wide.max()
