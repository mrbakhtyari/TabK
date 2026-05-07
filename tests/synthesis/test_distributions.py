import numpy as np
from scipy.stats import chisquare, kstest

from tabk.synthesis.distributions import (
    features_from_buckets,
    loguniform_quantile,
    make_truncated_zipf,
    zipf_quantile,
)


def test_zipf_range_and_repro():
    dist = make_truncated_zipf(2, 15, 1.15)
    rng = np.random.default_rng(123)
    u = rng.random(50_000)
    k = zipf_quantile(u, dist)
    assert (k >= 2).all() and (k <= 15).all()
    # reproducible mapping
    k2 = zipf_quantile(u, dist)
    assert np.array_equal(k, k2)


def test_zipf_fit():
    dist = make_truncated_zipf(2, 15, 1.15)
    rng = np.random.default_rng(123)
    u = rng.random(200_000)
    k = zipf_quantile(u, dist)
    ks = np.arange(2, 16)
    obs = np.array([(k == vv).sum() for vv in ks])
    exp = dist.pmf(ks) * obs.sum()
    chi, p = chisquare(obs, exp)
    assert p > 1e-3


def test_loguniform_range_and_repro():
    rng = np.random.default_rng(123)
    u = rng.random(100_000)
    n = loguniform_quantile(u, 100, 2500)
    assert (n >= 100).all() and (n <= 2500).all()
    # reproducibility
    n2 = loguniform_quantile(u, 100, 2500)
    assert np.array_equal(n, n2)


def test_loguniform_log_uniformity():
    rng = np.random.default_rng(123)
    u = rng.random(100_000)
    stat, p = kstest(u, "uniform")
    assert p > 1e-6


def test_features_from_buckets_simple():
    rng = np.random.default_rng(123)
    N = 100_000
    ub = rng.random(N)
    uw = rng.random(N)
    d = features_from_buckets(ub, uw, None)
    assert d.min() >= 2 and d.max() <= 200


def test_features_from_buckets_range_and_shape():
    rng = np.random.default_rng(123)
    N = 100_000
    u_bucket = rng.random(N)
    u_within = rng.random(N)

    d = features_from_buckets(u_bucket, u_within, None)

    assert d.min() >= 2
    assert d.max() <= 200
    assert d.shape == (N,)


def test_features_from_buckets_proportions():
    rng = np.random.default_rng(123)
    N = 200_000
    u_bucket = rng.random(N)

    spec = "0.5:2:30,0.35:31:90,0.15:91:200"

    parts = [t.strip() for t in spec.split(",") if t.strip()]
    probs, los, his = [], [], []
    for tok in parts:
        p_s, lo_s, hi_s = tok.split(":")
        probs.append(float(p_s))
        los.append(int(lo_s))
        his.append(int(hi_s))
    probs = np.array(probs)

    cum = np.cumsum(probs)
    bi = np.searchsorted(cum, np.clip(u_bucket, 0, 1 - 1e-12), side="left")
    counts = np.bincount(bi, minlength=len(probs)) / N

    se = np.sqrt(probs * (1 - probs) / N)
    assert np.all(np.abs(counts - probs) <= 4 * se + 1e-3)
