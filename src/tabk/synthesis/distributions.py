import numpy as np
from scipy.stats import loguniform, rv_discrete


def make_truncated_zipf(k_min: int = 2, k_max: int = 15, alpha: float = 1.6) -> rv_discrete:
    ks = np.arange(k_min, k_max + 1, dtype=int)
    pmf = 1.0 / (ks.astype(float) ** alpha)
    pmf /= pmf.sum()
    return rv_discrete(name="trunc_zipf", values=(ks, pmf))


def zipf_quantile(u: np.ndarray, dist: rv_discrete) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0 - 1e-12)
    return dist.ppf(u).astype(int)


def loguniform_quantile(u: np.ndarray, n_lo: int, n_hi: int) -> np.ndarray:
    """
    Map uniform samples u in [0,1) to discrete values in [n_lo, n_hi]
    using a log-uniform distribution.
    """
    if n_lo <= 0 or n_hi <= n_lo:
        raise ValueError("Require 0 < n_lo < n_hi.")
    u = np.clip(u, 0.0, 1.0 - 1e-12)
    n_cont = loguniform(a=n_lo, b=n_hi).ppf(u)
    n = np.rint(n_cont).astype(int)
    return np.clip(n, n_lo, n_hi)


def features_from_buckets(
    u_bucket: np.ndarray, u_within: np.ndarray, spec: str | None = None
) -> np.ndarray:
    """
    Map two uniforms to integer features d using stratified buckets.

    spec format: "prob:lo:hi,prob:lo:hi,..."
    default -> "0.5:2:30,0.35:31:90,0.15:91:200"
    """
    if spec is None or spec.strip() == "":
        spec = "0.5:2:30,0.35:31:90,0.15:91:200"

    # parse once per call
    parts = [t.strip() for t in spec.split(",") if t.strip()]
    probs, los, his = [], [], []
    for tok in parts:
        p_s, lo_s, hi_s = tok.split(":")
        p, lo, hi = float(p_s), int(lo_s), int(hi_s)
        if not (0.0 < p <= 1.0) or lo < 1 or lo > hi:
            raise ValueError(f"bad bucket: {tok}")
        probs.append(p)
        los.append(lo)
        his.append(hi)

    probs = np.asarray(probs, float)
    los = np.asarray(los, int)
    his = np.asarray(his, int)

    s = probs.sum()
    if not np.isclose(s, 1.0, atol=1e-8):
        raise ValueError(f"bucket probs must sum to 1 (got {s})")

    if u_bucket.shape != u_within.shape:
        raise ValueError("u_bucket and u_within must have same shape")

    ub = np.clip(u_bucket, 0.0, 1.0 - 1e-12)
    uw = np.clip(u_within, 0.0, 1.0 - 1e-12)

    cum = np.cumsum(probs)
    bi = np.searchsorted(cum, ub, side="left")  # bucket indices
    lo = los[bi]
    hi = his[bi]
    d = lo + np.floor(uw * (hi - lo + 1)).astype(int)
    return np.clip(d, lo, hi)
