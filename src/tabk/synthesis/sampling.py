import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

STRATEGY_PARAMS_PATH = Path(__file__).parent / "strategy_params.json"


def _load_params() -> dict:
    """Load sampling parameters from JSON file."""
    with open(STRATEGY_PARAMS_PATH) as f:
        return json.load(f)


PARAMS = _load_params()


def sample_loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def cesar_comin_sampler(rng: np.random.Generator) -> dict:
    cfg = PARAMS["CesarComin"]
    return {
        "alpha": sample_loguniform(rng, cfg["alpha"]["low"], cfg["alpha"]["high"]),
    }


def repliclust_sampler(rng: np.random.Generator) -> dict:
    cfg = PARAMS["Repliclust"]

    # Handle overlap choices which are lists of lists in JSON
    overlap_choices = [tuple(x) for x in cfg["overlap"]["choices"]]

    # safer to sample index
    idx = rng.choice(len(overlap_choices))
    min_overlap, max_overlap = overlap_choices[idx]

    transform_type = rng.choice(cfg["transform_type"]["choices"], p=cfg["transform_type"]["p"])

    return {
        "min_overlap": min_overlap,
        "max_overlap": max_overlap,
        "aspect_ref": rng.choice(cfg["aspect_ref"]["choices"]),
        "aspect_maxmin": rng.choice(cfg["aspect_maxmin"]["choices"]),
        "radius_maxmin": rng.choice(cfg["radius_maxmin"]["choices"]),
        "imbalance_ratio": rng.choice(cfg["imbalance_ratio"]["choices"]),
        "transform_type": transform_type,
    }


def concentric_hyperspheres_sampler(rng: np.random.Generator) -> dict:
    cfg = PARAMS["ConcentricHyperspheres"]
    return {
        "noise": rng.uniform(cfg["noise"]["low"], cfg["noise"]["high"]),
        "factor": rng.uniform(cfg["factor"]["low"], cfg["factor"]["high"]),
    }


def moons_sampler(rng: np.random.Generator) -> dict:
    cfg = PARAMS["MultiInterlocked2DMoons"]
    return {
        "noise": rng.uniform(cfg["noise"]["low"], cfg["noise"]["high"]),
    }


def densired_sampler(rng: np.random.Generator, num_clusters: int) -> dict:
    cfg = PARAMS["Densired"]

    # core size & spacing
    radius = sample_loguniform(rng, cfg["radius"]["low"], cfg["radius"]["high"])
    step = radius * rng.uniform(cfg["step_factor"]["low"], cfg["step_factor"]["high"])
    min_dist = rng.uniform(cfg["min_dist"]["low"], cfg["min_dist"]["high"])

    # noise & connectors
    ratio_noise = rng.uniform(cfg["ratio_noise"]["low"], cfg["ratio_noise"]["high"])
    use_connectors = rng.random() < cfg["use_connectors_prob"]

    max_edges = num_clusters * (num_clusters - 1) // 2
    upper = min(cfg["max_edges_cap"], max_edges)

    connections = int(rng.integers(1, upper + 1)) if use_connectors else 0
    ratio_con = (
        rng.uniform(cfg["ratio_con"]["low"], cfg["ratio_con"]["high"]) if use_connectors else 0
    )
    con_min_dist = (
        rng.uniform(cfg["con_min_dist"]["low"], cfg["con_min_dist"]["high"])
        if use_connectors
        else 0.9
    )
    con_step = (
        step * rng.uniform(cfg["con_step_factor"]["low"], cfg["con_step_factor"]["high"])
        if use_connectors
        else 2
    )

    # density heterogeneity & dynamics
    dens_factors = rng.random() < cfg["dens_factors_prob"]

    mom_cfg = cfg["momentum"]
    momentum = float(
        np.interp(
            rng.beta(mom_cfg["a"], mom_cfg["b"]), [0, 1], [mom_cfg["y_min"], mom_cfg["y_max"]]
        )
    )

    dist_choice = rng.choice(cfg["distribution"]["choices"], p=cfg["distribution"]["p"])

    return {
        "radius": radius,
        "step": step,
        "min_dist": min_dist,
        "ratio_noise": ratio_noise,
        "connections": connections,
        "ratio_con": ratio_con,
        "con_min_dist": con_min_dist,
        "con_step": con_step,
        "dens_factors": dens_factors,
        "momentum": momentum,
        "distribution": dist_choice,
    }


def pyclugen_sampler(rng: np.random.Generator, num_dims: int) -> dict:
    cfg = PARAMS["PyClugen"]

    # Random unit direction
    direction = rng.normal(size=num_dims)
    direction /= np.linalg.norm(direction)

    # Core elongation
    llength = sample_loguniform(rng, cfg["llength"]["low"], cfg["llength"]["high"])

    # Slight anisotropy across axes and gentle ↓ with sqrt
    base_sep = llength * rng.uniform(cfg["base_sep_factor"]["low"], cfg["base_sep_factor"]["high"])
    cluster_sep = (base_sep / np.sqrt(num_dims)) * (0.5 + rng.random(num_dims))

    proj_dist_fn = "norm" if (rng.random() < cfg["proj_dist_fn_prob"]) else "unif"
    point_dist_fn = "n" if (rng.random() < cfg["point_dist_fn_prob"]) else "n-1"

    return {
        "direction": direction,
        "angle_disp": rng.uniform(cfg["angle_disp"]["low"], cfg["angle_disp"]["high"]),
        "cluster_sep": cluster_sep,
        "llength": llength,
        "llength_disp": llength
        * rng.uniform(cfg["llength_disp_factor"]["low"], cfg["llength_disp_factor"]["high"]),
        "lateral_disp": rng.uniform(cfg["lateral_disp"]["low"], cfg["lateral_disp"]["high"]),
        "proj_dist_fn": proj_dist_fn,
        "point_dist_fn": point_dist_fn,
    }


def lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube Sampling."""
    sampler = qmc.LatinHypercube(d=d, seed=rng)
    return sampler.random(n=n)
