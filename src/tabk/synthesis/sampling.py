from typing import Any, cast

import numpy as np
from scipy.stats import qmc

from .config import (
    CesarCominConfig,
    ConcentricHyperspheresConfig,
    DensiredConfig,
    MoonsConfig,
    PyClugenConfig,
    RepliclustConfig,
)


def cesar_comin_sampler(
    rng: np.random.Generator, *, config: CesarCominConfig = CesarCominConfig()
) -> dict:
    return {
        "alpha": config.alpha.sample(rng),
    }


def repliclust_sampler(
    rng: np.random.Generator, *, config: RepliclustConfig = RepliclustConfig()
) -> dict:
    idx = rng.choice(len(config.overlap.choices), p=config.overlap.p)
    min_overlap, max_overlap = config.overlap.choices[idx]

    transform_type = rng.choice(config.transform_type.choices, p=config.transform_type.p)

    return {
        "min_overlap": min_overlap,
        "max_overlap": max_overlap,
        "aspect_ref": rng.choice(config.aspect_ref.choices, p=config.aspect_ref.p),
        "aspect_maxmin": rng.choice(config.aspect_maxmin.choices, p=config.aspect_maxmin.p),
        "radius_maxmin": rng.choice(config.radius_maxmin.choices, p=config.radius_maxmin.p),
        "imbalance_ratio": rng.choice(config.imbalance_ratio.choices, p=config.imbalance_ratio.p),
        "transform_type": transform_type,
    }


def concentric_hyperspheres_sampler(
    rng: np.random.Generator,
    *,
    config: ConcentricHyperspheresConfig = ConcentricHyperspheresConfig(),
) -> dict:
    return {
        "noise": config.noise.sample(rng),
        "factor": config.factor.sample(rng),
    }


def moons_sampler(rng: np.random.Generator, *, config: MoonsConfig = MoonsConfig()) -> dict:
    return {
        "noise": config.noise.sample(rng),
    }


def densired_sampler(
    rng: np.random.Generator, num_clusters: int, *, config: DensiredConfig = DensiredConfig()
) -> dict:
    # core size & spacing
    radius = config.radius.sample(rng)
    step = radius * config.step_factor.sample(rng)
    min_dist = config.min_dist.sample(rng)

    # noise & connectors
    ratio_noise = config.ratio_noise.sample(rng)
    use_connectors = rng.random() < config.use_connectors_prob

    max_edges = num_clusters * (num_clusters - 1) // 2
    upper = min(config.max_edges_cap, max_edges)

    connections = int(rng.integers(1, upper + 1)) if use_connectors else 0
    ratio_con = config.ratio_con.sample(rng) if use_connectors else 0
    con_min_dist = config.con_min_dist.sample(rng) if use_connectors else 0.9
    con_step = step * config.con_step_factor.sample(rng) if use_connectors else 2

    # density heterogeneity & dynamics
    dens_factors = rng.random() < config.dens_factors_prob
    momentum = config.momentum.sample(rng)

    dist_choice = rng.choice(config.distribution.choices, p=config.distribution.p)

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


def pyclugen_sampler(
    rng: np.random.Generator, num_dims: int, *, config: PyClugenConfig = PyClugenConfig()
) -> dict:
    # Random unit direction
    direction = rng.normal(size=num_dims)
    direction /= np.linalg.norm(direction)

    # Core elongation
    llength = config.llength.sample(rng)

    # Slight anisotropy across axes and gentle ↓ with sqrt
    base_sep = llength * config.base_sep_factor.sample(rng)
    cluster_sep = (base_sep / np.sqrt(num_dims)) * (0.5 + rng.random(num_dims))

    proj_dist_fn = "norm" if (rng.random() < config.proj_dist_fn_prob) else "unif"
    point_dist_fn = "n" if (rng.random() < config.point_dist_fn_prob) else "n-1"

    return {
        "direction": direction,
        "angle_disp": config.angle_disp.sample(rng),
        "cluster_sep": cluster_sep,
        "llength": llength,
        "llength_disp": llength * config.llength_disp_factor.sample(rng),
        "lateral_disp": config.lateral_disp.sample(rng),
        "proj_dist_fn": proj_dist_fn,
        "point_dist_fn": point_dist_fn,
    }


def lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube Sampling."""
    sampler = cast(Any, qmc.LatinHypercube)(d=d, seed=rng)
    return sampler.random(n=n)
