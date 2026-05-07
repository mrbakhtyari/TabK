from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import DataGenerationStrategy
from .sampling import (
    cesar_comin_sampler,
    concentric_hyperspheres_sampler,
    densired_sampler,
    moons_sampler,
    pyclugen_sampler,
    repliclust_sampler,
)
from .strategies import (
    CesarCominStrategy,
    ConcentricHyperspheresStrategy,
    DensiredStrategy,
    MultiInterlocked2DMoonsStrategy,
    PyClugenStrategy,
    RepliclustStrategy,
)

VariantHook = Callable[[np.ndarray, np.ndarray], dict[str, np.ndarray]]


@dataclass(frozen=True, slots=True)
class StrategySpec:
    name: str
    strategy_cls: type[DataGenerationStrategy]
    sampler: Callable[[np.random.Generator, Any], dict]
    supports_cfg: Callable[[Any], bool]


def default_registry() -> list[StrategySpec]:
    return [
        StrategySpec(
            "CesarComin",
            CesarCominStrategy,
            lambda rng, cfg: cesar_comin_sampler(rng),
            lambda cfg: True,
        ),
        StrategySpec(
            "Repliclust",
            RepliclustStrategy,
            lambda rng, cfg: repliclust_sampler(rng),
            lambda cfg: True,
        ),
        StrategySpec(
            "ConcentricHyperspheres",
            ConcentricHyperspheresStrategy,
            lambda rng, cfg: concentric_hyperspheres_sampler(rng),
            lambda cfg: True,
        ),
        StrategySpec(
            "MultiInterlocked2DMoons",
            MultiInterlocked2DMoonsStrategy,
            lambda rng, cfg: moons_sampler(rng),
            lambda cfg: cfg.num_dimensions == 2,
        ),
        StrategySpec(
            "Densired",
            DensiredStrategy,
            lambda rng, cfg: densired_sampler(rng, cfg.num_clusters),
            lambda cfg: True,
        ),
        StrategySpec(
            "PyClugen",
            PyClugenStrategy,
            lambda rng, cfg: pyclugen_sampler(rng, cfg.num_dimensions),
            lambda cfg: True,
        ),
    ]
