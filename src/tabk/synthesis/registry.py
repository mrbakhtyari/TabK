from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import SamplingConfig, StrategySamplingConfig
from .core import DataGenerationStrategy
from .sampling import (
    cesar_comin_sampler,
    concentric_hyperspheres_sampler,
    densired_sampler,
    pyclugen_sampler,
    repliclust_sampler,
)
from .strategies import (
    CesarCominStrategy,
    ConcentricHyperspheresStrategy,
    DensiredStrategy,
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
    sampling_config: StrategySamplingConfig | None = None


def default_registry(config: SamplingConfig = SamplingConfig()) -> list[StrategySpec]:
    """Bind the five default samplers to the supplied immutable priors."""
    return [
        StrategySpec(
            "CesarComin",
            CesarCominStrategy,
            lambda rng, cfg: cesar_comin_sampler(rng, config=config.cesar_comin),
            lambda cfg: True,
            sampling_config=config.cesar_comin,
        ),
        StrategySpec(
            "Repliclust",
            RepliclustStrategy,
            lambda rng, cfg: repliclust_sampler(rng, config=config.repliclust),
            lambda cfg: True,
            sampling_config=config.repliclust,
        ),
        StrategySpec(
            "ConcentricHyperspheres",
            ConcentricHyperspheresStrategy,
            lambda rng, cfg: concentric_hyperspheres_sampler(
                rng, config=config.concentric_hyperspheres
            ),
            lambda cfg: True,
            sampling_config=config.concentric_hyperspheres,
        ),
        StrategySpec(
            "Densired",
            DensiredStrategy,
            lambda rng, cfg: densired_sampler(rng, cfg.num_clusters, config=config.densired),
            lambda cfg: True,
            sampling_config=config.densired,
        ),
        StrategySpec(
            "PyClugen",
            PyClugenStrategy,
            lambda rng, cfg: pyclugen_sampler(rng, cfg.num_dimensions, config=config.pyclugen),
            lambda cfg: True,
            sampling_config=config.pyclugen,
        ),
    ]
