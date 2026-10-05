"""Immutable, validated sampling priors for synthetic data generation."""

from dataclasses import dataclass, field
from math import isclose, isfinite
from typing import Literal

import numpy as np


def _validate_bounds(low: float, high: float) -> None:
    if not (isfinite(low) and isfinite(high) and low <= high):
        raise ValueError("Sampling bounds must be finite and low <= high")


def _validate_probability(name: str, value: float) -> None:
    if not (isfinite(value) and 0 <= value <= 1):
        raise ValueError(f"{name} must be a finite probability in [0, 1]")


@dataclass(frozen=True, slots=True)
class UniformRange:
    low: float
    high: float
    dist: Literal["uniform"] = field(default="uniform", init=False)

    def __post_init__(self) -> None:
        _validate_bounds(self.low, self.high)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True, slots=True)
class LogUniformRange:
    low: float
    high: float
    dist: Literal["loguniform"] = field(default="loguniform", init=False)

    def __post_init__(self) -> None:
        _validate_bounds(self.low, self.high)
        if self.low <= 0:
            raise ValueError("Log-uniform bounds must be positive")

    def sample(self, rng: np.random.Generator) -> float:
        return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))


@dataclass(frozen=True, slots=True)
class BetaInterpolatedRange:
    a: float
    b: float
    y_min: float
    y_max: float
    dist: Literal["beta_interp"] = field(default="beta_interp", init=False)

    def __post_init__(self) -> None:
        _validate_bounds(self.y_min, self.y_max)
        if not (isfinite(self.a) and isfinite(self.b) and self.a > 0 and self.b > 0):
            raise ValueError("Beta shape parameters must be finite and positive")

    def sample(self, rng: np.random.Generator) -> float:
        return float(np.interp(rng.beta(self.a, self.b), [0, 1], [self.y_min, self.y_max]))


@dataclass(frozen=True, slots=True)
class Choices[T]:
    choices: tuple[T, ...]
    p: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.choices, tuple) or not self.choices:
            raise ValueError("Choices must be a non-empty tuple")
        for choice in self.choices:
            try:
                hash(choice)
            except TypeError as e:
                raise ValueError("Choices must contain immutable values") from e
        if self.p is not None:
            if not isinstance(self.p, tuple) or len(self.p) != len(self.choices):
                raise ValueError("Choice probabilities must be a tuple matching the choices")
            for value in self.p:
                _validate_probability("Choice probability", value)
            if not isclose(sum(self.p), 1.0, rel_tol=0, abs_tol=1e-8):
                raise ValueError("Choice probabilities must sum to 1")


@dataclass(frozen=True, slots=True, kw_only=True)
class CesarCominConfig:
    alpha: LogUniformRange = LogUniformRange(0.05, 5.0)


@dataclass(frozen=True, slots=True, kw_only=True)
class RepliclustConfig:
    overlap: Choices[tuple[float, float]] = Choices(((0.001, 0.002), (0.002, 0.25), (0.25, 0.5)))
    transform_type: Choices[Literal["none", "distort", "wrap"]] = Choices(
        ("none", "distort", "wrap"), (0.6, 0.2, 0.2)
    )
    aspect_ref: Choices[int] = Choices((1, 3))
    aspect_maxmin: Choices[int] = Choices((1, 3))
    radius_maxmin: Choices[int] = Choices((1, 3, 10))
    imbalance_ratio: Choices[int] = Choices((1, 3, 10))

    def __post_init__(self) -> None:
        for low, high in self.overlap.choices:
            _validate_bounds(low, high)
            if low < 0 or high > 1:
                raise ValueError("Overlap bounds must be in [0, 1]")


@dataclass(frozen=True, slots=True, kw_only=True)
class ConcentricHyperspheresConfig:
    noise: UniformRange = UniformRange(0.005, 0.1)
    factor: UniformRange = UniformRange(0.4, 0.8)


@dataclass(frozen=True, slots=True, kw_only=True)
class MoonsConfig:
    noise: UniformRange = UniformRange(0.05, 0.8)


@dataclass(frozen=True, slots=True, kw_only=True)
class DensiredConfig:
    radius: LogUniformRange = LogUniformRange(0.6, 1.8)
    step_factor: UniformRange = UniformRange(0.7, 1.6)
    min_dist: UniformRange = UniformRange(0.92, 1.06)
    ratio_noise: UniformRange = UniformRange(0.05, 0.35)
    use_connectors_prob: float = 0.6
    max_edges_cap: int = 6
    ratio_con: UniformRange = UniformRange(0.03, 0.25)
    con_min_dist: UniformRange = UniformRange(0.9, 1.05)
    con_step_factor: UniformRange = UniformRange(0.6, 1.1)
    dens_factors_prob: float = 0.8
    momentum: BetaInterpolatedRange = BetaInterpolatedRange(1.5, 1.5, 0.25, 0.9)
    distribution: Choices[Literal["studentt", "gaussian", "paper"]] = Choices(
        ("studentt", "gaussian", "paper"), (0.5, 0.3, 0.2)
    )

    def __post_init__(self) -> None:
        _validate_probability("use_connectors_prob", self.use_connectors_prob)
        _validate_probability("dens_factors_prob", self.dens_factors_prob)
        if type(self.max_edges_cap) is not int or self.max_edges_cap < 1:
            raise ValueError("max_edges_cap must be a positive integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class PyClugenConfig:
    llength: LogUniformRange = LogUniformRange(10.0, 60.0)
    base_sep_factor: UniformRange = UniformRange(0.2, 0.4)
    angle_disp: UniformRange = UniformRange(0.1, 0.6)
    llength_disp_factor: UniformRange = UniformRange(0.6, 1.1)
    lateral_disp: UniformRange = UniformRange(0.5, 3.0)
    proj_dist_fn_prob: float = 0.6
    point_dist_fn_prob: float = 0.8

    def __post_init__(self) -> None:
        _validate_probability("proj_dist_fn_prob", self.proj_dist_fn_prob)
        _validate_probability("point_dist_fn_prob", self.point_dist_fn_prob)


type StrategySamplingConfig = (
    CesarCominConfig
    | RepliclustConfig
    | ConcentricHyperspheresConfig
    | MoonsConfig
    | DensiredConfig
    | PyClugenConfig
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplingConfig:
    """Priors for the five strategies in the default registry."""

    cesar_comin: CesarCominConfig = CesarCominConfig()
    repliclust: RepliclustConfig = RepliclustConfig()
    concentric_hyperspheres: ConcentricHyperspheresConfig = ConcentricHyperspheresConfig()
    densired: DensiredConfig = DensiredConfig()
    pyclugen: PyClugenConfig = PyClugenConfig()
