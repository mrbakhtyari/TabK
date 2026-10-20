from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

StrategyConfig = dict[str, Any]


@dataclass(frozen=True)
class ClusterConfig:
    """Specification of a synthetic dataset."""

    num_clusters: int
    num_samples: int
    num_dimensions: int

    def validate(self) -> "ClusterConfig":
        if self.num_clusters < 1:
            raise ValueError("num_clusters must be >= 1")
        if self.num_samples < self.num_clusters:
            raise ValueError("num_samples must be >= num_clusters")
        if self.num_dimensions < 1:
            raise ValueError("num_dimensions must be >= 1")
        return self


class DataGenerationStrategy(ABC):
    def __init__(self, cluster_config: ClusterConfig) -> None:
        self.cluster_config = cluster_config

    @abstractmethod
    def generate(
        self,
        strategy_config: StrategyConfig,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) where X shape = (n_samples, d), y shape = (n_samples,)."""
        ...


class DataGenerator:
    def __init__(self, strategy: DataGenerationStrategy) -> None:
        """Initialize the data generator with a strategy."""
        self.strategy = strategy

    def generate_dataset(
        self,
        strategy_config: StrategyConfig,
        seed: int | None = None,
        shuffle: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        X, y = self.strategy.generate(strategy_config, rng)
        if shuffle:
            idx = rng.permutation(X.shape[0])
            X, y = X[idx], y[idx]
        return X, y
