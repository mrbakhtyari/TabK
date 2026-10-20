import os

import numpy as np
import pyclugen
from densired import datagen

from .core import DataGenerationStrategy, StrategyConfig
from .utils import seed_everything, suppress_output

with suppress_output():
    import repliclust


def _compute_cluster_sizes(total: int, k: int) -> list[int]:
    """Evenly distribute total samples into k parts (first remainder buckets get +1)."""
    base = total // k
    rem = total % k
    return [base + (1 if i < rem else 0) for i in range(k)]


class CesarCominStrategy(DataGenerationStrategy):
    """Implements approach from https://doi.org/10.1371/journal.pone.0210236."""

    def generate(
        self,
        strategy_config: StrategyConfig,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha = strategy_config["alpha"]  # required
        e, v, e_bar = 0.0, 0.25, 1.0  # parameters to control the data generation process.

        m = int(np.round((e_bar**2.0 - e**2.0) / v))
        e_chap = np.sqrt(e / m)
        v_chap = -(e_chap**2.0) + np.sqrt(e_chap**4.0 + v / m)

        n_dims = self.cluster_config.num_dimensions
        sizes = _compute_cluster_sizes(
            self.cluster_config.num_samples, self.cluster_config.num_clusters
        )

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for i, size in enumerate(sizes):
            q = rng.normal(size=[n_dims, m])
            f = (e_chap + np.sqrt(v_chap) * q) / alpha
            u = 2.0 * (rng.random(n_dims) - 0.5)
            z = rng.normal(size=[m, size])
            x = (f @ z).T + u
            X_parts.append(x)
            y_parts.append(np.full(size, i))

        return np.concatenate(X_parts), np.concatenate(y_parts)


class RepliclustStrategy(DataGenerationStrategy):
    """Implements approach from (https://doi.org/10.1007/s00357-025-09501-w)."""

    def generate(
        self,
        strategy_config: StrategyConfig,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

        # Extract transform_type from config (default to "none" if missing)
        transform_type = strategy_config.get("transform_type", "none")

        # Create a copy for Archetype to avoid passing unknown arguments
        archetype_config = strategy_config.copy()
        if "transform_type" in archetype_config:
            del archetype_config["transform_type"]

        seed = int(rng.integers(0, 2**31 - 1))

        # Repliclust's distort function relies on global random state (numpy and torch)
        # So we must seed them explicitly to ensure reproducibility.
        seed_everything(seed)

        with suppress_output():
            repliclust.set_seed(seed)
            archetype = repliclust.Archetype(
                n_clusters=self.cluster_config.num_clusters,
                dim=self.cluster_config.num_dimensions,
                n_samples=self.cluster_config.num_samples,
                **archetype_config,
            )
            gen = repliclust.DataGenerator(archetype=archetype, quiet=True)
            X, y, _meta = gen.synthesize(quiet=True)

            # Apply transformation if requested
            if transform_type == "distort":
                X = repliclust.distort(X)
            elif transform_type == "wrap":
                X = repliclust.wrap_around_sphere(X)

        return X, y


class ConcentricHyperspheresStrategy(DataGenerationStrategy):
    """Generate concentric hyperspherical clusters in arbitrary dimensions."""

    def generate(
        self, strategy_config: StrategyConfig, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        factor = strategy_config.get("factor", 0.5)
        noise_std = strategy_config.get("noise", 0.0)

        sizes = _compute_cluster_sizes(
            self.cluster_config.num_samples, self.cluster_config.num_clusters
        )
        n_dim = self.cluster_config.num_dimensions

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for i, size in enumerate(sizes):
            # Create uniform points on the hypersphere
            vec = rng.normal(0, 1, size=(size, n_dim))
            vec /= np.linalg.norm(vec, axis=1, keepdims=True)

            # Apply radius scaling based on factor
            radius = 1.0 * ((1.0 - factor) * i + 1.0)  # e.g. for factor=0.5: 1, 1.5, 2, 2.5
            vec *= radius

            # Optional Gaussian noise
            if noise_std > 0:
                vec += rng.normal(0, noise_std, size=vec.shape)

            X_parts.append(vec)
            y_parts.append(np.full(size, i))

        return np.vstack(X_parts), np.concatenate(y_parts)


class MultiInterlocked2DMoonsStrategy(DataGenerationStrategy):
    """Generate interlocked 2D moons (like make_moons) for any number of clusters."""

    def generate(
        self,
        strategy_config: StrategyConfig,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.cluster_config.num_dimensions != 2:
            raise ValueError("MultiInterlocked2DMoonsStrategy only supports 2D.")
        noise = strategy_config.get("noise", 0.05)

        num_samples = self.cluster_config.num_samples
        num_clusters = self.cluster_config.num_clusters
        sizes = _compute_cluster_sizes(num_samples, num_clusters)

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        cluster_id = 0

        def make_moon(n_points: int, flip: bool, offset: np.ndarray) -> np.ndarray:
            theta = np.linspace(0, np.pi, n_points)
            x = np.cos(theta)
            y = np.sin(theta)
            if flip:
                x = -x + 1
                y = -y
            moon = np.stack([x, y], axis=1)
            moon += rng.normal(0, noise, size=moon.shape)
            return moon + offset

        while cluster_id + 1 < num_clusters:
            size_a = sizes[cluster_id]
            size_b = sizes[cluster_id + 1]
            shift = rng.uniform(-10, 10, size=2)
            moon_a = make_moon(size_a, flip=False, offset=shift)
            moon_b = make_moon(size_b, flip=True, offset=shift)
            X_parts.extend([moon_a, moon_b])
            y_parts.extend([np.full(size_a, cluster_id), np.full(size_b, cluster_id + 1)])
            cluster_id += 2
        if cluster_id < num_clusters:  # odd leftover
            remaining = sizes[cluster_id]
            shift = rng.uniform(-10, 10, size=2)
            moon = make_moon(remaining, flip=False, offset=shift)
            X_parts.append(moon)
            y_parts.append(np.full(remaining, cluster_id))

        return np.vstack(X_parts), np.concatenate(y_parts)


class DensiredStrategy(DataGenerationStrategy):
    """Densired from https://doi.org/10.1007/978-3-031-70368-3_1"""

    def generate(
        self, strategy_config: StrategyConfig, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        seed = int(rng.integers(0, 2**31 - 1))
        gen = datagen.densityDataGen(
            dim=self.cluster_config.num_dimensions,
            clunum=self.cluster_config.num_clusters,
            seed=seed,
            **strategy_config,
        )
        data = gen.generate_data(self.cluster_config.num_samples)
        X = np.asarray(data[:, :-1], dtype=float)
        y = np.asarray(data[:, -1], dtype=int)
        return X, y


class PyClugenStrategy(DataGenerationStrategy):
    """PyClugen data generation strategy from https://doi.org/10.1016/j.knosys.2023.110836"""

    def generate(
        self,
        strategy_config: StrategyConfig,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        data = pyclugen.clugen(
            num_dims=self.cluster_config.num_dimensions,
            num_clusters=self.cluster_config.num_clusters,
            num_points=self.cluster_config.num_samples,
            rng=rng,
            **strategy_config,
        )

        X = np.asarray(data[0], dtype=float)
        y = np.asarray(data[1], dtype=int)
        return X, y
