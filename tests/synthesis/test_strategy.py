import importlib

import numpy as np
import pytest

from tabk.synthesis import (
    CesarCominStrategy,
    ClusterConfig,
    ConcentricHyperspheresStrategy,
    DataGenerator,
    DensiredStrategy,
    MultiInterlocked2DMoonsStrategy,
    PyClugenStrategy,
    RepliclustStrategy,
)


# Helper to skip if optional dependency missing
def _has(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def _assert_identical(X1, y1, X2, y2):
    assert np.allclose(X1, X2)
    assert np.array_equal(y1, y2)


def _assert_different(X1, y1, X2, y2):
    # Allow either data or labels to differ
    assert not (np.allclose(X1, X2) and np.array_equal(y1, y2))


@pytest.mark.parametrize(
    "strategy_cls, cfg, strat_conf",
    [
        (CesarCominStrategy, ClusterConfig(5, 300, 8), {"alpha": 1.2}),
        (ConcentricHyperspheresStrategy, ClusterConfig(4, 240, 6), {"factor": 0.4, "noise": 0.01}),
        (MultiInterlocked2DMoonsStrategy, ClusterConfig(6, 360, 2), {"noise": 0.02}),
    ],
)
def test_numpy_based_repro(strategy_cls, cfg, strat_conf):
    gen = DataGenerator(strategy_cls(cfg))
    X1, y1 = gen.generate_dataset(strat_conf, seed=123)
    X2, y2 = gen.generate_dataset(strat_conf, seed=123)
    X3, y3 = gen.generate_dataset(strat_conf, seed=124)

    _assert_identical(X1, y1, X2, y2)
    _assert_different(X1, y1, X3, y3)

    # Basic shape checks
    assert X1.shape == (cfg.num_samples, cfg.num_dimensions)
    assert y1.shape == (cfg.num_samples,)
    assert 1 <= np.unique(y1).size <= cfg.num_clusters


@pytest.mark.skipif(not _has("repliclust"), reason="repliclust not installed")
@pytest.mark.parametrize("transform_type", ["none", "distort", "wrap"])
def test_repliclust_repro(transform_type):
    cfg = ClusterConfig(5, 250, 10)
    strat_conf = {
        "min_overlap": 0.01,
        "max_overlap": 0.02,
        "aspect_ref": 1,
        "aspect_maxmin": 1,
        "radius_maxmin": 1,
        "distributions": ["normal"],
        "imbalance_ratio": 1,
        "transform_type": transform_type,
    }
    gen = DataGenerator(RepliclustStrategy(cfg))
    X1, y1 = gen.generate_dataset(strat_conf, seed=999)
    X2, y2 = gen.generate_dataset(strat_conf, seed=999)
    X3, y3 = gen.generate_dataset(strat_conf, seed=1000)

    _assert_identical(X1, y1, X2, y2)
    _assert_different(X1, y1, X3, y3)


@pytest.mark.skipif(not _has("densired"), reason="densired not installed")
def test_densired_repro():
    cfg = ClusterConfig(4, 200, 5)
    strat_conf = {"radius": 1.0, "step": 1.3}
    gen = DataGenerator(DensiredStrategy(cfg))
    X1, y1 = gen.generate_dataset(strat_conf, seed=2024)
    X2, y2 = gen.generate_dataset(strat_conf, seed=2024)
    X3, y3 = gen.generate_dataset(strat_conf, seed=2025)

    _assert_identical(X1, y1, X2, y2)
    _assert_different(X1, y1, X3, y3)


@pytest.mark.skipif(not _has("pyclugen"), reason="pyclugen not installed")
def test_pyclugen_repro():
    cfg = ClusterConfig(5, 300, 6)
    strat_conf = {
        "direction": np.array([1, 1, 1, 1, 1, 1], dtype=float),
        "cluster_sep": np.array([1, 2, 3, 4, 5, 6], dtype=float),
        "llength": 10.0,
        "llength_disp": 0.5,
        "lateral_disp": 1.2,
        "angle_disp": 0.15,
    }
    gen = DataGenerator(PyClugenStrategy(cfg))
    X1, y1 = gen.generate_dataset(strat_conf, seed=7)
    X2, y2 = gen.generate_dataset(strat_conf, seed=7)
    X3, y3 = gen.generate_dataset(strat_conf, seed=8)

    _assert_identical(X1, y1, X2, y2)
    _assert_different(X1, y1, X3, y3)


def test_moons_dimension_error_not_repro_related():
    # Ensures error path not mistaken for reproducibility failure
    cfg = ClusterConfig(4, 120, 3)  # invalid dims
    gen = DataGenerator(MultiInterlocked2DMoonsStrategy(cfg))
    with pytest.raises(ValueError):
        gen.generate_dataset({"noise": 0.05}, seed=1)
