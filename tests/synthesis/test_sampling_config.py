from dataclasses import FrozenInstanceError, replace

import pytest

from tabk.synthesis.config import (
    BetaInterpolatedRange,
    Choices,
    DensiredConfig,
    LogUniformRange,
    PyClugenConfig,
    RepliclustConfig,
    SamplingConfig,
    UniformRange,
)


@pytest.mark.parametrize("bounds", [(2, 1), (float("nan"), 1), (0, float("inf"))])
@pytest.mark.parametrize("range_cls", [UniformRange, LogUniformRange])
def test_ranges_reject_invalid_bounds(range_cls, bounds):
    with pytest.raises(ValueError, match="Sampling bounds"):
        range_cls(*bounds)


@pytest.mark.parametrize("low", [0, -1])
def test_loguniform_rejects_nonpositive_bounds(low):
    with pytest.raises(ValueError, match="positive"):
        LogUniformRange(low, 1)


@pytest.mark.parametrize("a, b", [(0, 1), (1, -1), (float("nan"), 1), (1, float("inf"))])
def test_beta_rejects_invalid_shape_parameters(a, b):
    with pytest.raises(ValueError, match="shape parameters"):
        BetaInterpolatedRange(a, b, 0, 1)


@pytest.mark.parametrize("p", [(1,), (0.2, 0.2), (-0.1, 1.1), (float("nan"), 0.5)])
def test_choices_reject_invalid_probabilities(p):
    with pytest.raises(ValueError):
        Choices((1, 2), p)


def test_choices_reject_empty_or_mutable_inputs():
    with pytest.raises(ValueError, match="non-empty tuple"):
        Choices(())
    with pytest.raises(ValueError, match="non-empty tuple"):
        Choices([1, 2])
    with pytest.raises(ValueError, match="immutable"):
        Choices(([1, 2],))
    with pytest.raises(ValueError, match="tuple matching"):
        Choices((1, 2), [0.5, 0.5])


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
@pytest.mark.parametrize(
    "config_cls, name",
    [
        (DensiredConfig, "use_connectors_prob"),
        (DensiredConfig, "dens_factors_prob"),
        (PyClugenConfig, "proj_dist_fn_prob"),
        (PyClugenConfig, "point_dist_fn_prob"),
    ],
)
def test_strategy_configs_validate_probabilities(config_cls, name, value):
    with pytest.raises(ValueError, match=name):
        config_cls(**{name: value})


@pytest.mark.parametrize("cap", [0, -1, 1.5, True])
def test_densired_requires_positive_integer_edge_cap(cap):
    with pytest.raises(ValueError, match="max_edges_cap"):
        DensiredConfig(max_edges_cap=cap)


@pytest.mark.parametrize("overlap", [((-0.1, 0.5),), ((0.1, 1.1),), ((0.5, 0.1),)])
def test_repliclust_validates_overlap_bounds(overlap):
    with pytest.raises(ValueError):
        RepliclustConfig(overlap=Choices(overlap))


def test_configs_are_immutable_and_can_be_replaced_independently():
    original = SamplingConfig()
    with pytest.raises(FrozenInstanceError):
        original.densired.use_connectors_prob = 0
    with pytest.raises(FrozenInstanceError):
        original.cesar_comin.alpha.low = 1

    changed = replace(original, densired=replace(original.densired, use_connectors_prob=0))
    assert changed.densired.use_connectors_prob == 0
    assert original.densired.use_connectors_prob == 0.6
