from tabk.synthesis import StrategySpec, default_registry


def test_default_registry_shapes_and_predicates():
    registry = default_registry()
    assert isinstance(registry, list) and all(isinstance(s, StrategySpec) for s in registry)

    names = {s.name for s in registry}
    expected = {
        "CesarComin",
        "Repliclust",
        "ConcentricHyperspheres",
        "Densired",
        "PyClugen",
    }
    assert names == expected
    assert len(registry) == len(expected)

    # All sampler and supports_cfg must be callable
    for spec in registry:
        assert callable(spec.sampler)
        assert callable(spec.supports_cfg)
