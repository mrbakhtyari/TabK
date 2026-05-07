from tabk.synthesis import StrategySpec, default_registry


def test_default_registry_shapes_and_predicates():
    registry = default_registry()
    assert isinstance(registry, list) and all(isinstance(s, StrategySpec) for s in registry)

    # Expected strategy names should be present (allowing for future additions)
    names = {s.name for s in registry}
    expected = {
        "CesarComin",
        "Repliclust",
        "ConcentricHyperspheres",
        "MultiInterlocked2DMoons",
        "Densired",
        "PyClugen",
    }
    assert expected.issubset(names)

    # All sampler and supports_cfg must be callable
    for spec in registry:
        assert callable(spec.sampler)
        assert callable(spec.supports_cfg)

    # MultiInterlocked2DMoons only supports 2D configs
    midm = next(s for s in registry if s.name == "MultiInterlocked2DMoons")

    class DummyCfg:
        def __init__(self, d: int):
            self.num_dimensions = d

    assert midm.supports_cfg(DummyCfg(2)) is True
    assert midm.supports_cfg(DummyCfg(3)) is False
