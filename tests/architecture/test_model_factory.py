import pytest

from tabk.architecture import AppConfig, ModelConfig
from tabk.architecture.ablation_model import AblationDoubleInvariantTransformer
from tabk.architecture.linear_model import FLADoubleInvariantTransformer
from tabk.architecture.model import DoubleInvariantTransformer
from tabk.architecture.utils import create_model


def _build_config(model_type: str) -> AppConfig:
    return AppConfig(
        model=ModelConfig(
            d_model=32,
            n_head=4,
            n_layers=2,
            num_bins=8,
            model_type=model_type,
            fla_backend="torch_linear",
            fla_allow_fallback=True,
            fla_eps=1e-6,
        )
    )


def test_create_model_default_backbone():
    model = create_model(_build_config("default"))
    assert isinstance(model, DoubleInvariantTransformer)


def test_create_model_fla_backbone():
    model = create_model(_build_config("dit_fla_v1"))
    assert isinstance(model, FLADoubleInvariantTransformer)


def test_create_model_ablation_backbone():
    model = create_model(_build_config("pool_mean"))
    assert isinstance(model, AblationDoubleInvariantTransformer)


def test_create_model_aliases():
    linear_alias_model = create_model(_build_config("linear_v1"))
    default_alias_model = create_model(_build_config("dit_v1"))

    assert isinstance(linear_alias_model, FLADoubleInvariantTransformer)
    assert isinstance(default_alias_model, DoubleInvariantTransformer)


def test_create_model_unknown_type_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown model_type"):
        create_model(_build_config("unknown_model"))
