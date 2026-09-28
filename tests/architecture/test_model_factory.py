import pytest

from tabk.architecture import AppConfig, ModelConfig
from tabk.architecture.ablation_models import (
    ModelWithoutColumnInteraction,
    ModelWithoutPMA,
    ModelWithoutQFE,
)
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
        )
    )


@pytest.mark.parametrize(
    ("model_type", "expected_cls"),
    [
        ("default", DoubleInvariantTransformer),
        ("qfe_simple", ModelWithoutQFE),
        ("pool_mean", ModelWithoutPMA),
        ("no_col_attn", ModelWithoutColumnInteraction),
    ],
)
def test_create_model_returns_expected_backbone(model_type, expected_cls):
    model = create_model(_build_config(model_type))
    assert isinstance(model, expected_cls)


def test_create_model_unknown_type_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown model type"):
        create_model(_build_config("unknown_model"))
