import numpy as np
import numpy.testing as npt
import pytest
import torch

from tabk.architecture import (
    AppConfig,
    ModelConfig,
    run_batch_inference,
    run_single_inference,
)
from tabk.architecture.utils import create_model


def _build_inference_context(model_type: str) -> dict:
    torch.manual_seed(7)
    config = AppConfig(
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

    model = create_model(config).to("cpu")
    model.eval()
    model.app_config = config

    return {
        "models": [model],
        "config": config,
        "max_rows": config.max_rows,
        "max_cols": config.max_cols,
        "device": torch.device("cpu"),
    }


@pytest.mark.parametrize("model_type", ["default", "dit_fla_v1"])
def test_batch_inference_matches_single_inference(model_type: str):
    ctx = _build_inference_context(model_type)
    rng = np.random.default_rng(123)

    tables = [
        rng.normal(size=(17, 6)).astype(np.float32),
        rng.normal(size=(9, 11)).astype(np.float32),
        rng.normal(size=(25, 4)).astype(np.float32),
    ]
    targets = [4, 6, 5]

    single_results = [
        run_single_inference(ctx, table_data=table, target_value=target)
        for table, target in zip(tables, targets)
    ]
    batch_results = run_batch_inference(
        ctx,
        tables_data=tables,
        target_values=targets,
        batch_size=2,
        bucket_by_shape=True,
    )

    assert len(batch_results) == len(single_results)

    for single, batched in zip(single_results, batch_results):
        npt.assert_allclose(
            np.asarray(single["prediction"]),
            np.asarray(batched["prediction"]),
            rtol=1e-5,
            atol=1e-5,
        )
        npt.assert_allclose(
            np.asarray(single["raw_output"]),
            np.asarray(batched["raw_output"]),
            rtol=1e-5,
            atol=1e-5,
        )
        assert single["predicted_value"] == batched["predicted_value"]
        assert sorted((single["head_metrics"] or {}).keys()) == sorted(
            (batched["head_metrics"] or {}).keys()
        )


def test_batch_inference_empty_input_returns_empty_list():
    ctx = _build_inference_context("default")
    assert run_batch_inference(ctx, tables_data=[], target_values=None) == []
