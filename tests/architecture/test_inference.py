import numpy as np
import numpy.testing as npt
import pytest
import torch
from sklearn.datasets import load_iris

from tabk.architecture import (
    AppConfig,
    ModelConfig,
    load_inference_context,
    predict_single,
    run_single_inference,
)
from tabk.architecture.utils import create_model
from tabk.utils import apply_standard_scaling


def _build_inference_context(max_rows: int = 2500) -> dict:
    torch.manual_seed(7)
    config = AppConfig(
        model=ModelConfig(d_model=32, n_head=4, n_layers=2, num_bins=8), max_rows=max_rows
    )

    model = create_model(config).to("cpu")
    model.eval()
    model.app_config = config

    return {"models": [model], "config": config, "device": torch.device("cpu")}


def test_run_single_inference_returns_k_in_supported_range():
    ctx = _build_inference_context()
    head_cfg = ctx["config"].head_config
    table = np.random.default_rng(0).normal(size=(40, 5)).astype(np.float32)

    result = run_single_inference(ctx, table)

    assert set(result) >= {"prediction", "predicted_value", "inference_time_ms", "raw_output"}
    assert head_cfg.min_k <= result["predicted_value"] <= head_cfg.max_k


def test_output_is_invariant_to_row_and_column_permutations():
    ctx = _build_inference_context()
    rng = np.random.default_rng(1)
    table = rng.normal(size=(50, 7)).astype(np.float32)
    permuted = table[rng.permutation(table.shape[0])][:, rng.permutation(table.shape[1])]

    original = run_single_inference(ctx, table)["raw_output"]
    shuffled = run_single_inference(ctx, permuted)["raw_output"]

    npt.assert_allclose(original, shuffled, rtol=1e-4, atol=1e-5)


def test_tables_above_max_rows_are_subsampled(caplog):
    ctx = _build_inference_context(max_rows=30)
    table = np.random.default_rng(2).normal(size=(100, 4)).astype(np.float32)

    first = run_single_inference(ctx, table)["raw_output"]
    second = run_single_inference(ctx, table)["raw_output"]

    assert "subsampling 30 rows" in caplog.text
    npt.assert_array_equal(first, second)


def test_loading_directory_without_checkpoints_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .pth checkpoints"):
        load_inference_context(tmp_path, device="cpu")


def test_pretrained_model_predicts_three_clusters_on_iris():
    X, _ = load_iris(return_X_y=True)
    ctx = load_inference_context(device="cpu")

    assert predict_single(ctx, apply_standard_scaling(X)) == 3
