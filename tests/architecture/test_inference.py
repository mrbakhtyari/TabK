import numpy as np
import numpy.testing as npt
import pytest
import torch
from sklearn.datasets import load_iris

from tabk import TabK
from tabk.architecture import AppConfig, ModelConfig
from tabk.architecture.utils import create_model


def _tiny_model(max_rows: int = 2500) -> TabK:
    torch.manual_seed(7)
    config = AppConfig(
        model=ModelConfig(d_model=32, n_head=4, n_layers=2, num_bins=8), max_rows=max_rows
    )
    return TabK([create_model(config).eval()], config, torch.device("cpu"))


def test_predict_returns_k_in_supported_range():
    model = _tiny_model()
    table = np.random.default_rng(0).normal(size=(40, 5))

    assert model.k_values[0] <= model.predict(table) <= model.k_values[-1]


def test_predict_proba_is_a_distribution_over_k_values():
    model = _tiny_model()
    table = np.random.default_rng(0).normal(size=(40, 5))

    proba = model.predict_proba(table)

    assert proba.shape == model.k_values.shape
    npt.assert_allclose(proba.sum(), 1.0, rtol=1e-6)
    assert model.k_values[proba.argmax()] == model.predict(table)


def test_output_is_invariant_to_row_and_column_permutations():
    model = _tiny_model()
    rng = np.random.default_rng(1)
    table = rng.normal(size=(50, 7))
    permuted = table[rng.permutation(table.shape[0])][:, rng.permutation(table.shape[1])]

    npt.assert_allclose(
        model.predict_proba(table, scale=False),
        model.predict_proba(permuted, scale=False),
        rtol=1e-4,
        atol=1e-5,
    )


def test_tables_above_max_rows_are_subsampled(caplog):
    model = _tiny_model(max_rows=30)
    table = np.random.default_rng(2).normal(size=(100, 4))

    first = model.predict_proba(table)
    second = model.predict_proba(table)

    assert "subsampling 30 rows" in caplog.text
    npt.assert_array_equal(first, second)


def test_missing_values_are_rejected():
    table = np.random.default_rng(3).normal(size=(20, 3))
    table[0, 0] = np.nan

    with pytest.raises(ValueError, match="missing"):
        _tiny_model().predict(table)


def test_loading_directory_without_checkpoints_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .pth checkpoints"):
        TabK.from_pretrained(tmp_path, device="cpu")


def test_pretrained_model_predicts_three_clusters_on_iris():
    X, _ = load_iris(return_X_y=True)

    assert TabK.from_pretrained(device="cpu").predict(X) == 3
