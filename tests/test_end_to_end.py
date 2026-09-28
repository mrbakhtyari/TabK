import numpy as np
import numpy.testing as npt
import pytest
from sklearn.datasets import load_iris

from tabk import TabK
from tabk.architecture import scan_h5_datalake
from tabk.cli import DATALAKE_NAME, main

TINY_MODEL = ["--d-model", "16", "--n-head", "2", "--n-layers", "1", "--num-bins", "8"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    main(
        ["generate", "--out", str(root / "data"), "--n-configs", "12"]
        + ["--n-high", "200", "--d-high", "5", "--test-ratio", "0.2"]
    )
    main(
        ["train", "--data", str(root / "data"), "--out", str(root / "model"), *TINY_MODEL]
        + ["--epochs", "2", "--k-folds", "2", "--batch-size", "4", "--accum-steps", "3"]
        + ["--device", "cpu"]
    )
    return root


def test_generate_builds_datalake_with_train_and_test_splits(run):
    splits = scan_h5_datalake(run / "data" / DATALAKE_NAME)

    assert splits["train"] and splits["test"]
    assert all(2 <= sample["k_value"] <= 15 for sample in splits["train"] + splits["test"])


def test_train_writes_one_checkpoint_per_fold(run):
    checkpoints = sorted((run / "model" / "checkpoints").glob("*.pth"))

    assert [path.name for path in checkpoints] == ["checkpoint_fold_1.pth", "checkpoint_fold_2.pth"]
    assert (run / "model" / "training_history.json").is_file()


def test_trained_model_predicts_from_the_cli(run, tmp_path, capsys):
    table = tmp_path / "iris.csv"
    np.savetxt(table, load_iris(return_X_y=True)[0], delimiter=",")

    main(["predict", str(table), "--model", str(run / "model"), "--device", "cpu"])

    assert 2 <= int(capsys.readouterr().out) <= 15


def test_exported_model_matches_the_trained_model(run, tmp_path):
    main(["export", "--model", str(run / "model"), "--out", str(tmp_path / "hub")])
    X = load_iris(return_X_y=True)[0]

    trained = TabK.from_pretrained(run / "model", device="cpu").predict_proba(X)
    exported = TabK.from_pretrained(tmp_path / "hub", device="cpu").predict_proba(X)

    npt.assert_array_equal(trained, exported)
