import numpy as np
import numpy.testing as npt
from sklearn.datasets import load_iris

from tabk.cli import main, read_csv


def test_read_csv_skips_header(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("a,b\n1,2\n3,4\n")

    npt.assert_array_equal(read_csv(path), [[1, 2], [3, 4]])


def test_read_csv_without_header(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("1,2\n3,4\n")

    npt.assert_array_equal(read_csv(path), [[1, 2], [3, 4]])


def test_predict_prints_k_for_iris(tmp_path, capsys):
    path = tmp_path / "iris.csv"
    X, _ = load_iris(return_X_y=True)
    np.savetxt(path, X, delimiter=",", header="a,b,c,d", comments="")

    main(["predict", str(path), "--device", "cpu"])

    assert capsys.readouterr().out.strip() == "3"
