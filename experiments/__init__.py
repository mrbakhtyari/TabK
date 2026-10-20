"""Reproduce the paper's tables, figures and in-text numbers from benchmark results."""

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
BENCHMARK = RESULTS / "benchmark"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
SECTIONS = RESULTS / "sections"
K_VALUES = np.arange(2, 16)
PROB_COLUMNS = [f"k_{k}" for k in K_VALUES]
PROB_FLOOR = 1e-12
METHODS = [
    "TabK",
    "SCARF+CH",
    "TCP",
    "CH",
    "SubTab+Sil",
    "SCARF+Sil",
    "IMWk",
    "VIME+Sil",
    "VIME+CH",
    "Sil",
    "CNAK",
    "SubTab+CH",
    "DB",
    "Dunn",
    "GMM-BIC",
    "X-Means",
    "Gap",
    "UNSEEN",
]
FIGURE_METHODS = [
    "TabK",
    "TCP",
    "X-Means",
    "Sil",
    "CH",
    "DB",
    "Dunn",
    "CNAK",
    "Gap",
    "GMM-BIC",
    "IMWk",
    "UNSEEN",
    "SCARF+CH",
    "SubTab+Sil",
    "VIME+Sil",
]


@cache
def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(BENCHMARK / f"{name}.csv")


def load(name: str) -> pd.DataFrame:
    """Read a benchmark results CSV (cached; callers get their own copy)."""
    return _read(name).copy()


def predictions() -> pd.DataFrame:
    """Read benchmark predictions indexed by OpenML ID."""
    return load("predictions").set_index("openml_id")


def aligned(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Align one-row-per-dataset results with a benchmark frame."""
    return load(name).set_index("openml_id").loc[frame.index]


def errors(frame: pd.DataFrame, methods: list[str] = METHODS) -> pd.DataFrame:
    """Return absolute cardinality errors."""
    return pd.DataFrame(frame[methods].sub(frame["true_k"], axis=0).abs())


def error_metrics(true_k: np.ndarray, pred_k: np.ndarray) -> dict[str, float]:
    """Compute MAE, exact match and tolerance percentages."""
    err = np.abs(pred_k - true_k)
    return {
        "MAE": float(err.mean()),
        "EM": float((err == 0).mean() * 100),
        "tol1": float((err <= 1).mean() * 100),
    }


def brier(probs: np.ndarray, true_k: np.ndarray) -> float:
    """Mean multiclass Brier score over k = 2..15."""
    target = np.eye(len(K_VALUES))[true_k.astype(int) - K_VALUES[0]]
    return float(((probs - target) ** 2).sum(axis=1).mean())


def nll(probs: np.ndarray, true_k: np.ndarray) -> float:
    """Mean negative log-likelihood of the true k, floored at PROB_FLOOR."""
    p_true = probs[np.arange(len(probs)), true_k.astype(int) - K_VALUES[0]]
    return float(-np.log(np.clip(p_true, PROB_FLOOR, 1)).mean())


def ari_at(frame: pd.DataFrame, k: np.ndarray, algorithm: str) -> np.ndarray:
    """Look up the downstream ARI at each dataset's selected k."""
    ari = load("downstream_ari")
    lookup = ari.loc[ari["algorithm"] == algorithm].set_index(["openml_id", "k"])["ari"]
    keys = pd.MultiIndex.from_arrays([frame.index, k])
    return lookup.loc[keys].to_numpy(float)


def ensemble_log_probs(frame: pd.DataFrame) -> np.ndarray:
    """Mean of the five members' log-probabilities, one row per dataset."""
    members = pd.DataFrame(load("tabk_log_probs").groupby("openml_id")[PROB_COLUMNS].mean())
    return members.loc[frame.index].to_numpy(float)


def write_section(name: str, values: dict[str, float]) -> None:
    """Write the in-text numbers of one paper section."""
    SECTIONS.mkdir(parents=True, exist_ok=True)
    pd.Series(values, name="value").rename_axis("quantity").reset_index().to_csv(
        SECTIONS / f"{name}.csv", index=False
    )
