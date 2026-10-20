"""Write booktabs tabulars in the paper's formatting."""

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from experiments import TABLES


def escape(text: str) -> str:
    """Escape LaTeX special characters in a dataset or method name."""
    return text.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def thousands(value: int) -> str:
    """Format an integer with LaTeX thousands separators."""
    return f"{value:,}".replace(",", "{,}")


def scientific(value: float) -> str:
    """Format a p-value as mantissa times a power of ten."""
    exponent = math.floor(math.log10(value))
    mantissa = value / 10**exponent
    if round(mantissa, 1) >= 10:
        mantissa, exponent = mantissa / 10, exponent + 1
    return rf"${mantissa:.1f} \times 10^{{{exponent}}}$"


def seconds(value: float) -> str:
    """Format a latency with three significant digits, as in the paper."""
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def ranked(
    values: ArrayLike,
    texts: Sequence[str],
    lower_is_better: bool,
    underline_second: bool = True,
) -> list[str]:
    """Bold the best value and underline the second best, keeping ties together."""
    numbers = np.asarray(values, dtype=float)
    distinct = sorted({round(float(v), 12) for v in numbers}, reverse=not lower_is_better)
    styled = []
    for value, text in zip(numbers, texts):
        position = distinct.index(round(float(value), 12))
        if position == 0:
            styled.append(rf"\textbf{{{text}}}")
        elif position == 1 and underline_second:
            styled.append(rf"\underline{{{text}}}")
        else:
            styled.append(text)
    return styled


type Formatter = int | Callable[[float], str]


def ranked_rows(
    metrics: pd.DataFrame,
    spec: list[tuple[str, Formatter, bool]],
    underline_second: bool = True,
) -> list[list[str]]:
    """Format metric columns, given as (column, decimals or formatter, lower is better),
    and mark each column's best value; returns one list of cells per metrics row."""
    columns = []
    for column, formatter, lower_is_better in spec:
        values = metrics[column].to_numpy(float)
        if isinstance(formatter, int):
            texts = fmt(values, formatter)
        else:
            texts = [formatter(v) for v in values]
        columns.append(ranked(values, texts, lower_is_better, underline_second))
    return [list(cells) for cells in zip(*columns)]


def recovery(pred_k: int, true_k: int) -> str:
    """Bold exact recoveries and underline predictions off by one."""
    if pred_k == true_k:
        return rf"\textbf{{{pred_k}}}"
    if abs(pred_k - true_k) == 1:
        return rf"\underline{{{pred_k}}}"
    return str(pred_k)


def write_tabular(name: str, spec: str, header: list[str], body: list[str]) -> None:
    """Write a booktabs tabular with the given header and body lines."""
    TABLES.mkdir(parents=True, exist_ok=True)
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", *header, r"\midrule", *body]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (TABLES / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def row(cells: Sequence[str]) -> str:
    """Join cells into one tabular row."""
    return " & ".join(cells) + r" \\"


def fmt(values: ArrayLike, digits: int) -> list[str]:
    """Format numbers with a fixed number of decimals."""
    return [f"{v:.{digits}f}" for v in np.asarray(values, dtype=float)]
