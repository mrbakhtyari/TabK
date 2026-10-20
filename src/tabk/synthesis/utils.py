import contextlib
import io
import random
import sys
from dataclasses import asdict, is_dataclass
from typing import Any, cast

import numpy as np
import torch


@contextlib.contextmanager
def suppress_output(suppress_stdout: bool = True, suppress_stderr: bool = True):
    """Temporarily silence prints/warnings from noisy libs."""
    old_out, old_err = sys.stdout, sys.stderr
    try:
        if suppress_stdout:
            sys.stdout = io.StringIO()
        if suppress_stderr:
            sys.stderr = io.StringIO()
        yield
    finally:
        if suppress_stdout:
            sys.stdout = old_out
        if suppress_stderr:
            sys.stderr = old_err


def seed_everything(seed: int) -> None:
    """Sets seeds for random, numpy, and torch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_jsonable(obj: Any):
    """Best-effort conversion of configs to JSON-safe structures."""
    if obj is None:
        return None
    if isinstance(obj, bool | int | float | str):
        return obj
    # NumPy
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, list | tuple):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return to_jsonable(asdict(cast(Any, obj)))
    return str(obj)
