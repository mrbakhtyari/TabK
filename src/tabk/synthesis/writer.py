import json
import logging
import random
from pathlib import Path
from typing import Any, Protocol, cast

import h5py
import numpy as np

from tabk.utils import apply_standard_scaling

from .core import ClusterConfig
from .utils import to_jsonable

logger = logging.getLogger(__name__)


class DatasetWriter(Protocol):
    def save_group(
        self,
        strategy_name: str,
        cfg_idx: int,
        cfg: ClusterConfig,
        repeats: list[dict[str, Any]],
    ) -> None: ...


def _sample_id(strategy: str, cfg_idx: int, rep_id: str) -> str:
    return f"{strategy}_cfg{cfg_idx:05d}_{rep_id}"


class H5Writer:
    """Write generated samples to one HDF5 file and assign splits at completion."""

    def __init__(self, path: Path, n_repeats: int, test_ratio: float, seed: int):
        self.path = path
        self.n_repeats = n_repeats
        self.test_ratio = test_ratio
        self.seed = seed
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._temp_path = path.with_name(f"{path.name}.tmp")
        self._file = h5py.File(self._temp_path, "w")
        self._root = self._file.create_group("datasets")
        self._samples: list[tuple[str, int, str, str]] = []

    def save_group(
        self,
        strategy_name: str,
        cfg_idx: int,
        cfg: ClusterConfig,
        repeats: list[dict[str, Any]],
    ) -> None:
        rep_width = len(str(self.n_repeats - 1))
        for rep_idx, rep in enumerate(repeats):
            rep_id = f"rep{rep_idx:0{rep_width}d}"
            sample_id = _sample_id(strategy_name, cfg_idx, rep_id)
            if sample_id in self._root:
                logger.warning("Duplicate sample_id: %s, skipping", sample_id)
                continue

            try:
                X_raw = np.asarray(rep["X"], dtype=np.float64)
                y = np.asarray(rep["y"], dtype=np.int64)
                X_norm = apply_standard_scaling(X_raw)
            except Exception as exc:
                logger.warning("Failed to normalize %s: %s", sample_id, exc)
                continue

            group = self._root.create_group(sample_id)
            group.create_dataset("normalized_features", data=X_norm)
            group.create_dataset("labels", data=y)
            group.attrs["config_group_id"] = f"{strategy_name}_cfg{cfg_idx:05d}"
            group.attrs["k_value"] = cfg.num_clusters
            group.attrs["strategy"] = strategy_name
            group.attrs["cfg_index"] = cfg_idx
            group.attrs["n_objects"] = cfg.num_samples
            group.attrs["n_dimensions"] = cfg.num_dimensions
            group.attrs["seed"] = int(rep["seed"])
            group.attrs["strategy_config_json"] = json.dumps(
                to_jsonable(rep["strategy_config"]), default=str
            )
            self._samples.append((strategy_name, cfg_idx, rep_id, sample_id))

    def finish(self) -> None:
        """Assign deterministic splits and publish the complete file."""
        try:
            if not self._samples:
                raise ValueError("No samples were generated")

            sample_ids = [item[3] for item in sorted(self._samples)]
            indices = list(range(len(sample_ids)))
            random.Random(self.seed).shuffle(indices)
            n_test = max(1, int(len(indices) * self.test_ratio))
            test_ids = {sample_ids[i] for i in indices[:n_test]}
            for sample_id in sample_ids:
                group = cast(h5py.Group, self._root[sample_id])
                group.attrs["split"] = "test" if sample_id in test_ids else "train"

            self._file.close()
            self._temp_path.replace(self.path)
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        """Close an incomplete file without replacing the previous output."""
        self._file.close()
        self._temp_path.unlink(missing_ok=True)
