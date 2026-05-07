import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .utils import to_jsonable


class DatasetWriter(Protocol):
    def save_group(
        self,
        strategy_name: str,
        cfg_idx: int,
        cfg: Any,
        repeats: list[dict],
    ) -> None: ...


@dataclass(slots=True)
class NPZWriter:
    base_dir: Path
    n_configs: int = 1000
    n_repeats: int = 100

    def save_group(
        self,
        strategy_name: str,
        cfg_idx: int,
        cfg: Any,
        repeats: list[dict],
    ) -> None:
        cfg_width = len(str(self.n_configs - 1))
        rep_width = len(str(self.n_repeats - 1))

        target_dir = Path(self.base_dir) / strategy_name / f"cfg{cfg_idx:0{cfg_width}d}"
        os.makedirs(target_dir, exist_ok=True)

        base_name = f"{strategy_name}_cfg{cfg_idx:0{cfg_width}d}"
        npz_path = target_dir / f"{base_name}.npz"
        meta_path = target_dir / f"{base_name}.json"

        save_dict: dict[str, np.ndarray] = {}
        seeds: dict[str, int] = {}
        strat_cfgs: dict[str, Any] = {}

        for rep_idx, rep in enumerate(repeats):
            X = np.asarray(rep["X"], dtype=np.float64)
            y = np.asarray(rep["y"], dtype=np.int64)

            save_dict[f"rep{rep_idx:0{rep_width}d}_X"] = X
            save_dict[f"rep{rep_idx:0{rep_width}d}_y"] = y

            seeds[f"rep{rep_idx:0{rep_width}d}"] = int(rep["seed"])
            strat_cfgs[f"rep{rep_idx:0{rep_width}d}"] = to_jsonable(rep["strategy_config"])

        np.savez_compressed(npz_path, **save_dict)

        meta = {
            "created_at": datetime.now().isoformat() + "Z",
            "strategy_name": strategy_name,
            "cfg_index": cfg_idx,
            "n_repeats": len(repeats),
            "file": os.path.relpath(npz_path, self.base_dir),
            "seeds": seeds,
            "cluster_config": to_jsonable(cfg),
            "strategy_configs": strat_cfgs,
        }

        meta_path.write_text(
            __import__("json").dumps(meta, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved grouped: {npz_path.name} and metadata JSON")
