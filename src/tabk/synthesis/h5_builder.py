import json
import logging
import random
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from tabk.utils import apply_standard_scaling

logger = logging.getLogger(__name__)


def _discover_npz_files(data_dir: Path) -> list[Path]:
    """Find all raw .npz files under the data directory."""
    return sorted(data_dir.rglob("*.npz"))


def _load_json_sidecar(npz_path: Path) -> dict:
    """Load the JSON metadata file that sits next to the NPZ."""
    json_path = npz_path.with_suffix(".json")
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _make_sample_id(strategy: str, cfg_index: int, rep_id: str) -> str:
    """Build a unique, human-readable sample identifier."""
    return f"{strategy}_cfg{cfg_index:05d}_{rep_id}"


def _assign_splits(
    sample_records: list[dict],
    test_ratio: float,
    seed: int,
) -> list[dict]:
    """Assign train/test split by shuffling and splitting sample indices."""
    indices = list(range(len(sample_records)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_test = max(1, int(len(indices) * test_ratio))
    test_set = set(indices[:n_test])

    for i, r in enumerate(sample_records):
        r["split"] = "test" if i in test_set else "train"
    return sample_records


def build_h5_from_raw(
    data_dir: Path | str,
    out_h5: str | None = None,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Walk raw NPZ files, normalize, and write a unified HDF5 DataLake."""
    data_dir = Path(data_dir)

    # Default output name matches the parent directory name
    if out_h5 is None:
        out_h5 = f"{data_dir.resolve().name}.h5"
    h5_path = data_dir / out_h5

    # Phase 1: Discover & index all samples
    npz_files = _discover_npz_files(data_dir)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {data_dir}")
    logger.info(f"Found {len(npz_files)} NPZ files")

    sample_records: list[dict] = []
    for npz_path in tqdm(npz_files, desc="Indexing NPZ files"):
        meta = _load_json_sidecar(npz_path)
        cluster_config = meta.get("cluster_config", {})
        seeds = meta.get("seeds", {})
        strategy_configs = meta.get("strategy_configs", {})

        k_value = cluster_config.get("num_clusters")
        if k_value is None:
            logger.warning(f"No k_value in sidecar for {npz_path}, skipping")
            continue

        strategy = meta.get("strategy_name", npz_path.parent.parent.name)
        cfg_index = meta.get("cfg_index", -1)
        config_group_id = f"{strategy}_cfg{cfg_index:05d}"

        # Discover all rep keys in the NPZ
        try:
            with np.load(npz_path) as npz:
                x_keys = sorted(k for k in npz.keys() if k.endswith("_X"))
        except Exception as e:
            logger.warning(f"Failed to read {npz_path}: {e}")
            continue

        for x_key in x_keys:
            rep_id = x_key.replace("_X", "")  # e.g. "rep0"
            sample_id = _make_sample_id(strategy, cfg_index, rep_id)

            # Per-replication metadata
            rep_seed = seeds.get(rep_id, -1)
            rep_strategy_config = strategy_configs.get(rep_id, {})

            sample_records.append(
                {
                    "sample_id": sample_id,
                    "npz_path": str(npz_path),
                    "x_key": x_key,
                    "y_key": f"{rep_id}_y",
                    # Core metadata
                    "config_group_id": config_group_id,
                    "k_value": int(k_value),
                    "strategy": strategy,
                    "cfg_index": int(cfg_index),
                    "n_objects": int(cluster_config.get("num_samples", -1)),
                    "n_dimensions": int(cluster_config.get("num_dimensions", -1)),
                    "seed": int(rep_seed),
                    # Strategy hyperparameters (JSON string for HDF5)
                    "strategy_config_json": json.dumps(rep_strategy_config, default=str),
                }
            )

    if not sample_records:
        raise ValueError("No samples found in any NPZ file")

    logger.info(f"Indexed {len(sample_records)} total samples")

    # Phase 2: Assign train/test split
    sample_records = _assign_splits(sample_records, test_ratio=test_ratio, seed=seed)

    n_train = sum(1 for r in sample_records if r["split"] == "train")
    n_test = sum(1 for r in sample_records if r["split"] == "test")
    logger.info(f"Split: {n_train} train, {n_test} test ({len(sample_records)} total)")

    # Phase 3: Build HDF5
    logger.info(f"Writing HDF5 to: {h5_path}")
    written = 0
    skipped = 0

    with h5py.File(h5_path, "w") as h5f:
        root = h5f.create_group("datasets")

        for rec in tqdm(sample_records, desc="Building HDF5"):
            sid = rec["sample_id"]
            if sid in root:
                logger.warning(f"Duplicate sample_id: {sid}, skipping")
                skipped += 1
                continue

            try:
                with np.load(rec["npz_path"]) as npz:
                    X_raw = np.asarray(npz[rec["x_key"]], dtype=np.float64)
                    y = np.asarray(npz[rec["y_key"]], dtype=np.int64)

                X_norm = apply_standard_scaling(X_raw)
            except Exception as e:
                logger.warning(f"Failed to load/normalize {sid}: {e}")
                skipped += 1
                continue

            grp = root.create_group(sid)
            grp.create_dataset("normalized_features", data=X_norm)
            grp.create_dataset("labels", data=y)

            # Store metadata as HDF5 attributes
            grp.attrs["config_group_id"] = rec["config_group_id"]
            grp.attrs["k_value"] = rec["k_value"]
            grp.attrs["split"] = rec["split"]
            grp.attrs["strategy"] = rec["strategy"]
            grp.attrs["cfg_index"] = rec["cfg_index"]
            grp.attrs["n_objects"] = rec["n_objects"]
            grp.attrs["n_dimensions"] = rec["n_dimensions"]
            grp.attrs["seed"] = rec["seed"]
            grp.attrs["strategy_config_json"] = rec["strategy_config_json"]

            written += 1

    logger.info(
        f"HDF5 complete: {written} samples written, {skipped} skipped "
        f"→ {h5_path} ({h5_path.stat().st_size / 1e9:.2f} GB)"
    )
