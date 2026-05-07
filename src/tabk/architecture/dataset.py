import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.progress import write_progress_line
from .config import AppConfig

logger = logging.getLogger(__name__)


def scan_h5_datalake(h5_path: Path | str) -> dict[str, list[dict]]:
    """Single-pass scan: reads all sample IDs and their metadata, grouped by split.

    Returns a dict like ``{"train": [{"sample_id": ..., "config_group_id": ..., ...}, ...], ...}``
    """
    splits: dict[str, list[dict]] = {}
    with h5py.File(h5_path, "r") as h5f:
        root = h5f["datasets"]
        for sid in root:
            attrs = dict(root[sid].attrs)
            attrs["sample_id"] = sid
            split = attrs.get("split", "train")
            splits.setdefault(split, []).append(attrs)
    for split, items in splits.items():
        logger.info(f"Split '{split}': {len(items)} samples")
    return splits


class H5Dataset(Dataset):
    """In-memory PyTorch Dataset pre-cached from the HDF5 DataLake.

    All features and targets are loaded and transformed into RAM during
    ``__init__`` so that training incurs zero disk I/O and zero redundant
    computation per batch.
    """

    def __init__(
        self,
        h5_path: Path | str,
        sample_ids: list[str],
        config: AppConfig,
    ):
        target_transform = config.head_config.target_transform
        self.target_dtype = config.head_config.target_dtype
        head_type = config.head_config.head_type

        self.features: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        total_samples = len(sample_ids)

        write_progress_line(f"Caching data into RAM | start | samples: {total_samples}")

        with h5py.File(str(h5_path), "r") as h5f:
            root = h5f["datasets"]
            for sid in sample_ids:
                group = root[sid]
                self.features.append(group["normalized_features"][:])

                if head_type == "k_estimator":
                    raw = np.array(group.attrs["k_value"])
                elif head_type == "algorithm_recommender":
                    raw = group["ari_scores"][:]
                else:
                    raise ValueError(f"Unsupported head_type: {head_type}")

                # Precompute transform once (not per __getitem__)
                self.targets.append(target_transform(raw))

        write_progress_line(f"Caching data into RAM | end | cached: {len(self.features)} samples")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.features[idx], dtype=torch.float32),
            torch.as_tensor(self.targets[idx], dtype=self.target_dtype),
        )


def collate_batch(batch):
    """Pad variable-size tables to the largest dimensions in the batch and produce masks."""
    max_rows = max(x.shape[0] for x, _ in batch)
    max_cols = max(x.shape[1] for x, _ in batch)

    data_list, target_list, row_masks, col_masks = [], [], [], []

    for data, target in batch:
        rows, cols = data.shape
        padded = torch.zeros(max_rows, max_cols, dtype=torch.float32)
        r_mask = torch.ones(max_rows, dtype=torch.bool)
        c_mask = torch.ones(max_cols, dtype=torch.bool)

        r, c = min(rows, max_rows), min(cols, max_cols)
        padded[:r, :c] = data[:r, :c]
        r_mask[:r] = False
        c_mask[:c] = False

        data_list.append(padded)
        target_list.append(target)
        row_masks.append(r_mask)
        col_masks.append(c_mask)

    return (
        torch.stack(data_list),
        torch.stack(target_list),
        torch.stack(row_masks),
        torch.stack(col_masks),
    )
