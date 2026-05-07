import datetime
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from .ablation_models import (
    ModelWithoutColumnInteraction,
    ModelWithoutPMA,
    ModelWithoutQFE,
)
from .config import AppConfig
from .heads import get_head
from .model import DoubleInvariantTransformer

logger = logging.getLogger(__name__)


def create_model(config: AppConfig) -> torch.nn.Module:
    """Create model based on config.

    Returns:
        Model instance with head attached
    """
    model_type = config.model.model_type

    # Head config is now directly available via polymorphism
    head = get_head(config)

    if model_type == "default":
        return DoubleInvariantTransformer(config.model, head=head)
    elif model_type == "qfe_simple":
        return ModelWithoutQFE(config.model, head=head)
    elif model_type == "pool_mean":
        return ModelWithoutPMA(config.model, head=head)
    elif model_type == "no_col_attn":
        return ModelWithoutColumnInteraction(config.model, head=head)

    raise ValueError(f"Unknown model type '{model_type}'")


def set_seed(seed: int):
    """
    Sets seeds for all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic operations ensure that running the same code on the same
    # hardware produces identical results.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For newer PyTorch versions using cuBLAS
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    logger.info(f"Random Seed set to: {seed}")


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    path,
    fold_idx=None,
    config: AppConfig = None,
    additional_info=None,
):
    """
    Save model checkpoint with comprehensive metadata.

    Args:
        model: The model to save
        optimizer: The optimizer state
        epoch: Current epoch
        loss: Best validation loss
        path: Path to save checkpoint
        fold_idx: Optional fold index for K-fold training
        config: Optional AppConfig object to save hyperparameters
        additional_info: Optional dict with additional metadata
    """

    logger.info(f"Saving model to {path}...")

    checkpoint = {
        # Model and training state
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        # Metadata for reproducibility
        "timestamp": datetime.datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        # Training configuration
        "config": config.to_dict() if config else None,
        # K-fold information
        "fold_idx": fold_idx,
    }

    # Add any additional information
    if additional_info:
        checkpoint["additional_info"] = additional_info

    torch.save(checkpoint, path)
    logger.info("✓ Checkpoint saved successfully")


def load_checkpoint(path, model, optimizer=None, device=None):
    logger.info(f"Loading model from {path}...")
    # weights_only=True is recommended for security and to silence warnings in newer PyTorch
    # Use map_location to handle loading CUDA models on CPU
    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]


def plot_training_history(all_histories: list, save_dir: str = ".") -> None:
    """
    Plots training and validation loss for each fold.

    Args:
        all_histories: List of history dicts, each containing 'train_loss' and 'val_loss' lists.
        save_dir: Directory to save the plot.
    """

    plt.figure(figsize=(12, 8))

    for fold_idx, history in enumerate(all_histories):
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.plot(
            epochs,
            history["train_loss"],
            label=f"Fold {fold_idx + 1} Train",
            linestyle="--",
        )
        plt.plot(epochs, history["val_loss"], label=f"Fold {fold_idx + 1} Val")

    plt.title("Training and Validation Loss per Fold")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    save_path = Path(save_dir) / "loss_curves.png"
    plt.savefig(save_path)
    logger.info(f"Loss curves saved to {save_path}")
    plt.close()


def save_training_history(all_histories: list, save_path: str = "training_history.json") -> None:
    """
    Saves training history to a JSON file.

    Args:
        all_histories: List of history dicts to save.
        save_path: Path to save the JSON file.
    """
    with open(save_path, "w") as f:
        json.dump(all_histories, f, indent=2)
    logger.info(f"Training history saved to {save_path}")


def save_run_manifest(config, metrics, path) -> None:
    """
    Saves a run manifest JSON sidecar.

    Args:
        config: Configuration dictionary (args).
        metrics: Dictionary of final validation metrics.
        path: Path to save the manifest.
    """
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": config,
        "results": metrics,
    }

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Run manifest saved to {path}")


def log_device_info(config: AppConfig) -> None:
    """Log information about the compute device."""
    if not config:
        logger.warning("No config provided to log_device_info")
        return

    device = config.training.torch_device
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory Usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")


def log_training_configuration(config: AppConfig) -> None:
    """Log key training parameters."""
    logger.info("Training Configuration:")

    if config:
        conf_dict = config.to_dict()
        # Pretty print or specific keys
        for section, values in conf_dict.items():
            if isinstance(values, dict):
                logger.info(f"  [{section}]")
                for k, v in values.items():
                    logger.info(f"    {k}: {v}")
            else:
                logger.info(f"  {section}: {values}")
    else:
        logger.warning("No config provided to log_training_configuration")


def calculate_class_weights(samples: list[dict], min_k: int, max_k: int) -> list[float] | None:
    """Calculate class weights from training samples.

    Weights are inversely proportional to class frequencies: w_j = N / (C * n_j).
    Only counts k_values from the provided samples to avoid information leakage.
    """
    num_classes = max_k - min_k + 1
    class_counts = np.zeros(num_classes)

    for sample in samples:
        k = sample.get("k_value")
        if k is not None and min_k <= int(k) <= max_k:
            class_counts[int(k) - min_k] += 1

    valid_classes = class_counts > 0
    if not np.any(valid_classes):
        logger.warning("No valid samples found in range [min_k, max_k]. Using uniform weights.")
        return None

    n_samples = np.sum(class_counts)
    weights = np.ones(num_classes)
    weights[valid_classes] = n_samples / (num_classes * class_counts[valid_classes])

    logger.info(f"Class weights calculated from {int(n_samples)} \n training samples: {weights}")
    return weights.tolist()


def _extract_k_from_sidecar_json(parent_dir: Path) -> int | None:
    """Read num_clusters from a JSON sidecar in the sample directory."""
    try:
        json_path = next(parent_dir.glob("*.json"))
    except StopIteration:
        return None

    try:
        with open(json_path) as f:
            data = json.load(f)
        value = data.get("cluster_config", {}).get("num_clusters")
        return int(value) if value is not None else None
    except Exception as e:
        logger.debug(f"Failed to read k_value from {json_path}: {e}")
        return None


def _discover_training_npz_files(data_dir: Path) -> list[Path]:
    """Find all normalized training NPZ files under data_dir."""
    return sorted(data_dir.rglob("*_training.npz"))


def _split_files_for_index(
    files: list[Path], test_ratio: float, seed: int
) -> tuple[list[tuple[Path, str]], int, int]:
    """Split files at file-level into train/test assignments."""
    rng = random.Random(seed)
    files_shuffled = files.copy()
    rng.shuffle(files_shuffled)

    n_test = int(len(files_shuffled) * test_ratio)
    n_train = len(files_shuffled) - n_test
    splits = [
        (file_path, "train" if i < n_train else "test")
        for i, file_path in enumerate(files_shuffled)
    ]
    return splits, n_train, n_test


def _derive_raw_relative_path(normalized_rel_path: Path) -> Path:
    """Map *_training.npz path to raw .npz path."""
    normalized_name = normalized_rel_path.name
    if normalized_name.endswith("_training.npz"):
        raw_name = normalized_name.replace("_training.npz", ".npz")
    else:
        raw_name = normalized_name.replace("_training", "", 1)
    return normalized_rel_path.with_name(raw_name)


def _scan_training_file(
    file_path: Path,
    split: str,
    data_dir: Path,
    k_cache: dict[Path, int | None],
) -> list[dict]:
    """Scan one training NPZ and return sample-level index records."""
    try:
        with np.load(file_path) as data:
            keys = set(data.keys())

        x_keys = sorted(k for k in keys if k.endswith("_X"))
        normalized_rel_path = file_path.relative_to(data_dir)
        raw_rel_path = _derive_raw_relative_path(normalized_rel_path)

        parent_dir = file_path.parent
        if parent_dir not in k_cache:
            k_cache[parent_dir] = _extract_k_from_sidecar_json(parent_dir)
        k_value = k_cache[parent_dir]

        records = []
        for key_x in x_keys:
            key_y = key_x.replace("_X", "_y")
            if key_y in keys:
                records.append(
                    {
                        "normalized_file_path": str(normalized_rel_path),
                        "raw_file_path": str(raw_rel_path),
                        "sample_key": key_x,
                        "target_key": key_y,
                        "k_value": k_value,
                        "split": split,
                    }
                )
        return records
    except Exception as e:
        logger.warning(f"Failed to scan {file_path}: {e}")
        return []


def _log_dataset_index_summary(df: pd.DataFrame, index_file: Path) -> None:
    """Log dataset index summary and output path."""
    logger.info(f"\n{'=' * 60}")
    logger.info("Created dataset index")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total samples: {len(df)}")

    if len(df) > 0:
        n_train_samples = (df["split"] == "train").sum()
        n_test_samples = (df["split"] == "test").sum()
        logger.info(f"Train samples: {n_train_samples} ({n_train_samples / len(df) * 100:.1f}%)")
        logger.info(f"Test samples: {n_test_samples} ({n_test_samples / len(df) * 100:.1f}%)")

    logger.info(f"\nSaved to: {index_file}")


def prepare_dataset_index(data_dir: Path, test_ratio: float = 0.2, seed: int = 42) -> None:
    """Create dataset index with file-level split and sample-level metadata."""
    logger.info(f"Creating dataset index for {data_dir}")

    files = _discover_training_npz_files(data_dir)
    if not files:
        logger.warning(f"No NPZ files found in {data_dir}")
        return

    logger.info(f"Found {len(files)} NPZ files. splitting...")
    file_splits, n_train, n_test = _split_files_for_index(files, test_ratio=test_ratio, seed=seed)
    logger.info(f"Files split: {n_train} train, {n_test} test")
    logger.info("Scanning files for samples...")

    k_cache: dict[Path, int | None] = {}
    records: list[dict] = []
    for file_path, split in tqdm(file_splits, desc="Scanning files"):
        records.extend(
            _scan_training_file(file_path, split=split, data_dir=data_dir, k_cache=k_cache)
        )

    df = pd.DataFrame(records)
    index_file = data_dir / "dataset_index.parquet"
    df.to_parquet(index_file, index=False)
    _log_dataset_index_summary(df, index_file=index_file)


def log_fold_results(fold_results: list[float]) -> None:
    """Log K-Fold cross-validation results summary."""
    logger.info("--- K-Fold Results ---")
    for i, mae in enumerate(fold_results):
        logger.info(f"Fold {i + 1}: MAE {mae:.6f}")

    if fold_results:
        avg_mae = sum(fold_results) / len(fold_results)
        logger.info(f"Average Best Val MAE: {avg_mae:.6f}")


def log_timing(label: str, duration: float) -> None:
    """Log timing information in seconds and minutes."""
    logger.info(f"{label}: {duration:.2f}s ({duration / 60:.2f}m)")


def extract_config_group_id(normalized_path: str) -> str:
    """
    Extract the unique structural config ID to prevent leakage.
    Example path: 'CesarComin/cfg000/CesarComin_cfg000_training.npz' -> 'CesarComin_cfg000'
    """
    parts = Path(normalized_path).parts
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return str(Path(normalized_path).parent)


def get_json_metadata(data_dir: Path, normalized_path: str) -> dict[str, Any]:
    """Find and load the corresponding .json sidecar for a given numpy file path."""
    p = Path(normalized_path)
    base_name = p.stem.replace("_training", "").replace("_predictions", "")
    json_path = data_dir / p.parent / f"{base_name}.json"

    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return {}


def build_unified_h5_datalake(data_dir: Path, parquet_file: str, out_h5_name: str) -> None:
    """
    Constructs a unified, leakage-proof HDF5 DataLake from Parquet, JSON, and NPZ logs.

    Args:
        data_dir: Root datasets directory.
        parquet_file: Path to the legacy parquet index file relative to data_dir.
        out_h5_name: Filename of the resulting HDF5 archive to be saved in data_dir.
    """
    parquet_path = data_dir / parquet_file
    h5_path = data_dir / out_h5_name

    logger.info(f"Loading base index from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    logger.info(f"Consolidating into Unified DataLake: {h5_path}...")

    with h5py.File(h5_path, "w") as h5f:
        root_group = h5f.create_group("datasets")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Constructing HDF5"):
            norm_rel_path = row["normalized_file_path"]
            raw_rel_path = row["raw_file_path"]
            sample_key = row["sample_key"]
            target_key = row.get("target_key", sample_key.replace("_X", "_y"))
            k_value = row["k_value"]
            split = row["split"]

            rep_id = sample_key.split("_")[0]
            config_group = extract_config_group_id(norm_rel_path)
            sample_id = f"{config_group}_{rep_id}"

            if sample_id in root_group:
                continue

            sample_group = root_group.create_group(sample_id)

            # Metadata Attributes
            sample_group.attrs["config_group_id"] = config_group
            sample_group.attrs["k_value"] = k_value
            sample_group.attrs["split"] = split
            sample_group.attrs["replication"] = rep_id

            metadata = get_json_metadata(data_dir, norm_rel_path)
            if metadata:
                sample_group.attrs["strategy"] = metadata.get("strategy_name", "unknown")
                if "cluster_config" in metadata:
                    sample_group.attrs["n_objects"] = metadata["cluster_config"].get(
                        "num_samples", -1
                    )
                    sample_group.attrs["n_dimensions"] = metadata["cluster_config"].get(
                        "num_dimensions", -1
                    )

            full_norm_path = data_dir / norm_rel_path
            full_raw_path = data_dir / raw_rel_path
            base_p = Path(norm_rel_path)
            base_name = base_p.stem.replace("_training", "")
            full_pred_path = data_dir / base_p.parent / f"{base_name}_predictions.npz"

            # Normalized Features & ARI Scores (_training.npz)
            try:
                with np.load(full_norm_path) as npz:
                    if sample_key in npz:
                        sample_group.create_dataset(
                            "normalized_features", data=npz[sample_key].astype(np.float32)
                        )
                    if target_key in npz:
                        sample_group.create_dataset(
                            "ari_scores", data=npz[target_key].astype(np.float32)
                        )
            except Exception as e:
                logger.warning(f"Failed loading training NPZ for {sample_id}: {e}")

            # Raw Features & True Labels (.npz)
            try:
                if full_raw_path.exists():
                    with np.load(full_raw_path) as npz:
                        if sample_key in npz:
                            sample_group.create_dataset(
                                "raw_features", data=npz[sample_key].astype(np.float32)
                            )
                        if target_key in npz:
                            sample_group.create_dataset(
                                "true_labels", data=npz[target_key].astype(np.int64)
                            )
            except Exception as e:
                logger.warning(f"Failed loading raw NPZ for {sample_id}: {e}")

            # Algorithm Predictions (_predictions.npz)
            try:
                if full_pred_path.exists():
                    with np.load(full_pred_path) as npz:
                        if sample_key in npz:
                            sample_group.create_dataset(
                                "algorithm_predictions", data=npz[sample_key].astype(np.int64)
                            )
            except Exception:
                pass

        logger.info("DataLake Construction Complete!")
        logger.info(f"Total Unified Samples created: {len(root_group.keys())}")
