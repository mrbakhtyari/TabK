import datetime
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from .config import AppConfig
from .head import KHead
from .model import DoubleInvariantTransformer

logger = logging.getLogger(__name__)


def create_model(config: AppConfig) -> DoubleInvariantTransformer:
    head = KHead(config.model.d_model, config.head_config)
    return DoubleInvariantTransformer(config.model, head=head)


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
    """Save model checkpoint with comprehensive metadata."""

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
    """Plots training and validation loss for each fold."""

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
    """Saves training history to a JSON file."""
    with open(save_path, "w") as f:
        json.dump(all_histories, f, indent=2)
    logger.info(f"Training history saved to {save_path}")


def save_run_manifest(config, metrics, path) -> None:
    """Saves a run manifest JSON sidecar."""
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
