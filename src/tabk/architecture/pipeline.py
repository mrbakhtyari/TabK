import logging
import time
from pathlib import Path

from .config import AppConfig
from .train import execute_kfold_training
from .utils import (
    log_device_info,
    log_fold_results,
    log_timing,
    log_training_configuration,
    plot_training_history,
    save_run_manifest,
    save_training_history,
    set_seed,
)

logger = logging.getLogger(__name__)


def run_training_pipeline(
    data_dir: Path,
    output_dir: Path,
    config: AppConfig,
    h5_filename: str,
) -> tuple[list[float], list[dict], list[float]]:
    """Complete training pipeline backed by the unified HDF5 DataLake.

    Args:
        data_dir: Root directory containing the HDF5 file.
        output_dir: Directory for checkpoints, plots, and history.
        config: Application configuration.
        h5_filename: Name of the HDF5 DataLake file inside data_dir.
    """
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.training.seed)
    logger.info("\nStarting training...")
    log_device_info(config)
    log_training_configuration(config)

    h5_path = data_dir / h5_filename
    if not h5_path.is_file():
        raise FileNotFoundError(
            f"HDF5 DataLake not found at: {h5_path}. Run scripts/build_h5_from_raw.py first."
        )

    logger.info(f"\nStarting {config.training.k_folds}-fold training from {h5_path}...")
    training_start_time = time.time()

    fold_results, all_histories, fold_times = execute_kfold_training(
        h5_path=h5_path,
        output_dir=output_dir,
        config=config,
        base_split="train",
    )

    logger.info("\nTraining complete")

    training_duration = time.time() - training_start_time
    log_timing("Training loop duration", training_duration)
    log_fold_results(fold_results)

    save_training_history(all_histories, str(output_dir / "training_history.json"))
    plot_training_history(all_histories, str(output_dir))

    total_time = time.time() - start_time
    log_timing("Total execution time", total_time)

    metrics = {
        "fold_durations_seconds": fold_times,
        "total_training_duration_seconds": training_duration,
        "total_execution_time_seconds": total_time,
    }
    save_run_manifest(
        config=config.to_dict(), metrics=metrics, path=str(output_dir / "manifest.json")
    )

    logger.info(f"\nResults saved to: {output_dir.absolute()}")
    return fold_results, all_histories, fold_times
