import csv
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from .core import ClusterConfig
from .sampling import STRATEGY_PARAMS_PATH
from .visualization import plot_coverage, plot_distributions

logger = logging.getLogger(__name__)


def save_strategy_params(output_dir: Path) -> None:
    """
    Save the strategy parameter descriptions to a JSON file.
    """
    try:
        # Copy params.json from source to output directory
        dst_params = output_dir / STRATEGY_PARAMS_PATH.name
        shutil.copy(STRATEGY_PARAMS_PATH, dst_params)
        logger.info(f"Saved strategy parameters to {dst_params}")
    except Exception as e:
        logger.error(f"Failed to save strategy parameters: {e}")


def save_configs_csv(configs: list[ClusterConfig], output_dir: Path) -> None:
    """
    Save the list of ClusterConfig objects to a CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "generated_configs.csv"

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["config_id", "num_clusters", "num_samples", "num_dimensions"])
            for i, cfg in enumerate(configs):
                writer.writerow([i, cfg.num_clusters, cfg.num_samples, cfg.num_dimensions])
        logger.info(f"Saved generated configs to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save configs CSV: {e}")


def config_report(configs: list[ClusterConfig], output_dir: Path, seed: int = 42) -> None:
    """
    Generate a report for the generated configurations, including a CSV file
    and visualization plots.
    """
    logger.info("Generating synthesis report...")

    # Save CSV
    save_configs_csv(configs, output_dir)

    # Prepare data for plotting
    k = np.array([c.num_clusters for c in configs])
    n = np.array([c.num_samples for c in configs])
    d = np.array([c.num_dimensions for c in configs])

    # Generate Plots
    try:
        plot_distributions(k, n, d, save_dir=output_dir, show=False)
        plot_coverage(k, n, d, save_dir=output_dir, show=False, seed=seed)
    except Exception as e:
        logger.error(f"Failed to generate synthesis plots: {e}")


def save_timeout_log(timeout_events: list[str], output_dir: Path) -> None:
    """
    Save a log of timeout events to a file.
    """
    if not timeout_events:
        return

    try:
        timeout_file = output_dir / "timeouts.log"
        with open(timeout_file, "w") as f:
            for event in timeout_events:
                f.write(f"{event}\n")
        logger.info(f"Timeout log saved to {timeout_file}")
    except Exception as e:
        logger.error(f"Failed to save timeout log: {e}")


def save_run_metadata(
    settings_dict: dict,
    pipeline_duration: float,
    total_gen_time: float,
    output_dir: Path,
) -> None:
    """
    Save run metadata including settings and timing information to a JSON file.
    """
    try:
        # Create a copy to avoid modifying the original dict
        metadata = settings_dict.copy()

        # Ensure Path objects are serialized as strings
        if "output_dir" in metadata and isinstance(metadata["output_dir"], Path):
            metadata["output_dir"] = str(metadata["output_dir"])

        metadata.update(
            {
                "timestamp": datetime.now().isoformat(),
                "pipeline_duration": pipeline_duration,
                "total_gen_time": total_gen_time,
            }
        )

        metadata_file = output_dir / "run_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save run metadata: {e}")


def save_dataset_manifest(output_dir: Path) -> None:
    """
    Scan the output directory for generated .npz datasets and write their
    absolute paths to a manifest file (dataset_paths.txt).
    """
    try:
        logger.info(f"Scanning {output_dir} for generated .npz datasets...")
        npz_files = sorted(output_dir.rglob("*.npz"))

        if npz_files:
            manifest_path = output_dir / "dataset_paths.txt"
            with open(manifest_path, "w") as f:
                for p in npz_files:
                    f.write(str(p.resolve()) + "\n")

            logger.info(f"Successfully wrote {len(npz_files)} dataset paths to {manifest_path}")
        else:
            logger.warning("No .npz files found after generation; manifest not created.")

    except Exception as e:
        logger.error(f"Failed to save dataset manifest: {e}")
