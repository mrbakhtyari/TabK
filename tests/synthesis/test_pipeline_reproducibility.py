import filecmp
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def run_generation_cli(output_dir: Path, n_configs: int, seed: int, n_repeats: int = 1) -> float:
    """
    Runs the generation script via subprocess. Returns execution time.
    """
    cmd = [
        sys.executable,
        "scripts/generate_datasets.py",
        "--output-dir",
        str(output_dir),
        "--n-configs",
        str(n_configs),
        "--seed",
        str(seed),
        "--n-repeats",
        str(n_repeats),
        # Use smaller bounds for faster verification
        "--n-low",
        "100",
        "--n-high",
        "500",
        "--d-low",
        "2",
        "--d-high",
        "5",  # Keep dimensions low to increase chance of d=2
    ]

    logger.info(f"Running generation: {' '.join(cmd)}")
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.perf_counter() - start

    if result.returncode != 0:
        logger.error(f"Generation failed:\n{result.stderr}")
        raise RuntimeError("Generation script failed")

    return duration


def verify_directories_identical(dir1: Path, dir2: Path) -> bool:
    """
    Recursively compare two directories to ensure they contain identical files.
    """
    logger.info(f"Comparing directories:\n A: {dir1}\n B: {dir2}")

    if not dir1.exists() or not dir2.exists():
        logger.error("One or both directories do not exist.")
        return False

    # Compare common files
    dc = filecmp.dircmp(dir1, dir2)

    if dc.left_only:
        logger.error(f"Files only in {dir1}: {dc.left_only}")
        return False
    if dc.right_only:
        logger.error(f"Files only in {dir2}: {dc.right_only}")
        return False

    # Compare contents of common files
    if dc.diff_files:
        # Check if the differences are only in JSON metadata (timestamps)
        real_diffs = []
        for filename in dc.diff_files:
            if filename == "generation.log":
                continue

            # Skip PDF comparison (binary files with metadata)
            if filename.endswith(".pdf"):
                continue

            if filename == "timeouts.log":
                # Compare sorted lines to handle multiprocessing order
                # Also strip timestamps
                try:
                    with open(dir1 / filename) as f1, open(dir2 / filename) as f2:
                        lines1 = f1.readlines()
                        lines2 = f2.readlines()

                    # Strip timestamp (first 21 chars: "YYYY-MM-DD HH:MM:SS - ")
                    # Or split by " - "
                    def clean_line(line):
                        parts = line.split(" - ", 1)
                        if len(parts) > 1:
                            return parts[1]
                        return line

                    lines1 = sorted([clean_line(l) for l in lines1])
                    lines2 = sorted([clean_line(l) for l in lines2])

                    if lines1 != lines2:
                        logger.error(f"Timeouts log mismatch in {filename}")
                        real_diffs.append(filename)
                except Exception as e:
                    logger.error(f"Error comparing timeouts log {filename}: {e}")
                    real_diffs.append(filename)
                continue

            if filename.endswith(".json"):
                # Load and compare content ignoring created_at and other metadata
                try:
                    with open(dir1 / filename) as f1, open(dir2 / filename) as f2:
                        j1 = json.load(f1)
                        j2 = json.load(f2)

                    # Ignore fields that are expected to differ
                    for key in [
                        "created_at",
                        "timestamp",
                        "pipeline_duration",
                        "total_gen_time",
                        "output_dir",
                    ]:
                        j1.pop(key, None)
                        j2.pop(key, None)

                    if j1 != j2:
                        real_diffs.append(filename)
                        logger.error(f"JSON content mismatch in {filename}")
                except Exception as e:
                    logger.error(f"Error comparing JSONs {filename}: {e}")
                    real_diffs.append(filename)
            elif filename.endswith(".npz"):
                # Compare NPZ content
                try:
                    d1 = np.load(dir1 / filename)
                    d2 = np.load(dir2 / filename)
                    keys1 = set(d1.keys())
                    keys2 = set(d2.keys())
                    if keys1 != keys2:
                        logger.error(f"NPZ keys mismatch in {filename}: {keys1 ^ keys2}")
                        real_diffs.append(filename)
                        continue

                    diff_keys = []
                    for k in keys1:
                        if not np.array_equal(d1[k], d2[k]):
                            diff_keys.append(k)

                    if diff_keys:
                        logger.error(f"NPZ content mismatch in {filename} keys: {diff_keys}")
                        real_diffs.append(filename)
                except Exception as e:
                    logger.error(f"Error comparing NPZs {filename}: {e}")
                    real_diffs.append(filename)
            else:
                real_diffs.append(filename)

        if real_diffs:
            logger.error(f"Files with different content: {real_diffs}")
            return False

    # Recursively compare subdirectories
    for subdir in dc.common_dirs:
        if not verify_directories_identical(dir1 / subdir, dir2 / subdir):
            return False

    return True


def test_pipeline_reproducibility():
    """
    Verifies that the entire generation pipeline is reproducible.
    Runs the generation twice with the same seed and checks if the output directories are identical.
    """
    n_configs = 100
    n_repeats = 10
    seed = 42

    # Create temporary directories for the two runs
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        run1_dir = base_dir / "run_1"
        run2_dir = base_dir / "run_2"

        # Run 1
        run_generation_cli(run1_dir, n_configs, seed, n_repeats)

        # Run 2
        run_generation_cli(run2_dir, n_configs, seed, n_repeats)

        # Verify
        is_identical = verify_directories_identical(run1_dir, run2_dir)

        assert is_identical, "Pipeline generation is not reproducible! Output directories differ."
