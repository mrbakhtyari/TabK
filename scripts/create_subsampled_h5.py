import argparse
import logging
import random
from pathlib import Path

import h5py

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_subsampled_h5(source_path: Path, fractions: list[float], seed: int = 42):
    """
    Creates new HDF5 files containing external links to the original datasets,
    simulating subsampling of the train set without duplicating large data.
    The test set is always included in full.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    logger.info(f"Scanning source HDF5: {source_path}")
    source_filename = source_path.name

    train_keys = []
    test_keys = []

    with h5py.File(source_path, "r") as f:
        if "datasets" not in f:
            raise KeyError("Source HDF5 must contain a top-level 'datasets' group")

        grp = f["datasets"]
        for k in grp.keys():
            if grp[k].attrs.get("split") == "train":
                train_keys.append(k)
            else:
                test_keys.append(k)

    num_train = len(train_keys)
    num_test = len(test_keys)
    logger.info(f"Found {num_train} train samples and {num_test} test samples in '{source_path}'")

    # Strictly sort keys prior to shuffle to completely guarantee cross-platform determinism
    train_keys.sort()
    test_keys.sort()

    # Seed the random number generator
    rng = random.Random(seed)
    rng.shuffle(train_keys)

    # Sort fractions ascending so we can create progressively larger files
    fractions = sorted(fractions)

    for fraction in fractions:
        if fraction < 0 or fraction > 1.0:
            logger.warning(f"Fraction {fraction} is out of bounds [0, 1]. Skipping.")
            continue

        actual_train_size = int(num_train * fraction)
        subset_keys = train_keys[:actual_train_size] + test_keys
        frac_percent = int(fraction * 100)

        target_name = f"{source_path.stem}_frac{frac_percent}{source_path.suffix}"
        target_path = source_path.parent / target_name

        logger.info(
            f"Creating from {target_path} (train size: {actual_train_size}, test size: {num_test})"
        )

        with h5py.File(target_path, "w") as tf:
            root = tf.create_group("datasets")
            for sid in subset_keys:
                # Create an external link matching the internal structure
                root[sid] = h5py.ExternalLink(source_filename, f"/datasets/{sid}")

        logger.info(f"Successfully generated: {target_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HDF5 subsets via External Links for scaling laws."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the original HDF5 file (e.g. datasets/synthetic_n40000_r1.h5)",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        required=True,
        help="List of train set fractions to generate (e.g. 0.1 0.25 0.5 0.75)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic nested shuffling"
    )

    args = parser.parse_args()
    create_subsampled_h5(args.source, args.fractions, args.seed)


if __name__ == "__main__":
    main()
