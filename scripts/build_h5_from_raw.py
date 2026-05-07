"""
Build a unified HDF5 DataLake directly from raw generated NPZ files.

Skips the clustering benchmark pipeline entirely — only needs the raw
features and JSON sidecar metadata. Applies StandardScaler normalization
inline and writes the result in the same schema that H5Dataset / train.py
expect.

Usage:
    python scripts/build_h5_from_raw.py --data-dir datasets/synthetic_n40000_r1
"""

import argparse
from pathlib import Path

from tabk.synthesis import build_h5_from_raw
from tabk.utils import configure_logging


def main():
    parser = argparse.ArgumentParser(
        description="Build unified HDF5 DataLake from raw generated NPZ files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing strategy subdirectories with NPZ files.",
    )
    parser.add_argument(
        "--out-h5",
        type=str,
        default=None,
        help="Name of output HDF5 file (default: {parent_dir_name}.h5).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples to hold out as test (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase logging verbosity (-v, -vv)."
    )
    args = parser.parse_args()

    configure_logging(args.verbose)

    build_h5_from_raw(
        data_dir=args.data_dir,
        out_h5=args.out_h5,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
