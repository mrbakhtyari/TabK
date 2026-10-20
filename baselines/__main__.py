import argparse
from pathlib import Path

from tabk.cli import read_csv
from tabk.utils import apply_standard_scaling

from . import METHODS, estimate_k


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the number of clusters with classical baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("table", type=Path, help="CSV file with one row per sample")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=15)
    parser.add_argument("--no-scale", action="store_true", help="Skip feature standardization")
    args = parser.parse_args(argv)
    try:
        if args.k_min < 2 or args.k_max < args.k_min:
            raise ValueError("Require 2 <= k-min <= k-max")
        X = read_csv(args.table)
        if not args.no_scale:
            X = apply_standard_scaling(X)
        results = estimate_k(X, list(range(args.k_min, args.k_max + 1)), args.methods)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"baselines: error: {exc}\n")
    for method, (predicted_k, seconds) in results.items():
        print(f"{method}\t{predicted_k}\t{seconds:.6f}")


if __name__ == "__main__":
    main()
