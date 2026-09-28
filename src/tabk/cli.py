import argparse
from pathlib import Path

import numpy as np

from .architecture import AppConfig, HeadConfig, ModelConfig, TrainingConfig
from .utils import configure_logging

DATA_DIR = Path("datasets/synthetic")
MODEL_DIR = Path("models/TabK_retrained")
HUB_DIR = Path("hub/TabK_retrained")
DATALAKE_NAME = "datalake.h5"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tabk", description="Estimate the number of clusters in tabular data."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="count", default=0)
    fmt = argparse.ArgumentDefaultsHelpFormatter

    predict = commands.add_parser(
        "predict", parents=[common], formatter_class=fmt, help="Estimate k for a CSV table"
    )
    predict.add_argument("table", type=Path, help="CSV file with one row per sample")
    predict.add_argument("--model", default=None, help="Model directory or Hub repo id")
    predict.add_argument("--device", default=None)
    predict.add_argument("--no-scale", action="store_true", help="Skip feature standardization")
    predict.add_argument("--proba", action="store_true", help="Print the probability of each k")
    predict.set_defaults(func=_predict)

    generate = commands.add_parser(
        "generate", parents=[common], formatter_class=fmt, help="Generate the training data"
    )
    generate.add_argument("--out", type=Path, default=DATA_DIR)
    generate.add_argument("--n-configs", type=int, default=40000)
    generate.add_argument("--n-repeats", type=int, default=1)
    generate.add_argument("--k-min", type=int, default=2)
    generate.add_argument("--k-max", type=int, default=15)
    generate.add_argument("--n-low", type=int, default=100)
    generate.add_argument("--n-high", type=int, default=2500)
    generate.add_argument("--d-low", type=int, default=2)
    generate.add_argument("--d-high", type=int, default=200)
    generate.add_argument("--strategies", nargs="+", default=None)
    generate.add_argument("--timeout", type=float, default=30.0)
    generate.add_argument("--test-ratio", type=float, default=0.1)
    generate.add_argument("--seed", type=int, default=42)
    generate.set_defaults(func=_generate)

    model_cfg, head_cfg, train_cfg = ModelConfig(), HeadConfig(), TrainingConfig()
    train = commands.add_parser(
        "train", parents=[common], formatter_class=fmt, help="Train a TabK ensemble"
    )
    train.add_argument("--data", type=Path, default=DATA_DIR)
    train.add_argument("--out", type=Path, default=MODEL_DIR)
    train.add_argument("--d-model", type=int, default=model_cfg.d_model)
    train.add_argument("--n-head", type=int, default=model_cfg.n_head)
    train.add_argument("--n-layers", type=int, default=model_cfg.n_layers)
    train.add_argument("--dropout", type=float, default=model_cfg.dropout)
    train.add_argument("--num-bins", type=int, default=model_cfg.num_bins)
    train.add_argument("--min-k", type=int, default=head_cfg.min_k)
    train.add_argument("--max-k", type=int, default=head_cfg.max_k)
    train.add_argument("--sigma", type=float, default=head_cfg.sigma)
    train.add_argument("--lr", type=float, default=train_cfg.learning_rate)
    train.add_argument("--weight-decay", type=float, default=train_cfg.weight_decay)
    train.add_argument("--batch-size", type=int, default=train_cfg.batch_size)
    train.add_argument("--accum-steps", type=int, default=train_cfg.accum_steps)
    train.add_argument("--epochs", type=int, default=train_cfg.epochs)
    train.add_argument("--k-folds", type=int, default=train_cfg.k_folds)
    train.add_argument("--patience", type=int, default=train_cfg.patience)
    train.add_argument("--device", default=train_cfg.device)
    train.add_argument("--seed", type=int, default=train_cfg.seed)
    train.set_defaults(func=_train)

    export = commands.add_parser(
        "export", parents=[common], formatter_class=fmt, help="Convert a model for the Hub"
    )
    export.add_argument("--model", type=Path, default=MODEL_DIR)
    export.add_argument("--out", type=Path, default=HUB_DIR)
    export.set_defaults(func=_export)

    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"tabk {args.command}: error: {exc}\n")


def _predict(args: argparse.Namespace) -> None:
    from .architecture import TabK
    from .architecture.inference import PRETRAINED_REPO_ID

    X = read_csv(args.table)
    model = TabK.from_pretrained(args.model or PRETRAINED_REPO_ID, device=args.device)
    if args.proba:
        for k, p in zip(model.k_values, model.predict_proba(X, scale=not args.no_scale)):
            print(f"{k}\t{p:.4f}")
    else:
        print(model.predict(X, scale=not args.no_scale))


def _generate(args: argparse.Namespace) -> None:
    from .synthesis import GenerationSettings, build_h5_from_raw, default_registry, run_generation

    strategies = None
    if args.strategies:
        specs = {spec.name: spec for spec in default_registry()}
        unknown = sorted(set(args.strategies) - set(specs))
        if unknown:
            raise ValueError(f"Unknown strategies: {', '.join(unknown)}")
        strategies = [specs[name] for name in args.strategies]

    settings = GenerationSettings(
        n_repeats=args.n_repeats,
        master_seed=args.seed,
        output_dir=args.out,
        timeout=args.timeout,
        n_configs=args.n_configs,
        k_min=args.k_min,
        k_max=args.k_max,
        n_low=args.n_low,
        n_high=args.n_high,
        d_low=args.d_low,
        d_high=args.d_high,
    )
    run_generation(settings, strategies=strategies)
    build_h5_from_raw(args.out, out_h5=DATALAKE_NAME, test_ratio=args.test_ratio, seed=args.seed)
    print(f"Training data written to {args.out}")


def _train(args: argparse.Namespace) -> None:
    from .architecture import run_training_pipeline

    config = AppConfig(
        head_config=HeadConfig(min_k=args.min_k, max_k=args.max_k, sigma=args.sigma),
        model=ModelConfig(
            d_model=args.d_model,
            n_head=args.n_head,
            n_layers=args.n_layers,
            dropout=args.dropout,
            num_bins=args.num_bins,
        ),
        training=TrainingConfig(
            batch_size=args.batch_size,
            accum_steps=args.accum_steps,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            k_folds=args.k_folds,
            patience=args.patience,
            seed=args.seed,
            device=args.device,
        ),
    )
    run_training_pipeline(args.data, args.out, config, h5_filename=DATALAKE_NAME)
    print(f"Model written to {args.out}")


def _export(args: argparse.Namespace) -> None:
    from .architecture.export import export_for_hub

    n_folds = export_for_hub(args.model, args.out)
    print(f"Exported {n_folds} folds to {args.out}")


def read_csv(path: Path) -> np.ndarray:
    """Read a numeric CSV table, skipping the header row if there is one."""
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    with open(path, encoding="utf-8") as f:
        first_row = f.readline().strip().split(",")
    has_header = not all(_is_number(value) for value in first_row)
    return np.loadtxt(path, delimiter=",", skiprows=int(has_header), ndmin=2)


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True
