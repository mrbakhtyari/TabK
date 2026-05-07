import argparse
import logging
from pathlib import Path

from tabk.architecture import (
    AppConfig,
    KEstimatorConfig,
    ModelConfig,
    TrainingConfig,
    run_training_pipeline,
)
from tabk.utils import configure_logging

logger = logging.getLogger(__name__)

MODEL_TYPES = [
    "default",
    "qfe_simple",
    "pool_mean",
    "no_col_attn",
]


def main():
    parser = argparse.ArgumentParser(
        description="Train KEstimator (Cluster Number Prediction)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model Architecture
    parser.add_argument("--d-model", type=int, default=128, help="Model embedding dimension")
    parser.add_argument("--n-head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--num-bins", type=int, default=50, help="Number of QFE bins")
    parser.add_argument(
        "--model-type",
        type=str,
        default="default",
        choices=MODEL_TYPES,
        help="Backbone type to train",
    )
    parser.add_argument(
        "--fla-backend",
        type=str,
        default="flash_linear_attention",
        choices=["auto", "flash_linear_attention", "torch_linear"],
        help="Linear attention backend used by dit_fla_v1",
    )
    parser.add_argument(
        "--fla-allow-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow fallback to torch linear attention if flash-linear-attention is unavailable",
    )
    parser.add_argument(
        "--fla-eps",
        type=float,
        default=1e-6,
        help="Numerical stabilization epsilon for torch linear attention backend",
    )

    # K-Estimator Head
    parser.add_argument("--min-k", type=int, default=2, help="Minimum number of clusters")
    parser.add_argument("--max-k", type=int, default=15, help="Maximum number of clusters")
    parser.add_argument(
        "--k-head-mode",
        type=str,
        default="distribution",
        choices=["distribution", "classification", "focal", "ordinal", "regression"],
        help="K-estimator head mode",
    )
    parser.add_argument(
        "--sigma", type=float, default=0.5, help="Gaussian sigma for distribution mode"
    )

    # Training
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--accum-steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Dataset & Output
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="datasets",
        help="Data root directory",
    )
    parser.add_argument(
        "--h5-filename", type=str, default="synthetic_n40000_r1.h5", help="HDF5 DataLake filename"
    )
    parser.add_argument(
        "--output-dir", type=Path, default="results/k_estimator", help="Output directory"
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Verbosity")

    args = parser.parse_args()
    configure_logging(args.verbose)

    app_config = AppConfig(
        head_config=KEstimatorConfig(
            min_k=args.min_k,
            max_k=args.max_k,
            mode=args.k_head_mode,
            sigma=args.sigma,
        ),
        model=ModelConfig(
            d_model=args.d_model,
            n_head=args.n_head,
            n_layers=args.n_layers,
            dropout=args.dropout,
            num_bins=args.num_bins,
            model_type=args.model_type,
            fla_backend=args.fla_backend,
            fla_allow_fallback=args.fla_allow_fallback,
            fla_eps=args.fla_eps,
        ),
        training=TrainingConfig(
            batch_size=args.batch_size,
            accum_steps=args.accum_steps,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            k_folds=args.k_folds,
            patience=args.patience,
            num_workers=args.num_workers,
            seed=args.seed,
        ),
    )

    logger.info("Configuration built successfully.")

    run_training_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=app_config,
        h5_filename=args.h5_filename,
    )


if __name__ == "__main__":
    main()
