"""
Training module: single-epoch training, validation, per-fold training, and K-fold orchestration.
All data is loaded from the unified HDF5 DataLake — no parquet or NPZ dependencies remain.
"""

import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..utils.progress import is_interactive_stream, write_progress_line
from .config import AppConfig
from .dataset import H5Dataset, collate_batch, scan_h5_datalake
from .utils import calculate_class_weights, create_model, save_checkpoint

logger = logging.getLogger(__name__)


def validate(model, loader, criterion, config: AppConfig, amp_dtype=torch.float16):
    """Evaluate the model on a validation DataLoader. Returns per-sample averaged metrics."""
    model.eval()
    total_loss = 0.0
    metric_sums: dict[str, float] = {}
    total_samples = 0

    device = config.training.torch_device
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for x, y, r_mask, c_mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            r_mask = r_mask.to(device, non_blocking=True)
            c_mask = c_mask.to(device, non_blocking=True)
            bs = x.size(0)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(x, r_mask, c_mask)
                loss = criterion(logits, y)

            batch_metrics = model.head.compute_metrics(logits, y)
            total_loss += loss.item() * bs
            for name, value in batch_metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value) * bs
            total_samples += bs

    metrics = {"val_loss": total_loss / total_samples}
    for name, value in metric_sums.items():
        metrics[f"val_{name}"] = value / total_samples
    if "val_mse" in metrics:
        metrics["val_rmse"] = math.sqrt(metrics["val_mse"])

    return metrics


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    epoch_idx,
    fold_idx,
    config: AppConfig,
    scaler=None,
    scheduler=None,
    amp_dtype=torch.float16,
):
    """Train the model for one epoch with gradient accumulation and optional AMP."""
    model.train()
    total_loss = 0
    total_samples = 0

    device = config.training.torch_device
    use_amp = device.type == "cuda"
    use_scaler = scaler is not None and scaler.is_enabled()
    accum_steps = config.training.accum_steps
    num_batches = len(loader)

    remainder = num_batches % accum_steps
    last_window_size = remainder if remainder != 0 else accum_steps
    last_window_start = num_batches - last_window_size
    progress_desc = f"Fold {fold_idx + 1} | Epoch {epoch_idx + 1}/{config.training.epochs}"
    use_live_progress = is_interactive_stream()

    if use_live_progress:
        progress_iter = tqdm(loader, desc=progress_desc, total=num_batches)
    else:
        progress_iter = loader

    optimizer.zero_grad(set_to_none=True)

    for i, (x, y, r_mask, c_mask) in enumerate(progress_iter):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        r_mask = r_mask.to(device, non_blocking=True)
        c_mask = c_mask.to(device, non_blocking=True)

        current_accum = last_window_size if i >= last_window_start else accum_steps

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            logits = model(x, r_mask, c_mask)
            loss = criterion(logits, y)

        scaled_loss = loss / accum_steps

        if use_scaler:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        is_step = (i + 1) % accum_steps == 0 or (i + 1) == num_batches

        if is_step:
            if use_scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skip_lr_sched = scale_before > scaler.get_scale()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                skip_lr_sched = False

            optimizer.zero_grad(set_to_none=True)
            if scheduler and not skip_lr_sched:
                scheduler.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

        if use_live_progress:
            progress_iter.set_postfix(loss=f"{loss.item():.4f}")

    if use_live_progress:
        progress_iter.close()

    return total_loss / total_samples


def train_fold(
    train_loader,
    val_loader,
    fold_idx,
    output_dir,
    config: AppConfig,
):
    """Train a fresh model for a single fold. Returns (best_val_mae, history)."""
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = config.training.torch_device
    model = create_model(config).to(device)

    decay_params = [p for _, p in model.named_parameters() if p.dim() > 1]
    no_decay_params = [p for _, p in model.named_parameters() if p.dim() <= 1]
    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.training.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.training.learning_rate,
    )

    batches_per_epoch = len(train_loader)
    accum_steps = config.training.accum_steps
    steps_per_epoch = (batches_per_epoch + accum_steps - 1) // accum_steps
    total_steps = steps_per_epoch * config.training.epochs

    warmup_steps = max(100, int(0.05 * total_steps))
    main_steps = total_steps - warmup_steps

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=main_steps, eta_min=config.training.learning_rate * 0.1
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    criterion = model.head.get_loss_fn().to(device)

    # BF16 on Ampere+ (no scaler needed); FP16+GradScaler on older GPUs
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not use_bf16)

    best_val_mae = float("inf")
    history = {"train_loss": [], "val_loss": []}
    counter = 0
    best_checkpoint_path = None
    use_live_progress = is_interactive_stream()

    for epoch in range(config.training.epochs):
        if not use_live_progress:
            write_progress_line(
                f"Fold {fold_idx + 1} | Epoch {epoch + 1}/{config.training.epochs} | start"
            )

        avg_train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            epoch,
            fold_idx,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            amp_dtype=amp_dtype,
        )

        val_metrics = validate(model, val_loader, criterion, config=config, amp_dtype=amp_dtype)
        avg_val_loss = val_metrics["val_loss"]
        val_mae = val_metrics.get("val_mae", avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        for k, v in val_metrics.items():
            if k not in history:
                history[k] = []
            history[k].append(v)

        metric_summary = " | ".join(
            f"{k.replace('val_', '').upper()}: {v:.4f}"
            for k, v in val_metrics.items()
            if k.startswith("val_") and k != "val_loss"
        )
        if use_live_progress:
            logger.info(
                f"Fold {fold_idx + 1} | Epoch {epoch + 1}/{config.training.epochs} | "
                f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
                + (f" | {metric_summary}" if metric_summary else "")
            )
        else:
            write_progress_line(
                f"Fold {fold_idx + 1} | Epoch {epoch + 1}/{config.training.epochs} | end | "
                f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
                + (f" | {metric_summary}" if metric_summary else "")
            )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_checkpoint_path = Path(output_dir) / f"checkpoint_fold_{fold_idx + 1}.pth"

            checkpoint_info = dict(val_metrics)
            head_config = config.head_config
            if hasattr(head_config, "class_weights") and head_config.class_weights is not None:
                checkpoint_info["class_weights"] = head_config.class_weights

            save_checkpoint(
                model,
                optimizer,
                epoch,
                avg_val_loss,
                best_checkpoint_path,
                fold_idx=fold_idx,
                config=config,
                additional_info=checkpoint_info,
            )
            counter = 0
        else:
            counter += 1
            if counter >= config.training.patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

    if best_checkpoint_path and best_checkpoint_path.exists():
        logger.info(f"Restoring best model from {best_checkpoint_path.name}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    return best_val_mae, history


def execute_kfold_training(
    h5_path: Path | str,
    output_dir: Path,
    config: AppConfig,
    base_split: str = "train",
) -> tuple[list[float], list[dict], list[float]]:
    """Execute GroupKFold cross-validation directly from the HDF5 DataLake.

    Groups are defined by `config_group_id` so that datasets sharing the same
    structural parameters (n_clusters, n_dimensions, n_objects) never leak
    across train/validation folds.

    Data is cached into RAM once; each fold uses torch Subset views.
    """
    all_splits = scan_h5_datalake(h5_path)
    metadata = all_splits.get(base_split, [])
    if not metadata:
        raise ValueError(f"No samples found in HDF5 for split: {base_split}")

    sample_ids = [m["sample_id"] for m in metadata]
    k_values = [m["k_value"] for m in metadata]

    logger.info(f"{config.training.k_folds}-Fold StratifiedKFold on {len(sample_ids)} samples")

    # Cache ALL training data into RAM once (avoid re-reading per fold)
    full_dataset = H5Dataset(h5_path, sample_ids, config)

    kf = StratifiedKFold(
        n_splits=config.training.k_folds, shuffle=True, random_state=config.training.seed
    )
    dummy_x = np.arange(len(sample_ids))

    fold_results: list[float] = []
    all_histories: list[dict] = []
    fold_times: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dummy_x, k_values)):
        fold_start = time.time()
        logger.info(f"\nFold {fold + 1}/{config.training.k_folds}")

        # Compute class weights from training fold only (leakage-safe)
        head_config = config.head_config
        if getattr(head_config, "needs_class_weights", False):
            train_meta = [metadata[i] for i in train_idx]
            head_config.class_weights = calculate_class_weights(
                [{"k_value": m["k_value"]} for m in train_meta],
                head_config.min_k,
                head_config.max_k,
            )

        # Subset views — no data copy, no re-read
        train_subset = Subset(full_dataset, train_idx.tolist())
        val_subset = Subset(full_dataset, val_idx.tolist())

        common = dict(collate_fn=collate_batch, pin_memory=True)
        train_loader = DataLoader(
            train_subset, batch_size=config.training.batch_size, shuffle=True, **common
        )
        val_loader = DataLoader(
            val_subset, batch_size=config.training.batch_size, shuffle=False, **common
        )

        best_val_mae, history = train_fold(
            train_loader,
            val_loader,
            fold,
            output_dir,
            config=config,
        )

        fold_duration = time.time() - fold_start
        fold_results.append(best_val_mae)
        all_histories.append(history)
        fold_times.append(fold_duration)

        logger.info(
            f"Fold {fold + 1} Best Val MAE: {best_val_mae:.6f} (Duration: {fold_duration:.2f}s)"
        )

    return fold_results, all_histories, fold_times
