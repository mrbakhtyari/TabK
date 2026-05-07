import logging
import time
from pathlib import Path

import numpy as np
import torch

from .config import AppConfig
from .utils import create_model

logger = logging.getLogger(__name__)


def load_inference_context(
    model_dir: str | Path,
    device: str | torch.device | None = None,
) -> dict:
    """Load ensemble models and return an inference artifact bundle."""
    models = _load_models(model_dir, device)
    if not models:
        raise ValueError(f"No models loaded from {model_dir}")

    config = getattr(models[0], "app_config", None)
    runtime_device = next(models[0].parameters()).device

    return {
        "models": models,
        "config": config,
        "device": runtime_device,
    }


def _load_models(
    model_dir: str | Path,
    device: str | torch.device | None = None,
) -> list[torch.nn.Module]:
    models: list[torch.nn.Module] = []
    start_dir = Path(model_dir) / "checkpoints"
    model_files = sorted(start_dir.glob("*.pth"))

    if not model_files:
        logger.error("No model files found in %s", model_dir)
        return []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    logger.info("Found %d models in %s. Loading to %s...", len(model_files), model_dir, device)

    for model_path in model_files:
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if "config" in checkpoint and checkpoint["config"]:
                config = AppConfig.from_dict(checkpoint["config"])
                config.training.device = device.type
            else:
                logger.warning("No config found in %s. Using default AppConfig.", model_path.name)
                config = AppConfig()
                config.training.device = device.type

            model = create_model(config).to(device)
            model.app_config = config

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)

            epoch = checkpoint.get("epoch", "Unknown")
            loss = checkpoint.get("loss", 0.0)
            loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
            logger.info(
                "Loaded model from %s (Epoch %s, Loss %s)",
                model_path.name,
                epoch,
                loss_str,
            )
        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_path, exc)

    return models


def _decode_predictions(models: list[torch.nn.Module], avg_output: torch.Tensor) -> np.ndarray:
    """Decode averaged model output using the active head."""
    head = models[0].head
    if hasattr(head, "predict_k") and callable(head.predict_k):
        return head.predict_k(avg_output).detach().cpu().numpy()
    if hasattr(head, "predict") and callable(head.predict):
        return head.predict(avg_output).detach().cpu().numpy()
    return avg_output.detach().cpu().numpy()


def forward_single(inference_context: dict, table_data: np.ndarray) -> tuple[torch.Tensor, float]:
    """Run ensemble forward pass for a single table and return averaged output tensor
    along with inference time in milliseconds (measures only the model forward
    passes inside the `torch.no_grad()` block).
    """
    rows, cols = table_data.shape
    device: torch.device = inference_context["device"]
    models: list[torch.nn.Module] = inference_context["models"]

    tensor_data = torch.from_numpy(table_data).float().unsqueeze(0).to(device)
    r_mask = torch.zeros(1, rows, dtype=torch.bool, device=device)
    c_mask = torch.zeros(1, cols, dtype=torch.bool, device=device)

    all_outputs = []
    start = time.perf_counter()
    with torch.no_grad():
        for model in models:
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(tensor_data, r_mask, c_mask)
                all_outputs.append(out)
    end = time.perf_counter()

    avg = torch.stack(all_outputs).mean(dim=0)
    inference_time_ms = (end - start) * 1000.0
    return avg, float(inference_time_ms)


def _build_target_tensor(inference_context: dict, target_value: float | int) -> torch.Tensor:
    """Encode scalar target to head-specific tensor representation."""
    models: list[torch.nn.Module] = inference_context["models"]
    device: torch.device = inference_context["device"]
    head_cfg = models[0].app_config.head_config
    encoded_target = head_cfg.target_transform(target_value)
    target_dtype = head_cfg.target_dtype
    mode = getattr(head_cfg, "mode", None)

    if mode in ("classification", "focal"):
        target_values = [int(encoded_target)]
    elif mode == "regression":
        target_values = [float(encoded_target)]
    else:
        target_values = np.expand_dims(np.asarray(encoded_target), axis=0)

    return torch.tensor(target_values, dtype=target_dtype, device=device)


def _prediction_to_scalar(prediction: np.ndarray) -> float | list:
    if prediction.size == 1:
        return float(np.squeeze(prediction))
    pred_val = prediction.tolist()
    if isinstance(pred_val, list) and len(pred_val) == 1:
        return float(pred_val[0])
    return pred_val


def run_single_inference(
    inference_context: dict,
    table_data: np.ndarray,
    target_value: float | int | None = None,
) -> dict:
    """Run single-table inference and return prediction, raw output, and optional metrics."""
    avg_output, inference_time_ms = forward_single(inference_context, table_data)
    models: list[torch.nn.Module] = inference_context["models"]
    decoded_prediction = _decode_predictions(models, avg_output)
    head_metrics = None

    if target_value is not None and np.isfinite(float(target_value)):
        target_tensor = _build_target_tensor(inference_context, target_value)
        with torch.no_grad():
            metrics = models[0].head.compute_metrics(avg_output, target_tensor)
        head_metrics = {
            key: float(val.cpu().item() if hasattr(val, "cpu") else val)
            for key, val in metrics.items()
        }

    raw_output = np.squeeze(avg_output.detach().cpu().numpy()).tolist()

    return {
        "prediction": decoded_prediction,
        "predicted_value": _prediction_to_scalar(decoded_prediction),
        "inference_time_ms": inference_time_ms,
        "raw_output": raw_output,
        "head_metrics": head_metrics,
    }


def predict_single(inference_context: dict, table_data: np.ndarray) -> np.ndarray:
    """Run single-table inference and return only decoded prediction."""
    return run_single_inference(inference_context, table_data)["prediction"][0]
