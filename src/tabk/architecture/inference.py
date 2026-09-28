import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from .config import AppConfig
from .utils import create_model

logger = logging.getLogger(__name__)

PRETRAINED_REPO_ID = "mrbakhtyari/TabK"


def load_inference_context(
    model: str | Path = PRETRAINED_REPO_ID,
    device: str | torch.device | None = None,
    revision: str | None = None,
) -> dict:
    """Load ensemble models and return an inference artifact bundle.

    Args:
        model: A local directory (training output with ``checkpoints/*.pth``, or an
            exported folder with ``config.json`` and ``*.safetensors``) or a
            Hugging Face Hub repo id. Defaults to the pretrained TabK ensemble.
        device: Target device. Defaults to CUDA when available.
        revision: Hub revision (branch, tag or commit) when ``model`` is a repo id.
    """
    model_dir = Path(model)
    if not model_dir.is_dir():
        model_dir = Path(snapshot_download(repo_id=str(model), revision=revision))

    if (model_dir / "config.json").is_file():
        models = _load_safetensors_models(model_dir, device)
    else:
        models = _load_models(model_dir, device)
    if not models:
        raise ValueError(f"No models loaded from {model}")

    config = models[0].app_config
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
    model_files = sorted((Path(model_dir) / "checkpoints").glob("*.pth"))
    if not model_files:
        logger.error("No model files found in %s", model_dir)
        return []

    device = _resolve_device(device)
    for model_path in model_files:
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            config = AppConfig.from_dict(checkpoint.get("config") or {})
            config.training.device = device.type
            model = create_model(config)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device).eval()
            model.app_config = config
            models.append(model)
        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_path, exc)

    return models


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_safetensors_models(
    model_dir: Path,
    device: str | torch.device | None = None,
) -> list[torch.nn.Module]:
    """Load an exported ensemble: one shared ``config.json`` and one ``*.safetensors`` per fold."""
    device = _resolve_device(device)
    with open(model_dir / "config.json", encoding="utf-8") as f:
        config_dict = json.load(f)

    model_files = sorted(model_dir.glob("*.safetensors"))
    if not model_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    models: list[torch.nn.Module] = []
    for model_path in model_files:
        config = AppConfig.from_dict(config_dict)
        config.training.device = device.type
        model = create_model(config)
        model.load_state_dict(load_file(model_path))
        model.to(device).eval()
        model.app_config = config
        models.append(model)

    logger.info("Loaded %d models from %s to %s", len(models), model_dir, device)
    return models


def _decode_predictions(models: list[torch.nn.Module], avg_output: torch.Tensor) -> np.ndarray:
    return models[0].head.predict_k(avg_output).cpu().numpy()


def forward_single(inference_context: dict, table_data: np.ndarray) -> tuple[torch.Tensor, float]:
    """Run ensemble forward pass for a single table and return averaged output tensor
    along with inference time in milliseconds (measures only the model forward
    passes inside the `torch.no_grad()` block).
    """
    device: torch.device = inference_context["device"]
    models: list[torch.nn.Module] = inference_context["models"]

    rows, cols = table_data.shape

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
    config: AppConfig = inference_context["config"]
    target = config.head_config.target_transform(int(target_value))
    return torch.tensor(target[None], dtype=torch.float32, device=inference_context["device"])


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
        head_metrics = models[0].head.compute_metrics(avg_output, target_tensor)

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
