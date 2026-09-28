import json
import logging
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from ..utils import apply_standard_scaling
from .config import AppConfig
from .model import DoubleInvariantTransformer
from .utils import create_model

logger = logging.getLogger(__name__)

PRETRAINED_REPO_ID = "mrbakhtyari/TabK"
SUBSAMPLE_SEED = 0


class TabK:
    """Ensemble of trained TabK models that estimates the number of clusters in a table."""

    def __init__(
        self,
        models: list[DoubleInvariantTransformer],
        config: AppConfig,
        device: torch.device,
    ):
        self.models = models
        self.config = config
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path = PRETRAINED_REPO_ID,
        device: str | torch.device | None = None,
        revision: str | None = None,
    ) -> "TabK":
        """Load from a Hub repo id, an exported folder, or a training output directory."""
        device = _resolve_device(device)
        model_dir = Path(model)
        if not model_dir.is_dir():
            model_dir = Path(snapshot_download(repo_id=str(model), revision=revision))

        if (model_dir / "config.json").is_file():
            config, state_dicts = _read_exported(model_dir)
        else:
            config, state_dicts = _read_checkpoints(model_dir)

        models = []
        for state_dict in state_dicts:
            net = create_model(config)
            net.load_state_dict(state_dict)
            models.append(net.to(device).eval())

        logger.info("Loaded %d models from %s to %s", len(models), model_dir, device)
        return cls(models, config, device)

    @property
    def k_values(self) -> np.ndarray:
        """The values of k that the model can predict, aligned with predict_proba."""
        head = self.config.head_config
        return np.arange(head.min_k, head.max_k + 1)

    def predict(self, X: np.ndarray, scale: bool = True) -> int:
        """Estimate the number of clusters in X (rows are samples, columns are features)."""
        log_probs = self._ensemble_log_probs(X, scale)
        return int(self.models[0].head.predict_k(log_probs).item())

    def predict_proba(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """Probability of each value in k_values."""
        probs = self._ensemble_log_probs(X, scale).exp().squeeze(0).cpu().numpy()
        return probs / probs.sum()

    def _ensemble_log_probs(self, X: np.ndarray, scale: bool) -> torch.Tensor:
        X = _validate_table(X)
        if scale:
            X = apply_standard_scaling(X)
        X = _limit_rows(X, self.config.max_rows)
        rows, cols = X.shape
        if cols > self.config.max_cols:
            logger.warning(
                "Table has %d columns; the model was trained on at most %d",
                cols,
                self.config.max_cols,
            )

        x = torch.from_numpy(X).float().unsqueeze(0).to(self.device)
        row_mask = torch.zeros(1, rows, dtype=torch.bool, device=self.device)
        col_mask = torch.zeros(1, cols, dtype=torch.bool, device=self.device)

        outputs = []
        with torch.no_grad():
            for model in self.models:
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    outputs.append(model(x, row_mask, col_mask))
        return torch.stack(outputs).mean(dim=0)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _read_exported(model_dir: Path) -> tuple[AppConfig, list[dict]]:
    """Read an exported ensemble: one shared config.json and one *.safetensors file per fold."""
    with open(model_dir / "config.json", encoding="utf-8") as f:
        config = AppConfig.from_dict(json.load(f))

    model_files = sorted(model_dir.glob("*.safetensors"))
    if not model_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    return config, [load_file(path) for path in model_files]


def _read_checkpoints(model_dir: Path) -> tuple[AppConfig, list[dict]]:
    """Read a training output directory: one checkpoints/*.pth file per fold."""
    model_files = sorted((model_dir / "checkpoints").glob("*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No .pth checkpoints found in {model_dir / 'checkpoints'}")

    config_dicts, state_dicts = [], []
    for path in model_files:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not checkpoint.get("config"):
            raise ValueError(f"{path} has no stored config")
        config_dicts.append(checkpoint["config"])
        state_dicts.append(checkpoint["model_state_dict"])

    configs = [AppConfig.from_dict(d) for d in config_dicts]
    if any(c.model != configs[0].model or c.head_config != configs[0].head_config for c in configs):
        raise ValueError(f"Checkpoints in {model_dir} were trained with different configs")
    return configs[0], state_dicts


def _validate_table(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D table, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("Table must be numeric and contain no missing or infinite values")
    return X


def _limit_rows(X: np.ndarray, max_rows: int) -> np.ndarray:
    """Uniformly subsample rows of tables larger than the training range."""
    rows = X.shape[0]
    if rows <= max_rows:
        return X
    logger.warning("Table has %d rows; subsampling %d rows uniformly at random", rows, max_rows)
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    return X[np.sort(rng.choice(rows, size=max_rows, replace=False))]
