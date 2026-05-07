import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class BaseHeadConfig:
    """Abstract base configuration for task heads."""

    head_type: str = field(init=False)

    def target_transform(self, raw_target: np.ndarray) -> np.ndarray:
        """Default: cast to float32."""
        return raw_target.astype(np.float32)

    @property
    def target_dtype(self) -> torch.dtype:
        return torch.float32


@dataclass
class KEstimatorConfig(BaseHeadConfig):
    head_type: str = field(default="k_estimator", init=False)
    min_k: int = 2
    max_k: int = 15
    mode: str = (
        "distribution"  # "distribution" | "classification" | "focal" | "ordinal" | "regression"
    )
    sigma: float = 0.5
    class_weights: list[float] | None = None

    @property
    def needs_class_weights(self) -> bool:
        """Whether this mode uses weighted loss (class weights should be computed)."""
        return False

    @property
    def num_classes(self) -> int:
        return self.max_k - self.min_k + 1

    def target_transform(self, raw_target: np.ndarray) -> np.ndarray:
        """Transform raw k value to target.

        - distribution: soft Gaussian distribution over classes (float32)
        - classification / focal: 0-indexed class label (int64)
        - ordinal: cumulative binary vector [1,..,1,0,..,0] (float32)
        - regression: raw k value as float scalar (float32)
        """
        if isinstance(raw_target, np.ndarray):
            k = int(raw_target.item())
        else:
            k = int(raw_target)

        k = max(self.min_k, min(k, self.max_k))
        class_idx = k - self.min_k

        if self.mode == "distribution":
            class_indices = np.arange(self.num_classes, dtype=np.float32)
            variance = 2 * (self.sigma**2)
            squared_diff = (class_indices - class_idx) ** 2
            target_dist = np.exp(-squared_diff / variance)
            target_dist /= target_dist.sum()
            return target_dist

        if self.mode in ("classification", "focal"):
            return np.array(class_idx, dtype=np.int64)

        if self.mode == "ordinal":
            # Cumulative encoding: [1, 1, ..., 1, 0, ..., 0]
            # Length = num_classes - 1  (thresholds between adjacent classes)
            # class_idx=0 → all zeros, class_idx=max → all ones
            target = np.zeros(self.num_classes - 1, dtype=np.float32)
            target[:class_idx] = 1.0
            return target

        if self.mode == "regression":
            return np.array(float(k), dtype=np.float32)

        raise ValueError(f"Unknown mode '{self.mode}'")

    @property
    def target_dtype(self) -> torch.dtype:
        if self.mode in ("classification", "focal"):
            return torch.long
        # distribution, ordinal, regression all use float32
        return torch.float32


@dataclass
class ModelConfig:
    d_model: int = 64
    n_head: int = 4
    n_layers: int = 4
    dropout: float = 0.3
    num_bins: int = 50
    model_type: str = "default"
    # Linear-attention backend settings (used by model_type="dit_fla_v1").
    fla_backend: str = "flash_linear_attention"  # auto | flash_linear_attention | torch_linear
    fla_allow_fallback: bool = True
    fla_eps: float = 1e-6


@dataclass
class TrainingConfig:
    batch_size: int = 16
    accum_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    epochs: int = 20
    k_folds: int = 5
    patience: int = 5
    num_workers: int = 8
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


@dataclass
class AppConfig:
    head_config: BaseHeadConfig = field(default_factory=KEstimatorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Dataset Constraints (Global / Fixed for now)
    max_rows: int = 2500
    max_cols: int = 200

    def to_dict(self) -> dict:
        d = asdict(self)
        # Ensure head_config.head_type is included if it was excluded by init=False but present
        # asdict usually includes properties if they are fields.
        return d

    def save(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        """Create AppConfig from dictionary."""
        config = cls()

        if "head_config" in data:
            hc_data = data["head_config"]
            h_type = hc_data.get("head_type", "k_estimator")

            # Filter out fields with init=False (e.g. head_type) that
            # are present in serialized dicts but can't be passed to __init__
            if h_type == "k_estimator":
                filtered = {k: v for k, v in hc_data.items() if k != "head_type"}
                config.head_config = KEstimatorConfig(**filtered)
            # Add other types here

        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])

        # Global fields
        for k in ["max_rows", "max_cols"]:
            if k in data:
                setattr(config, k, data[k])

        return config

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        """Load configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
