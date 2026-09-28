import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch


@dataclass
class HeadConfig:
    min_k: int = 2
    max_k: int = 15
    sigma: float = 0.5

    @property
    def num_classes(self) -> int:
        return self.max_k - self.min_k + 1

    def target_transform(self, k: int | np.ndarray) -> np.ndarray:
        """Encode k as a discretized Gaussian over the supported range (DLDL target)."""
        k = min(max(int(np.asarray(k).item()), self.min_k), self.max_k)
        classes = np.arange(self.num_classes, dtype=np.float32)
        target = np.exp(-((classes - (k - self.min_k)) ** 2) / (2 * self.sigma**2))
        return target / target.sum()


@dataclass
class ModelConfig:
    d_model: int = 64
    n_head: int = 4
    n_layers: int = 4
    dropout: float = 0.3
    num_bins: int = 50


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
    head_config: HeadConfig = field(default_factory=HeadConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Largest table seen during training; larger inputs are row-subsampled at inference
    max_rows: int = 2500
    max_cols: int = 200

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        """Create AppConfig from a dictionary, ignoring keys from older config versions."""
        return cls(
            head_config=_from_known_fields(HeadConfig, data.get("head_config", {})),
            model=_from_known_fields(ModelConfig, data.get("model", {})),
            training=_from_known_fields(TrainingConfig, data.get("training", {})),
            **{k: data[k] for k in ("max_rows", "max_cols") if k in data},
        )

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        """Load configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


def _from_known_fields[T](cls: type[T], data: dict) -> T:
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})
