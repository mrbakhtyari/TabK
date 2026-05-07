"""Base head protocol.

All task heads must subclass BaseHead and implement the required methods.
This ensures the backbone, dataset, and training loop are fully decoupled
from task-specific logic.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


class BaseHead(ABC, nn.Module):
    """Abstract base for all prediction heads.

    A head owns everything task-specific:
    - forward():             (B, D) backbone output → task prediction
    - get_loss_fn():         loss function appropriate for this task
    - target_transform():    raw target value → preprocessed target
    """

    @abstractmethod
    def forward(self, table_vector: torch.Tensor) -> torch.Tensor:
        """Map backbone output to task-specific prediction.

        Args:
            table_vector: (Batch, D_Model) from backbone pooling.

        Returns:
            Task-specific output tensor.
        """
        ...

    @abstractmethod
    def get_loss_fn(self) -> nn.Module:
        """Return the loss function appropriate for this head."""
        ...

    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Compute head-specific metrics for a batch.

        Returns:
            Dictionary of metric_name -> scalar float (batch-level).
            Names should be unprefixed (e.g., "acc", "mae").
        """
        return {}

    def target_transform(self, raw_target: np.ndarray) -> np.ndarray:
        """Preprocess the raw target array loaded from an NPZ file.

        Override this per task to apply task-specific preprocessing
        (e.g., clipping, argmax, dtype conversion).

        Default: cast to float32.
        """
        return raw_target.astype(np.float32)

    @property
    def target_dtype(self) -> torch.dtype:
        """Tensor dtype for the target.

        Override per task: torch.long for classification, torch.float32 for regression.
        """
        return torch.float32
