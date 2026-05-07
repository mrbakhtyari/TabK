"""KEstimator head for Cluster Number Prediction.

Supports six modes:
- "distribution": DLDL — soft Gaussian targets + KL-divergence (ordinal-aware)
- "classification": Standard CrossEntropy (baseline)
- "focal": Classification with Focal Loss (imbalance-aware)
- "ordinal": CORAL ordinal regression (ordinal-aware, threshold-based)
- "regression": Direct scalar regression with Huber loss (simplest baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AppConfig
from .base import BaseHead

# ---- Valid modes (single source of truth) ----
VALID_MODES = ("distribution", "classification", "focal", "ordinal", "regression")


# ---- Custom loss modules ----


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification (Lin et al., 2017).

    Downweights well-classified examples so the model focuses on hard,
    misclassified ones.  Particularly effective when classes are imbalanced
    — common in k-estimation where mid-range k values dominate.

    Loss for sample i, true class y:
        FL(p_y) = -(1 - p_y)^gamma * log(p_y)

    Args:
        gamma: focusing parameter (0 = standard CE, 2 = recommended default).
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard log-softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)  # (B, C)
        probs = log_probs.exp()

        # Gather the probability of the true class
        targets_1d = targets.view(-1)  # (B,)
        log_p_y = log_probs.gather(1, targets_1d.unsqueeze(1)).squeeze(1)  # (B,)
        p_y = probs.gather(1, targets_1d.unsqueeze(1)).squeeze(1)  # (B,)

        # Focal modulation: (1 - p_y)^gamma
        focal_weight = (1.0 - p_y) ** self.gamma

        loss = -focal_weight * log_p_y
        return loss.mean()


class KEstimator(BaseHead):
    """Predicts number of clusters (k).

    Modes:
        distribution   - Soft Gaussian targets + KL-divergence (ordinal-aware).
        classification - Standard CrossEntropy baseline.
        focal          - Focal Loss for imbalanced classes.
        ordinal        - CORAL: cumulative threshold probabilities + BCE.
        regression     - Scalar prediction + Huber loss (simplest baseline).
    """

    def __init__(self, config: AppConfig):
        super().__init__()

        self.d_model = config.model.d_model
        if hasattr(config.head_config, "min_k"):
            self.head_config = config.head_config
        else:
            raise TypeError("KEstimator requires a KEstimatorConfig")

        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")

        self.min_k = self.head_config.min_k
        self.max_k = self.head_config.max_k
        self.num_classes = self.head_config.num_classes
        self.mode = self.head_config.mode

        if self.mode not in VALID_MODES:
            raise ValueError(f"Unknown mode '{self.mode}'. Valid: {', '.join(VALID_MODES)}")

        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")

        # ---- Build MLP head ----
        # Output dimension depends on mode:
        #   ordinal    → num_classes - 1  (one logit per cumulative threshold)
        #   regression → 1               (scalar k)
        #   others     → num_classes      (one logit per class)
        if self.mode == "ordinal":
            out_dim = self.num_classes - 1
        elif self.mode == "regression":
            out_dim = 1
        else:
            out_dim = self.num_classes

        self.fc = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.d_model, out_dim),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, table_vector: torch.Tensor) -> torch.Tensor:
        logits = self.fc(table_vector)

        if self.mode == "distribution":
            return F.log_softmax(logits, dim=-1)

        if self.mode == "ordinal":
            # Return raw logits — BCEWithLogitsLoss handles sigmoid internally
            return logits

        if self.mode == "regression":
            return logits.squeeze(-1)  # (B,)

        # classification / focal / raw logits
        return logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def get_loss_fn(self) -> nn.Module:
        if self.mode == "distribution":
            return nn.KLDivLoss(reduction="batchmean")

        if self.mode == "focal":
            return FocalLoss(gamma=2.0)

        if self.mode == "classification":
            return nn.CrossEntropyLoss()

        if self.mode == "ordinal":
            # BCEWithLogitsLoss = sigmoid + BCE fused for numerical stability
            return nn.BCEWithLogitsLoss()

        if self.mode == "regression":
            return nn.HuberLoss(delta=1.0)

        raise ValueError(f"No loss for mode '{self.mode}'")

    # ------------------------------------------------------------------
    # Prediction (output → integer k)
    # ------------------------------------------------------------------

    def predict_k(self, output: torch.Tensor) -> torch.Tensor:
        """Convert model output to predicted k values.

        distribution   → expected value of predicted distribution
        classification → argmax
        focal          → argmax
        ordinal        → count cumulative probs > 0.5, add min_k
        regression     → round and clamp
        """
        if self.mode == "distribution":
            # Use argmax (mode of distribution) for exact-match accuracy
            return output.argmax(dim=-1) + self.min_k

        if self.mode in ("classification", "focal"):
            return output.argmax(dim=-1) + self.min_k

        if self.mode == "ordinal":
            # Apply sigmoid here (forward returns raw logits for loss stability)
            probs = torch.sigmoid(output)
            class_idx = (probs > 0.5).sum(dim=-1)
            return class_idx + self.min_k

        if self.mode == "regression":
            return output.round().clamp(self.min_k, self.max_k)

        raise ValueError(f"No predict_k for mode '{self.mode}'")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Compute k-estimator metrics for a batch.

        Returns:
            acc: exact-match accuracy on rounded predicted k
            mae: mean absolute error in predicted k
            mse: mean squared error (aggregatable; RMSE derived in validate)
            off1: fraction of predictions within ±1 of true k
        """
        pred_k = self.predict_k(outputs)

        # Recover true k from whatever target encoding this mode uses
        if self.mode == "distribution":
            true_class_idx = targets.argmax(dim=-1)
        elif self.mode in ("classification", "focal"):
            true_class_idx = targets
        elif self.mode == "ordinal":
            # Cumulative target [1,1,..,1,0,..,0] → count of 1s = class index
            true_class_idx = targets.sum(dim=-1)
        elif self.mode == "regression":
            # Target is raw k → class_idx = k - min_k (but we need true_k directly)
            true_k = targets.float()
            abs_err = (pred_k - true_k).abs()
            acc = (pred_k.round() == true_k.round()).float().mean()
            mae = abs_err.mean()
            mse = (abs_err**2).mean()
            off1 = (abs_err <= 1.0).float().mean()
            return {
                "acc": float(acc.item()),
                "mae": float(mae.item()),
                "mse": float(mse.item()),
                "off1": float(off1.item()),
            }
        else:
            raise ValueError(f"No metrics for mode '{self.mode}'")

        true_k = true_class_idx.float() + self.min_k
        abs_err = (pred_k - true_k).abs()

        acc = (pred_k.round() == true_k.round()).float().mean()
        mae = abs_err.mean()
        mse = (abs_err**2).mean()
        off1 = (abs_err <= 1.0).float().mean()

        return {
            "acc": float(acc.item()),
            "mae": float(mae.item()),
            "mse": float(mse.item()),
            "off1": float(off1.item()),
        }
