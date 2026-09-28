import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HeadConfig


class KHead(nn.Module):
    """DLDL head: maps the table embedding to a log-distribution over k in [min_k, max_k]."""

    def __init__(self, d_model: int, config: HeadConfig):
        super().__init__()
        self.min_k = config.min_k
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, config.num_classes),
        )

    def forward(self, table_vector: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.fc(table_vector), dim=-1)

    def loss_fn(self) -> nn.Module:
        return nn.KLDivLoss(reduction="batchmean")

    def predict_k(self, log_probs: torch.Tensor) -> torch.Tensor:
        return log_probs.argmax(dim=-1) + self.min_k

    def compute_metrics(self, log_probs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Batch-mean exact-match accuracy, MAE, MSE and ±1 accuracy of the predicted k."""
        pred_k = self.predict_k(log_probs).float()
        true_k = targets.argmax(dim=-1).float() + self.min_k
        abs_err = (pred_k - true_k).abs()
        return {
            "acc": float((abs_err == 0).float().mean()),
            "mae": float(abs_err.mean()),
            "mse": float((abs_err**2).mean()),
            "off1": float((abs_err <= 1).float().mean()),
        }
