import torch
import torch.nn as nn

from .config import ModelConfig
from .model import (
    ColumnSetEmbedding,
    DoubleInvariantTransformer,
)

# ===============================================
# Ablation 1: Without QFE
# ===============================================


class SimpleStatsIdentifier(nn.Module):
    """
    Ablation replacement for QFE (Quantile Feature Encoder).
    Uses simple Mean and Standard Deviation instead of quantiles.
    """

    def __init__(self, d_model):
        super().__init__()
        # We have 2 features: mean and std instead of num_bins features
        self.stat_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x, row_mask):
        # x: (Batch, Rows, Cols)
        B, R, C = x.shape
        x_clean = x.clone()

        if row_mask is not None:
            # Mask out invalid values
            mask_expanded = row_mask.unsqueeze(-1).expand(-1, -1, C)
            x_clean[mask_expanded] = 0.0

            valid_counts = (~row_mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)
            valid_counts_expanded = valid_counts.unsqueeze(-1)

            # Compute valid mean
            means = x_clean.sum(dim=1) / valid_counts_expanded.squeeze(1)

            # Compute valid std
            x_zero_mean = (x_clean - means.unsqueeze(1)) * (~mask_expanded).float()
            variances = (x_zero_mean**2).sum(dim=1) / valid_counts_expanded.squeeze(1)
            stds = torch.sqrt(variances + 1e-8)
        else:
            means = x.mean(dim=1)
            stds = x.std(dim=1, unbiased=False)

        # features: (Batch, Cols, 2)
        stats = torch.stack([means, stds], dim=-1)

        # Generate ID vectors
        return self.stat_encoder(stats)


class SimpleColumnEmbeddingIdentifier(nn.Module):
    """Simple learnable embedding per column index (position-based identity)."""

    def __init__(self, d_model: int, max_cols: int = 512):
        super().__init__()
        self.max_cols = max_cols
        self.col_embed = nn.Embedding(max_cols, d_model)
        self.post = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x, row_mask):
        B, _, C = x.shape
        if C > self.max_cols:
            raise ValueError(
                f"Input has {C} columns, but max_cols={self.max_cols} for "
                "SimpleColumnEmbeddingIdentifier"
            )

        col_idx = torch.arange(C, device=x.device)
        col_ids = self.col_embed(col_idx).unsqueeze(0).expand(B, -1, -1)
        return self.post(col_ids)


class ColumnSetEmbeddingWithoutQFE(ColumnSetEmbedding):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.qfe = SimpleStatsIdentifier(config.d_model)


class ModelWithoutQFE(DoubleInvariantTransformer):
    def __init__(self, config: ModelConfig, head: nn.Module | None = None):
        super().__init__(config, head)
        # Override the column processor
        self.col_processor = ColumnSetEmbeddingWithoutQFE(config)


# ===============================================
# Ablation 2: Without PMA
# ===============================================


class MeanPooling(nn.Module):
    """
    Ablation replacement for PMAPooling.
    Uses standard mean pooling across rows instead of multi-head attention.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, key_padding_mask=None):
        # x: (Batch, Seq_Len, D_Model)
        if key_padding_mask is not None:
            # key_padding_mask is True for padded positions
            weights = (~key_padding_mask).float().unsqueeze(-1)
            # Sum over valid sequences, divide by number of valid sequences
            sum_x = (x * weights).sum(dim=1)
            count = weights.sum(dim=1).clamp(min=1.0)
            return sum_x / count
        else:
            return x.mean(dim=1)


class ModelWithoutPMA(DoubleInvariantTransformer):
    def __init__(self, config: ModelConfig, head: nn.Module | None = None):
        super().__init__(config, head)
        # Override the row pooler with simple mean pooling
        self.row_pooler = MeanPooling()


# ===============================================
# Ablation 3: Without Column Interaction
# ===============================================


class ColumnSetEmbeddingWithoutInteraction(ColumnSetEmbedding):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Remove the column transformer completely
        self.col_transformer = None

    def forward(self, x, row_mask, col_mask):
        B, R, C = x.shape

        # 1. Get Column Identities
        qfe_ids = self.qfe(x, row_mask)

        # 2. SKIP column disambiguation! Just use raw QFE IDs.
        refined_qfe = qfe_ids

        # 3. Embed Values
        val_emb = self.val_encoder(x.unsqueeze(-1))

        # 4. Addition aggregation
        qfe_expanded = refined_qfe.unsqueeze(1).expand(-1, R, -1, -1)
        combined = val_emb + qfe_expanded

        if col_mask is not None:
            mask = (~col_mask).float().unsqueeze(1).unsqueeze(-1)
            combined = combined * mask
            counts = mask.sum(dim=2).clamp(min=1)
        else:
            counts = C

        row_vectors = combined.sum(dim=2) / counts
        return self.projection(row_vectors)


class ModelWithoutColumnInteraction(DoubleInvariantTransformer):
    def __init__(self, config: ModelConfig, head: nn.Module | None = None):
        super().__init__(config, head)
        # Override the column processor
        self.col_processor = ColumnSetEmbeddingWithoutInteraction(config)
