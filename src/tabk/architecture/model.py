import torch
import torch.nn as nn

from .config import ModelConfig


class QuantileFeatureEncoder(nn.Module):
    """
    (QFE) Generates column embeddings based on their statistical distribution.
    Ensures Column Permutation Invariance.
    """

    def __init__(self, d_model, num_bins=10):
        super().__init__()

        self.num_bins = num_bins

        # Encodes the statistical fingerprint (quantiles) into a vector

        self.stat_encoder = nn.Sequential(
            nn.Linear(num_bins, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x, row_mask):
        # x: (Batch, Rows, Cols)

        B, R, C = x.shape

        # 1. Filter Padding for Statistics
        x_clean = x.clone()

        if row_mask is not None:
            # Push padded rows to infinity so they sort to the end
            mask_expanded = row_mask.unsqueeze(-1).expand(-1, -1, C)
            x_clean[mask_expanded] = float("inf")

            # Count valid rows per batch item: (B,)
            valid_counts = (~row_mask).sum(dim=1).float()
        else:
            valid_counts = torch.full((B,), R, device=x.device, dtype=torch.float)

        # 2. Sort to find distribution shape (Row-independent)
        x_sorted, _ = torch.sort(x_clean, dim=1)

        # 3. Sample Quantiles from VALID range only
        # For each batch item, sample num_bins indices from [0, valid_count-1]
        # We use a batch-aware approach: compute fractional indices, then gather

        # Fractional positions: (num_bins,) in [0, 1]
        fracs = torch.linspace(0, 1, self.num_bins, device=x.device)  # (num_bins,)

        # Scale to valid range per batch: (B, num_bins)
        # indices[b, i] = fracs[i] * (valid_counts[b] - 1)
        max_idx = valid_counts - 1
        indices_float = fracs.unsqueeze(0) * max_idx.unsqueeze(1)  # (B, num_bins)

        # Round (not truncate) then clamp to [0, R-1]
        indices = indices_float.round().long().clamp(min=0, max=R - 1)  # (B, num_bins)

        # Gather quantiles: need (B, num_bins, C)
        # x_sorted is (B, R, C), indices is (B, num_bins)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, C)  # (B, num_bins, C)
        quantiles = torch.gather(x_sorted, dim=1, index=indices_expanded)  # (B, num_bins, C)

        # Permute to (Batch, Cols, Num_Bins) for the linear layer
        quantiles = quantiles.permute(0, 2, 1)

        # 4. Generate ID Vectors
        return self.stat_encoder(quantiles)  # (Batch, Cols, D_Model)


class PMAPooling(nn.Module):
    """
    Pooling by Multihead Attention (PMA) with k=1.

    A Permutation-Invariant Pooling mechanism from the Set Transformer paper.
    Uses a learned 'Seed Vector' to query the set and extract a summary.
    Better than MeanPooling for detecting outliers or specific clusters.
    """

    def __init__(self, d_model, num_heads=4):
        super().__init__()

        self.seed_vector = nn.Parameter(torch.randn(1, 1, d_model))

        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        # x: (Batch, Seq_Len, D_Model)

        B = x.shape[0]

        # Broadcast seed to batch size

        query = self.seed_vector.repeat(B, 1, 1)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.clone()

        attended, _ = self.mha(query, x, x, key_padding_mask=key_padding_mask)

        return self.norm(attended.squeeze(1))  # (Batch, D_Model)


class ColumnSetEmbedding(nn.Module):
    """
    Inner Loop: Converts a Set of Columns into a Single Row Vector.

    Achieves Column Invariance by:
    1. Using QFE to create position-independent column identities
    2. Running self-attention on QFE embeddings to disambiguate columns
       with similar distributions (O(B·C²) instead of O(B·R·C²))
    3. Combining refined QFE with value embeddings
    4. Using permutation-invariant addition aggregation
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        d_model = config.d_model
        num_bins = config.num_bins
        n_head = config.n_head
        dropout = config.dropout

        # Column identity encoder (based on distribution statistics)
        self.qfe = QuantileFeatureEncoder(d_model, num_bins)

        # Value encoder (per-cell)
        self.val_encoder = nn.Linear(1, d_model)

        # Column interaction transformer — runs on QFE embeddings (B, C, D)
        # to disambiguate columns via inter-column distributional context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.col_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=1, enable_nested_tensor=False
        )

        # Output projection
        self.projection = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())

    def forward(self, x, row_mask, col_mask):
        B, R, C = x.shape

        # 1. Get Column Identities (Invariant to Col Order) -> (B, C, D)
        qfe_ids = self.qfe(x, row_mask)

        # 2. Disambiguate columns via self-attention on QFE embeddings
        #    This is O(B·C²) instead of O(B·R·C²) — the key optimization
        refined_qfe = self.col_transformer(qfe_ids, src_key_padding_mask=col_mask)  # (B, C, D)

        # 3. Embed Values -> (B, R, C, D)
        val_emb = self.val_encoder(x.unsqueeze(-1))

        # 4. Addition aggregation: refined QFE + value embeddings, then mean
        qfe_expanded = refined_qfe.unsqueeze(1).expand(-1, R, -1, -1)
        combined = val_emb + qfe_expanded  # (B, R, C, D)

        if col_mask is not None:
            mask = (~col_mask).float().unsqueeze(1).unsqueeze(-1)  # (B, 1, C, 1)
            combined = combined * mask
            counts = mask.sum(dim=2).clamp(min=1)
        else:
            counts = C

        row_vectors = combined.sum(dim=2) / counts  # (B, R, D)

        return self.projection(row_vectors)


class DoubleInvariantTransformer(nn.Module):
    """
    The Main Model.
    Structure: [Column Set -> Row Vector] -> [Row Set -> Table Vector] -> Class
    """

    def __init__(
        self,
        config: ModelConfig,
        head: nn.Module | None = None,
    ):
        d_model = config.d_model
        n_head = config.n_head
        n_layers = config.n_layers
        dropout = config.dropout
        super().__init__()

        # 1. Inner Processor (Columns)

        self.col_processor = ColumnSetEmbedding(config)

        # 2. Outer Processor (Rows)

        # Standard Transformer to find relationships between rows

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.row_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # 3. Global Pooling (Rows -> Table)

        self.row_pooler = PMAPooling(d_model, num_heads=n_head)

        # 4. Task Head (pluggable)

        self.head = head

    def forward(self, x, row_mask=None, col_mask=None):
        # x: (Batch, Rows, Cols)

        B, R, C = x.shape

        device = x.device

        # Auto-Masking for Inference (BS=1)

        if row_mask is None:
            row_mask = torch.zeros((B, R), dtype=torch.bool, device=device)

        if col_mask is None:
            col_mask = torch.zeros((B, C), dtype=torch.bool, device=device)

        # --- Phase 1: Column Invariance ---

        # "Bag of Columns" -> Row Vector

        row_vectors = self.col_processor(x, row_mask, col_mask)  # (B, R, D)

        # --- Phase 2: Row Interaction ---

        # "Sequence of Rows" (but permutation invariant due to pooling later)

        encoded_rows = self.row_transformer(row_vectors, src_key_padding_mask=row_mask)

        # --- Phase 3: Row Invariance ---

        # "Bag of Rows" -> Table Vector

        table_vector = self.row_pooler(encoded_rows, key_padding_mask=row_mask)  # (B, D)

        if self.head is not None:
            return self.head(table_vector)
        return table_vector  # backbone-only: return embeddings
