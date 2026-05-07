import numpy as np

METHOD_NAME = "gmm_bic"
DIRECTION = "min"


def score_from_cached_bic(
    bic_by_k: dict[int, float | None],
    k_values: list[int],
) -> tuple[dict[int, float], dict[int, float]]:
    """Use cached per-k BIC scores stored in the base H5 file."""
    score_by_k: dict[int, float] = {}
    elapsed_ms_by_k: dict[int, float] = {}

    for k in k_values:
        bic_val = bic_by_k.get(k)
        score_by_k[k] = (
            float(bic_val) if bic_val is not None and np.isfinite(float(bic_val)) else np.nan
        )
        # Reading cached values is near-zero overhead; keep explicit timing column.
        elapsed_ms_by_k[k] = 0.0

    return score_by_k, elapsed_ms_by_k
