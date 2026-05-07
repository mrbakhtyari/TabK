import numpy as np
from sklearn.preprocessing import StandardScaler


def coerce_predicted_k_to_int(value: object) -> int | None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(numeric_value):
        return None

    return int(round(numeric_value))


def apply_standard_scaling(X: np.ndarray) -> np.ndarray:
    """
    Apply StandardScaler to dataset (per-sample normalization).

    This is the canonical preprocessing for ALL tabk workflows.
    Each dataset is scaled independently to have zero mean and unit variance
    per feature.

    Args:
        X: Input data array of shape (n_samples, n_features)

    Returns:
        Scaled data array with same shape, dtype float32

    Example:
        >>> X = np.random.randn(100, 10)
        >>> X_scaled = apply_standard_scaling(X)
        >>> assert X_scaled.shape == X.shape
        >>> assert X_scaled.dtype == np.float32
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(X)}")

    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    return X_scaled
