import numpy as np
from sklearn.preprocessing import StandardScaler


def apply_standard_scaling(X: np.ndarray) -> np.ndarray:
    """Standardize each feature of a table to zero mean and unit variance (float32)."""
    if not isinstance(X, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(X)}")

    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    return X_scaled
