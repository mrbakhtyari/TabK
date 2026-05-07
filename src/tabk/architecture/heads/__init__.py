"""Head registry and factory.

Register new heads here to make them available via AppConfig.
"""

from .base import BaseHead
from .factory import HEAD_REGISTRY, get_head
from .k_estimator import KEstimator

__all__ = [
    "BaseHead",
    "KEstimator",
    "HEAD_REGISTRY",
    "get_head",
]
