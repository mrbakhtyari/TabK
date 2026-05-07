"""Head factory and registry."""

from ..config import AppConfig
from .base import BaseHead
from .k_estimator import KEstimator

HEAD_REGISTRY: dict[str, type[BaseHead]] = {
    "k_estimator": KEstimator,
}


def get_head(config: AppConfig) -> BaseHead:
    """Create a head instance from the configuration.

    Args:
        config: Application configuration object.

    Returns:
        Instantiated head module.

    Raises:
        ValueError: If head_type is not registered.
    """
    head_type = config.head_config.head_type

    if head_type not in HEAD_REGISTRY:
        available = ", ".join(sorted(HEAD_REGISTRY))
        raise ValueError(f"Unknown head type '{head_type}'. Available: {available}")

    return HEAD_REGISTRY[head_type](config)
