import logging

from .preprocessing import apply_standard_scaling


def configure_logging(verbosity: int = 0) -> None:
    """Configure logging level and format."""
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("tabk").setLevel(level)


__all__ = [
    "configure_logging",
    "apply_standard_scaling",
]
