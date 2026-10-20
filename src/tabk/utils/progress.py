import sys
from typing import TextIO


def is_interactive_stream(stream: TextIO | None = None) -> bool:
    """Return True when the target stream is an interactive terminal."""
    target = sys.stderr if stream is None else stream
    isatty = getattr(target, "isatty", None)
    return bool(isatty and isatty())


def write_progress_line(message: str, stream: TextIO | None = None) -> None:
    """Emit a progress update that remains visible in terminals and redirected logs."""
    target = sys.stderr if stream is None else stream
    print(message, file=target, flush=True)
