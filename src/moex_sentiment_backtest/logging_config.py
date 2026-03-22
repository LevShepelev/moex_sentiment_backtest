from __future__ import annotations

import logging
import sys
from typing import Optional

_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """
    Idempotent logging setup for CLI + library modules.
    Safe to call multiple times.
    """
    lvl = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(lvl)

    # Avoid duplicate handlers if setup_logging() is called multiple times
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT))
        root.addHandler(handler)

    # Reduce noise from some libs (optional)
    for noisy in ("urllib3", "matplotlib", "numba", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Project-wide logger getter used by modules.
    """
    return logging.getLogger(name if name else __name__)
