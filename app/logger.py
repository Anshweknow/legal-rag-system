"""Application logging helpers."""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure application logging once for console output."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=False,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for application modules."""
    return logging.getLogger(name)
