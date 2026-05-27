"""
src/utils/logger.py
───────────────────
Centralised logging setup. Creates a rotating file handler alongside
a colourised console handler so operators can tail logs easily.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# ANSI colour codes for console
_COLOURS = {
    "DEBUG": "\033[36m",      # cyan
    "INFO": "\033[32m",       # green
    "WARNING": "\033[33m",    # yellow
    "ERROR": "\033[31m",      # red
    "CRITICAL": "\033[1;31m", # bold red
    "RESET": "\033[0m",
}


class _ColourFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        reset = _COLOURS["RESET"]
        record.levelname = f"{colour}{record.levelname}{reset}"
        return super().format(record)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a logger with console + rotating-file handlers (idempotent)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler (colourised)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_ColourFormatter(fmt=_FMT, datefmt=_DATE_FMT))
    logger.addHandler(ch)

    # Rotating file handler (plain text)
    fh = RotatingFileHandler(
        _LOG_DIR / "autofe.log",
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter(fmt=_FMT, datefmt=_DATE_FMT))
    logger.addHandler(fh)

    logger.propagate = False
    return logger
