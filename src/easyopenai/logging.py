"""Centralised loguru configuration for the library.

All modules should `from loguru import logger` directly; this module's
`setup_logging()` function configures sinks, formatters, and key-masking
filters once at application startup.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from loguru import logger

_MASK_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9]{4})[A-Za-z0-9]+([A-Za-z0-9]{4})"),
    re.compile(r"(Bearer\s+[A-Za-z0-9]{4})[A-Za-z0-9]+([A-Za-z0-9]{4})"),
]


def _mask_record(record: dict) -> bool:
    msg = record["message"]
    for pat in _MASK_PATTERNS:
        msg = pat.sub(r"\1****\2", msg)
    record["message"] = msg
    return True


_configured = False


def setup_logging(level: str = "INFO", file: str | Path | None = None) -> None:
    global _configured
    if _configured:
        return
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        filter=_mask_record,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    if file is not None:
        logger.add(
            str(file),
            level=level,
            filter=_mask_record,
            rotation="10 MB",
            retention=5,
            encoding="utf-8",
        )
    _configured = True
