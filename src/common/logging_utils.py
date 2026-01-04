# File: MedicalCare+/src/common/logging_utils.py

"""
logging_utils.py

Centralized logging configuration for MedicalCare+.

Industrial & clinical guarantees:
- Single logging policy across entire system
- Deterministic, structured, and timestamped logs
- CLI, training, inference, evaluation safe
- Designed for audit, debugging, and regulatory review
- No duplicate handlers (critical in large systems)
"""

import logging
import sys
from typing import Optional

from src.common.constants import PROJECT_NAME, PROJECT_VERSION

# --------------------------------------------------
# Internal registry to avoid duplicate handlers
# --------------------------------------------------
_LOGGER_REGISTRY = {}


def setup_logger(
    name: str,
    level: int = logging.INFO,
    stream: Optional[object] = sys.stdout
) -> logging.Logger:
    """
    Creates or retrieves a configured logger.

    This function guarantees:
    - No duplicate handlers
    - Consistent formatting
    - Safe reuse across modules

    Args:
        name (str): Logger name (usually __name__)
        level (int): Logging level (default: INFO)
        stream (object): Output stream (stdout by default)

    Returns:
        logging.Logger
    """

    if name in _LOGGER_REGISTRY:
        return _LOGGER_REGISTRY[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # prevent double logging

    # --------------------------------------------------
    # Formatter (auditable & readable)
    # --------------------------------------------------
    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)-7s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --------------------------------------------------
    # Stream handler (CLI / notebooks / servers)
    # --------------------------------------------------
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # --------------------------------------------------
    # Context banner (once per logger)
    # --------------------------------------------------
    logger.debug(
        f"{PROJECT_NAME} v{PROJECT_VERSION} logger initialized"
    )

    _LOGGER_REGISTRY[name] = logger
    return logger
