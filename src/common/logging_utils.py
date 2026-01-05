"""
logging_utils.py

Centralized logging configuration for MedicalCare+.

CRITICAL FILE.

Industrial & clinical guarantees:
- Single logging policy across entire system
- Deterministic, structured, timestamped logs
- Process & thread traceability (AUDIT REQUIRED)
- CLI, training, inference, evaluation safe
- No duplicate handlers (CRITICAL)
- Optional file-based logging (clinical deployments)
"""

import logging
import sys
import os
from typing import Optional

from src.common.constants import PROJECT_NAME, PROJECT_VERSION

# --------------------------------------------------
# Internal registry to avoid duplicate handlers
# --------------------------------------------------
_LOGGER_REGISTRY: dict[str, logging.Logger] = {}


# ==================================================
# LOGGER FACTORY (SINGLE SOURCE OF TRUTH)
# ==================================================
def setup_logger(
    name: str,
    level: int = logging.INFO,
    stream: Optional[object] = sys.stdout,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Creates or retrieves a configured logger.

    Guarantees:
    - No duplicate handlers
    - Deterministic formatting
    - Process & thread traceability
    - Optional persistent file logging

    Args:
        name (str): Logger name (usually __name__)
        level (int): Logging level
        stream (object): Output stream (stdout by default)
        log_file (str | None): Optional log file path

    Returns:
        logging.Logger
    """

    if name in _LOGGER_REGISTRY:
        return _LOGGER_REGISTRY[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent root logger duplication

    # --------------------------------------------------
    # Formatter (AUDIT-GRADE)
    # --------------------------------------------------
    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)-7s | "
            "%(process)d | "
            "%(threadName)s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --------------------------------------------------
    # Stream handler (CLI / notebooks)
    # --------------------------------------------------
    if stream is not None:
        stream_handler = logging.StreamHandler(stream)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # --------------------------------------------------
    # Optional file handler (CLINICAL / PROD)
    # --------------------------------------------------
    if log_file is not None:
        log_file = os.path.abspath(log_file)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # --------------------------------------------------
    # Initialization banner (ONCE)
    # --------------------------------------------------
    logger.info(
        f"{PROJECT_NAME} v{PROJECT_VERSION} logger initialized"
    )

    _LOGGER_REGISTRY[name] = logger
    return logger
