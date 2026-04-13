# functions/utils/logging.py
from __future__ import annotations

"""
functions.utils.logging

Centralized logging utilities for the HyDE Feed Recommendation project.

Design goals
------------
- Single, consistent logger configuration across:
  - local development
  - Docker containers
  - Cloud Run / managed environments
- Safe to call repeatedly (idempotent):
  - No duplicate handlers
  - No duplicated log lines on re-import
- stdout-only logging:
  - Required for Cloud Run / container log collectors
- Minimal abstraction:
  - This is NOT a logging framework
  - No JSON logging, no structured payloads (by design for POC)

Environment variables
---------------------
- LOG_LEVEL:
    Controls default log level.
    Examples: DEBUG, INFO, WARNING, ERROR
    Default: INFO

Log format
----------
ISO-like timestamp (UTC-style suffix), level, logger name, message:

    2026-01-16T10:32:01Z | INFO | functions.utils.llm_client | LLM call done | ...

Why not use basicConfig?
------------------------
- basicConfig is global and hard to control in libraries
- It can break test isolation
- It makes idempotency harder to guarantee

This helper ensures:
- Explicit handler setup
- Explicit formatter
- Predictable behavior across imports
"""

import logging
import os
from typing import Optional


# Default log level (env override friendly for Cloud Run / Docker)
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger that writes to stdout.

    Properties
    ----------
    - Idempotent:
        Calling get_logger() multiple times with the same name will NOT
        attach duplicate handlers.
    - Non-propagating:
        Prevents double-logging when root logger is also configured.
    - Environment-aware:
        LOG_LEVEL env var controls default verbosity.

    Parameters
    ----------
    name:
        Logger name, typically __name__ of the calling module.
    level:
        Optional explicit log level override (e.g., "DEBUG").
        If None, falls back to LOG_LEVEL env var.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Resolve log level (explicit argument > env var > INFO)
    lvl = (level or _LOG_LEVEL).upper()
    logger.setLevel(getattr(logging, lvl, logging.INFO))

    # Attach handler only once (critical for reloads, tests, notebooks)
    if not logger.handlers:
        handler = logging.StreamHandler()  # stdout by default

        formatter = logging.Formatter(
            fmt="%(asctime)sZ | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent propagation to root logger (avoids duplicated lines)
    logger.propagate = False

    return logger
