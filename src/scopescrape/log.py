"""Logging configuration for ScopeScrape.

Provides a pre-configured logger with verbose/quiet modes
and colored output via Rich when available.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """Configure and return the scopescrape logger.

    Args:
        verbose: If True, set level to DEBUG.
        quiet: If True, set level to WARNING. Overrides verbose.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("scopescrape")

    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)

        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        datefmt = "%H:%M:%S"
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setLevel(level)

    return logger


def get_logger(name: str = "scopescrape") -> logging.Logger:
    """Get a child logger for a specific module.

    Usage:
        from scopescrape.log import get_logger
        logger = get_logger(__name__)
        logger.info("Starting adapter...")
    """
    return logging.getLogger(name)
