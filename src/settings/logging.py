"""Logging configuration for the application."""

import logging
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application.

    Args:
        level: The logging level to use. Defaults to INFO.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: The name for the logger. If None, returns the root logger.

    Returns:
        A configured logger instance.
    """
    return logging.getLogger(name)


setup_logging()
