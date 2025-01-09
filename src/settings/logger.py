"""Logging configuration for the ML project."""

import logging
import sys
from pathlib import Path

from src.settings.config import config


def setup_logging() -> None:
    """Configure logging for both file and console output."""
    # Create logs directory if it doesn't exist
    log_dir = Path(config().monitoring.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create formatters
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
    )

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # File handler
    if config().monitoring.enable_file_logging:
        file_handler = logging.FileHandler(log_dir / "app.log")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Add handlers
    root_logger.addHandler(console_handler)
    if config().monitoring.enable_file_logging:
        root_logger.addHandler(file_handler)


# Initialize logging when this module is imported
setup_logging()
