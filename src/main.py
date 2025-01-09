"""Main entry point for the ML project."""

import logging
import sys
from pathlib import Path

from src.settings import config
from src.settings.logger import setup_logging
from src.training import train_model
from tools.lint import run_lint

# Ensure logging is set up first
setup_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    if not run_lint():
        logger.error("Linting failed. Please fix the issues before continuing.")
        sys.exit(1)

    """Main entry point that trains and evaluates the model."""
    try:
        # Get paths from settings
        data_dir = Path(config.data.dir)
        train_dir = data_dir / "processed" / "train" if Path(config.data.dir) is not None else None

        # Train and evaluate the model
        save_path = train_model(data_dir=train_dir)

        logger.info("Training completed successfully")
        logger.info(f"Model saved to: {save_path}")

    except ImportError as e:
        logger.error(f"Failed to import required modules: {str(e)}")
        logger.error("Please ensure TensorFlow is properly installed.")
        sys.exit(1)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
