"""Main entry point for the ML project."""

import logging
import sys
from pathlib import Path

from src.settings import config

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point that trains and evaluates the model."""
    try:
        logger.info("Loading configuration...")

        # Get paths from settings
        model_dir = Path(config.model.dir) / config.model.version
        data_dir = Path(config.data.dir)
        train_dir = data_dir / "train"

        logger.info(f"Using model directory: {model_dir}")
        logger.info(f"Using training data directory: {train_dir}")

        # Import training module after TensorFlow is configured
        from src.training import train_model

        # Train and evaluate the model
        logger.info("Starting model training and evaluation...")
        results = train_model(
            data_dir=train_dir,
            model_dir=model_dir,
        )

        # Log results summary
        logger.info(f"Model saved to: {results['model']['path']}")
        logger.info("Training Results:")
        logger.info(f"- Final loss: {results['training']['history']['loss'][-1]:.4f}")
        logger.info(f"- Final accuracy: {results['training']['history']['accuracy'][-1]:.4f}")
        logger.info("Evaluation Results:")
        logger.info(f"- Test loss: {results['evaluation']['loss']:.4f}")
        logger.info(f"- Test accuracy: {results['evaluation']['accuracy']:.4f}")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
