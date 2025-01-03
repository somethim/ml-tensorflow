"""Model training functionality."""

from pathlib import Path
from typing import Optional

from src.config import config


def train_model(
    data_dir: Path,
    model_dir: Path,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Optional[str]:
    """Train a model on the provided data.

    Args:
        data_dir: Directory containing training data
        model_dir: Directory to save trained model
        epochs: Number of training epochs (overrides config)
        batch_size: Training batch size (overrides config)
    """
    # Get training config
    cfg = config()
    epochs = epochs or cfg.training.epochs
    batch_size = batch_size or cfg.training.batch_size

    print(epochs, batch_size, data_dir, model_dir)
    # todo: Implement train logic

    return None
