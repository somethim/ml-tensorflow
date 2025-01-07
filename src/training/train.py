"""Model training functionality."""

from pathlib import Path
from typing import Any, Dict, Optional

import tensorflow as tf
from src.settings import config, get_logger
from src.utils import SaveModel

from .evaluate import evaluate_model

logger = get_logger(__name__)


def train_model(
    data_dir: Path,
    model_dir: Path,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train a model on the provided data, evaluate it, and save results.

    Args:
        data_dir: Directory containing training data
        model_dir: Directory to save trained model
        epochs: Number of training epochs (overrides settings)
        batch_size: Training batch size (overrides settings)

    Returns:
        dict: Dictionary containing training history, evaluation metrics, and model info
    """
    cfg = config()
    epochs = epochs or cfg.training.epochs
    batch_size = batch_size or cfg.training.batch_size

    logger.info(f"Training on {data_dir}")
    logger.info(f"Using batch size: {batch_size}, epochs: {epochs}")

    logger.info("Loading MNIST dataset...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train / 255.0
    logger.info(f"Loaded {len(x_train)} training samples")

    logger.info("Building model...")
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    logger.info("Starting training...")
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Get test data path if it exists
    test_data_path = data_dir / "test.npz" if data_dir.exists() else None
    evaluation_results = evaluate_model(model, test_data_path)

    results = {
        "training": {
            "history": history.history,
            "epochs": epochs,
            "batch_size": batch_size,
            "samples": len(x_train),
        },
        "evaluation": evaluation_results,
    }

    logger.info(f"Saving model to {model_dir}")
    save_path = SaveModel(model_dir).save(results)

    results["model"] = {"path": str(save_path), "version": cfg.model.version}

    return results
