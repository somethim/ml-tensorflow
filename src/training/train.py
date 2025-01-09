import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input

from src.settings import config
from src.training.evaluate import evaluate_model
from src.utils.save_model import SaveModel

logger = logging.getLogger(__name__)


def __load_data(data_path: Path) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load training data from the specified path.

    Args:
        data_path: Path to the training data file (.npz format)

    Returns:
        Tuple of (features, labels) tensors
    """
    logger.info(f"Loading training data from {data_path}")
    data = np.load(data_path)
    x_train = data["features"] / 255.0  # Normalize pixel values
    y_train = data["labels"]
    logger.info(f"Loaded {x_train.shape[0]} training samples")
    return x_train, y_train


def __create_model() -> Sequential:
    model = Sequential(
        [
            Input(shape=(28, 28)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


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

    Throws:
        RuntimeError: If training fails
    """
    try:
        # Load data and create model
        x_train, y_train = (
            __load_data(data_dir)
            if data_dir
            else tf.keras.datasets.mnist.load_data()[0]
        )
        model = __create_model()

        # Train the model
        logger.info("Starting model training...")
        history = model.fit(
            x_train,
            y_train,
            epochs=config.training.epochs if epochs is None else epochs,
            batch_size=config.training.batch_size if batch_size is None else batch_size,
            validation_split=config.training.validation_split,
            verbose=1,
        )

        # Evaluate the model
        logger.info("Evaluating model...")
        eval_results = evaluate_model(model)

        # Save the model
        logger.info("Saving model...")
        saver = SaveModel(model_dir)
        save_path = saver.save(
            {
                "training": {
                    "history": history.history,
                    "parameters": {
                        "epochs": epochs or config.training.epochs,
                        "batch_size": batch_size or config.training.batch_size,
                        "validation_split": config.training.validation_split,
                    },
                },
                "evaluation": eval_results,
            }
        )

        return {
            "model": {"path": str(save_path), "version": saver.get_latest_version()},
            "training": {"history": history.history},
            "evaluation": {
                "loss": eval_results["metrics"]["loss"],
                "accuracy": eval_results["metrics"]["accuracy"],
            },
        }

    except Exception as e:
        raise RuntimeError(f"Training failed: {str(e)}")
