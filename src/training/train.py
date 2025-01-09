import logging
from pathlib import Path
from typing import Optional, Tuple

import keras
import numpy as np
import tensorflow as tf
from keras import Input, Sequential
from keras.src.layers import Dense, Flatten

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
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def train_model(
    data_dir: Optional[Path],
    save_path: Optional[Path] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Path:
    """
    Train a model on the provided data, evaluate it, and save results.

    Args:
        data_dir: Directory containing training data
        save_path: Path to save the trained model
        epochs: Number of training epochs (overrides settings)
        batch_size: Training batch size (overrides settings)

    Returns:
        dict: Dictionary containing training history, evaluation metrics, and model info

    Throws:
        RuntimeError: If training fails
    """
    try:
        # Load data and create model
        if data_dir and data_dir.exists():
            logger.info(f"Loading data from {data_dir}")
            x_train, y_train = __load_data(data_dir)
        else:
            logger.info("Using MNIST dataset as fallback")
            (x_train, y_train), _ = keras.datasets.mnist.load_data()
            x_train = tf.cast(x_train, tf.float32) / 255.0  # Convert to float and normalize

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
        saver = SaveModel(save_path)
        save_path = saver.save(
            model=model,
            metrics={
                "training": {
                    "history": history.history,
                    "parameters": {
                        "epochs": epochs or config.training.epochs,
                        "batch_size": batch_size or config.training.batch_size,
                        "validation_split": config.training.validation_split,
                    },
                },
                "evaluation": eval_results,
            },
        )

        return save_path

    except Exception as e:
        raise RuntimeError(f"Training failed: {str(e)}")
