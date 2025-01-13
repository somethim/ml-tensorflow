from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np
import numpy.typing as npt
from keras import Input, Sequential
from keras.src.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

from src.settings import config, logger
from src.utils.preprocessor_factory import PreprocessorFactory
from src.utils.save_model import SaveModel


class TrainModel:
    """Class to handle data of a Keras model."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int,
        preprocessor_type: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        data_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the data configuration.

        Args:
            input_shape: Shape of input data (excluding batch dimension)
            num_classes: Number of output classes
            preprocessor_type: Type of preprocessor to use ('image', 'text', or 'tabular')
            model_config: Configuration for model architecture (layers, activations, etc.)
            data_config: Configuration for data loading (normalization, preprocessing, etc.)
        """
        self.config = config
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_config = model_config or {}
        self.data_config = data_config or {
            "normalization_factor": 1.0,
            "data_format": "npz",
            "feature_key": "features",
            "label_key": "labels",
        }

        # Initialize preprocessor if specified
        self.preprocessor = None
        if preprocessor_type:
            self.preprocessor = PreprocessorFactory.get_preprocessor(preprocessor_type)

    @staticmethod
    def __load_data(data_path: Path) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Load preprocessed data from npz files."""
        logger.info(f"Loading data from {data_path}")

        data = np.load(data_path / "data.npz")
        return data["features"], data["labels"]

    def __create_model(self) -> Sequential:
        """Create and compile the model based on configuration."""
        # Use model_config to customize architecture, or fall back to default CNN
        layers = self.model_config.get(
            "layers",
            [
                Input(shape=self.input_shape),
                # First convolutional block
                Conv2D(32, (3, 3), activation="relu", padding="same"),
                Conv2D(32, (3, 3), activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                # Second convolutional block
                Conv2D(64, (3, 3), activation="relu", padding="same"),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                # Third convolutional block
                Conv2D(128, (3, 3), activation="relu", padding="same"),
                Conv2D(128, (3, 3), activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                # Dense layers
                Flatten(),
                Dense(512, activation="relu"),
                Dropout(0.5),
                Dense(self.num_classes, activation="softmax"),
            ],
        )

        model = Sequential(layers)

        # Get compilation parameters from config or use defaults
        optimizer = self.model_config.get("optimizer", "adam")
        loss = self.model_config.get("loss", "sparse_categorical_crossentropy")
        metrics = self.model_config.get(
            "metrics",
            [
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
            ],
        )

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
        return model

    def train(
        self,
        train_data: Path,
        evaluate_data: Path,
        save_path: Optional[Path] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Path:
        """Train a model on the provided data, evaluate it, and save results."""
        try:
            # Load data and create model
            if not train_data.exists():
                raise ValueError(f"Invalid data directory: {train_data}")

            x_train, y_train = self.__load_data(train_data)
            x_val, y_val = self.__load_data(evaluate_data)

            model = self.__create_model()

            # Train the model
            logger.info("Starting model training...")
            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=epochs or self.config.training.epochs,
                batch_size=batch_size or self.config.training.batch_size,
                verbose=1,
            )

            # Save the model
            logger.info("Saving model...")
            save_dir = Path(save_path) if save_path else Path(self.config.model.saved_models_dir)
            saver = SaveModel(save_dir)
            save_path = saver.save(
                model=model,
                metrics={
                    "training": {
                        "history": history.history,
                        "parameters": {
                            "epochs": epochs or self.config.training.epochs,
                            "batch_size": batch_size or self.config.training.batch_size,
                            "train_samples": len(x_train),
                            "val_samples": len(x_val),
                        },
                    },
                },
            )

            return save_path

        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")
