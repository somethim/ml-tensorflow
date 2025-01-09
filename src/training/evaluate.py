import logging
from pathlib import Path
from typing import Any, Dict, Optional

import keras
import numpy as np

logger = logging.getLogger(__name__)


def evaluate_model(
    model: keras.Model,
    test_data_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate a trained model.

    Args:
        model: Trained model to evaluate
        test_data_path: Optional path to test dataset. If not provided, uses MNIST test set

    Returns:
        Dictionary containing evaluation metrics and metadata
    """
    if test_data_path is not None:
        logger.info(f"Loading test data from {test_data_path}")
        try:
            test_data = np.load(test_data_path)
            x_test = test_data["x_test"]
            y_test = test_data["y_test"]
            x_test = x_test / 255.0
        except Exception as e:
            logger.error(f"Failed to load test data from {test_data_path}: {str(e)}")
            logger.warning("Falling back to default MNIST test data")
            _, (x_test, y_test) = keras.datasets.mnist.load_data()
            x_test = x_test / 255.0
    else:
        logger.info("Loading default MNIST test dataset...")
        _, (x_test, y_test) = keras.datasets.mnist.load_data()
        x_test = x_test / 255.0

    logger.info(f"Evaluating model on {len(x_test)} test samples...")
    eval_results = model.evaluate(x_test, y_test, verbose=0)
    logger.info("Evaluation complete")

    metrics = dict(zip(model.metrics_names, eval_results))

    return {"metrics": metrics, "test_samples": len(x_test)}
