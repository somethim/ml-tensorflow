"""Model saving and versioning utilities."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import keras

from src.settings import config

logger = logging.getLogger(__name__)


class SupportsWrite(Protocol):
    def write(self, __s: str) -> Any: ...


class SaveModel:
    """
    Handles model saving and versioning.

    Args:
        save_path: Optional path to save the model. If not provided, uses default save directory.
    """

    def __init__(self, save_path: Optional[Path] = None) -> None:
        """Initialize model saving."""
        self.base_dir = Path(config.model.saved_models_dir) if save_path is None else save_path

        # Get required metrics and thresholds from settings
        self.required_metrics: List[str] = (
            config.model.required_metrics if hasattr(config.model, "required_metrics") else []
        )
        self.min_accuracy = config.model.min_accuracy

    def save(self, model: keras.Model, metrics: Dict[str, Dict[str, Any]]) -> Path:
        """Save model and version info with metrics.

        Args:
            model: The trained Keras model to save
            metrics: Dictionary of model metrics (e.g. training history)

        Returns:
            Path: Path where model was saved

        Raises:
            ValueError: If required metrics are missing
        """
        # Create version-specific directory
        version = f"{config.model.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version_dir = self.base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save the model in the version directory with .keras extension
        model_path = version_dir / "model.keras"
        model.save(model_path)
        logger.info(f"Model saved to: {model_path}")

        # Save metadata
        version_info = {
            "version": version,
            "path": str(version_dir),
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "model_path": str(model_path),
            "production_ready": self._is_production_ready(
                metrics.get("evaluation", {}).get("metrics", {})
            ),
        }

        metadata_file = version_dir / "metadata.json"
        f: SupportsWrite
        with open(metadata_file, "w") as f:
            json.dump(version_info, f, indent=2)
        logger.info(f"Model metadata saved to: {metadata_file}")

        return version_dir

    def get_latest_version(self) -> Dict[str, Any]:
        """Get latest version info.

        Returns:
            Dictionary containing version information or empty dict if no version exists
        """
        if not self.base_dir.exists():
            return {}

        # Find the latest version directory
        version_dirs = sorted([d for d in self.base_dir.iterdir() if d.is_dir()], reverse=True)
        if not version_dirs:
            return {}

        latest_metadata = version_dirs[0] / "metadata.json"
        if not latest_metadata.exists():
            return {}

        try:
            with open(latest_metadata) as f:
                return dict(json.load(f))
        except json.JSONDecodeError:
            return {}

    def _is_production_ready(self, metrics: Dict[str, float]) -> bool:
        """Check if model meets production requirements.

        Args:
            metrics: Dictionary of model metrics

        Returns:
            True if model meets all requirements
        """
        # Check accuracy threshold
        accuracy = metrics.get("accuracy", 0)
        if accuracy < self.min_accuracy:
            return False

        # Check inference time if specified
        max_inference_time = config.model.max_inference_time
        if "inference_time" in metrics and metrics["inference_time"] > max_inference_time:
            return False

        # Verify all required metrics are above minimum accuracy
        for metric in self.required_metrics:
            metric_value = metrics.get(metric, 0)
            if metric_value < self.min_accuracy:
                return False

        return True
