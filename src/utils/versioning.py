"""Model versioning utilities."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from _typeshed import SupportsWrite

from src.config import config


class ModelVersion:
    """Handles model versioning and metadata."""

    def __init__(self, model_path: Path):
        """Initialize model versioning.

        Args:
            model_path: Path to model directory
        """
        self.model_path = Path(model_path)
        self.version_file = self.model_path / "version.json"

        # Get required metrics and thresholds from config
        self.required_metrics = config().get("model.performance.required_metrics", [])
        self.min_accuracy = config().model.min_accuracy

    def save_version(self, metrics: Dict[str, float], version: str) -> None:
        """Save model version info with metrics.

        Args:
            metrics: Dictionary of model metrics
            version: Version identifier

        Raises:
            ValueError: If required metrics are missing
        """
        # Validate required metrics are present
        missing_metrics = [m for m in self.required_metrics if m not in metrics]
        if missing_metrics:
            raise ValueError(f"Missing required metrics: {missing_metrics}")

        version_info = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "production_ready": self._is_production_ready(metrics),
        }

        # Create parent directories if they don't exist
        self.version_file.parent.mkdir(parents=True, exist_ok=True)

        f: SupportsWrite[str]
        with open(self.version_file, "w") as f:
            json.dump(version_info, f, indent=2)

    def get_latest_version(self) -> Dict[str, Any]:
        """Get latest version info.

        Returns:
            Dictionary containing version information or empty dict if no version exists
        """
        if not self.version_file.exists():
            return {}

        try:
            with open(self.version_file) as f:
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
        if metrics.get("accuracy", 0) < self.min_accuracy:
            return False

        # Check inference time if specified
        max_inference_time = config().model.max_inference_time
        if "inference_time" in metrics and metrics["inference_time"] > max_inference_time:
            return False

        # Verify all required metrics are above minimum accuracy
        return all(metrics.get(metric, 0) >= self.min_accuracy for metric in self.required_metrics)
