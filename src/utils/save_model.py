"""Model saving and versioning utilities."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.settings import config


class SaveModel:
    """Handles model saving and versioning."""

    def __init__(self, model_dir: Path):
        """Initialize model saving.

        Args:
            model_dir: Path to save model and version info
        """
        self.model_dir = Path(model_dir)
        self.version_file = self.model_dir / "version.json"

        # Get required metrics and thresholds from settings
        self.required_metrics: List[str] = (
            config.model.required_metrics
            if hasattr(config.model, "required_metrics")
            else []
        )
        self.min_accuracy = config.model.min_accuracy

    def save(self, metrics: Dict[str, Dict[str, Any]]) -> Path:
        """Save model version info with metrics.

        Args:
            metrics: Dictionary of model metrics (e.g. training history)

        Returns:
            Path: Path where model was saved

        Raises:
            ValueError: If required metrics are missing
        """
        # Create version string based on timestamp
        version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create model directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)

        version_info = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "production_ready": self._is_production_ready(
                metrics.get("evaluation", {}).get("metrics", {})
            ),
        }

        with open(self.version_file, "w") as f:
            json.dump(version_info, f, indent=2)

        return self.model_dir

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
        max_inference_time = config.model.max_inference_time
        if (
            "inference_time" in metrics
            and metrics["inference_time"] > max_inference_time
        ):
            return False

        # Verify all required metrics are above minimum accuracy
        return all(
            metrics.get(metric, 0) >= self.min_accuracy
            for metric in self.required_metrics
        )
