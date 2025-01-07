"""Configuration management system with environment variable support."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration settings."""

    dir: str = Field(default="models", description="Model storage directory")
    version: str = Field(default="v1", description="Model version")
    format: str = Field(default="saved_model", description="Model storage format")
    min_accuracy: float = Field(default=0.90, description="Minimum required model accuracy")
    max_inference_time: int = Field(
        default=100, description="Maximum inference time in milliseconds"
    )


class TrainingConfig(BaseModel):
    """Training configuration settings."""

    batch_size: int = Field(default=32, description="Training batch size")
    epochs: int = Field(default=10, description="Number of training epochs")
    learning_rate: float = Field(default=0.001, description="Model learning rate")
    validation_split: float = Field(default=0.2, description="Validation data split ratio")


class ProductionConfig(BaseModel):
    """Production environment settings."""

    monitoring: bool = Field(default=True, description="Enable monitoring")
    version_control: bool = Field(default=True, description="Enable version control")
    metrics_tracking: bool = Field(default=True, description="Enable metrics tracking")
    monitoring_interval: int = Field(default=60, description="Monitoring interval in seconds")
    alert_threshold: float = Field(default=0.85, description="Alert threshold for metrics")


class DataConfig(BaseModel):
    """Data management configuration."""

    dir: str = Field(default="data", description="Data storage directory")
    cache_dir: str = Field(default=".cache", description="Cache directory")


class Config:
    """Main configuration class with Laravel-like interface."""

    def __init__(self) -> None:
        self._config = self._load_config()
        # Override settings values with environment variables
        self.model = ModelConfig(
            dir=os.getenv("ML_MODEL_DIR", self._config.get("model", {}).get("dir")),
            version=os.getenv("ML_MODEL_VERSION", self._config.get("model", {}).get("version")),
            format=os.getenv("ML_MODEL_FORMAT", self._config.get("model", {}).get("format")),
            min_accuracy=float(
                os.getenv(
                    "ML_MIN_ACCURACY", self._config.get("model", {}).get("min_accuracy", 0.90)
                )
            ),
            max_inference_time=int(
                os.getenv(
                    "ML_MAX_INFERENCE_TIME",
                    self._config.get("model", {}).get("max_inference_time", 100),
                )
            ),
        )
        self.training = TrainingConfig(
            batch_size=int(
                os.getenv("ML_BATCH_SIZE", self._config.get("training", {}).get("batch_size", 32))
            ),
            epochs=int(os.getenv("ML_EPOCHS", self._config.get("training", {}).get("epochs", 10))),
            learning_rate=float(
                os.getenv(
                    "ML_LEARNING_RATE", self._config.get("training", {}).get("learning_rate", 0.001)
                )
            ),
            validation_split=float(
                os.getenv(
                    "ML_VAL_SPLIT", self._config.get("training", {}).get("validation_split", 0.2)
                )
            ),
        )
        self.production = ProductionConfig(
            monitoring=os.getenv(
                "ML_MONITORING", self._config.get("production", {}).get("monitoring", True)
            )
            in ("true", "1", "yes"),
            version_control=os.getenv(
                "ML_VERSION_CONTROL",
                self._config.get("production", {}).get("version_control", True),
            )
            in ("true", "1", "yes"),
            metrics_tracking=os.getenv(
                "ML_METRICS_TRACKING",
                self._config.get("production", {}).get("metrics_tracking", True),
            )
            in ("true", "1", "yes"),
            monitoring_interval=int(
                os.getenv(
                    "ML_MONITORING_INTERVAL",
                    self._config.get("production", {}).get("monitoring_interval", 60),
                )
            ),
            alert_threshold=float(
                os.getenv(
                    "ML_ALERT_THRESHOLD",
                    self._config.get("production", {}).get("alert_threshold", 0.85),
                )
            ),
        )
        self.data = DataConfig(
            dir=os.getenv("ML_DATA_DIR", self._config.get("data", {}).get("dir")),
            cache_dir=os.getenv("ML_CACHE_DIR", self._config.get("data", {}).get("cache_dir")),
        )

    @staticmethod
    def _load_config() -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path("settings/settings.yml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get settings value using dot notation."""
        try:
            parts = key.split(".")
            value = self._config
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default


@lru_cache()
def config() -> Config:
    """Get the global settings instance."""
    return Config()
