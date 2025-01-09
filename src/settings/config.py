"""Configuration management system with environment variable support."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration settings."""

    dir: str = Field(default="models", description="Base model directory")
    base_dir: str = Field(default="models", description="Base model directory")
    saved_models_dir: str = Field(default="saved_models", description="Saved models directory")
    checkpoints_dir: str = Field(default="checkpoints", description="Checkpoints directory")
    tensorboard_dir: str = Field(
        default="logs/tensorboard", description="TensorBoard logs directory"
    )
    version: str = Field(default="v1", description="Model version")
    format: str = Field(default="saved_model", description="Model storage format")
    min_accuracy: float = Field(default=0.90, description="Minimum required model accuracy")
    max_inference_time: int = Field(
        default=100, description="Maximum inference time in milliseconds"
    )
    min_precision: float = Field(default=0.85, description="Minimum required precision")
    min_recall: float = Field(default=0.85, description="Minimum required recall")
    min_f1_score: float = Field(default=0.85, description="Minimum required F1 score")


class TrainingConfig(BaseModel):
    """Training configuration settings."""

    batch_size: int = Field(default=32, description="Training batch size")
    epochs: int = Field(default=10, description="Number of training epochs")
    learning_rate: float = Field(default=0.001, description="Model learning rate")
    validation_split: float = Field(default=0.2, description="Validation data split ratio")
    early_stopping_patience: int = Field(default=5, description="Early stopping patience")
    reduce_lr_patience: int = Field(default=3, description="Learning rate reduction patience")
    reduce_lr_factor: float = Field(default=0.2, description="Learning rate reduction factor")
    shuffle_buffer_size: int = Field(default=10000, description="Dataset shuffle buffer size")


class DataConfig(BaseModel):
    """Data management configuration."""

    dir: str = Field(default="data", description="Base data directory")
    base_dir: str = Field(default="data", description="Base data directory")
    raw_dir: str = Field(default="raw", description="Raw data directory")
    processed_dir: str = Field(default="processed", description="Processed data directory")
    cache_dir: str = Field(default=".cache", description="Cache directory")
    temp_dir: str = Field(default="/tmp/ml-tensorflow", description="Temporary directory")
    max_cache_size: int = Field(default=10, description="Maximum cache size in GB")
    compression_format: str = Field(default="gzip", description="Data compression format")


class MonitoringConfig(BaseModel):
    """Monitoring and logging configuration."""

    log_level: str = Field(default="INFO", description="Logging level")
    enable_file_logging: bool = Field(default=True, description="Enable file logging")
    log_dir: str = Field(default="logs", description="Log directory")
    monitoring_interval: int = Field(default=60, description="Monitoring interval in seconds")
    alert_threshold: float = Field(default=0.85, description="Alert threshold")
    enable_mlflow: bool = Field(default=True, description="Enable MLflow tracking")
    enable_wandb: bool = Field(default=False, description="Enable Weights & Biases tracking")
    metrics_tracking: bool = Field(default=True, description="Enable metrics tracking")


def _load_all_configs() -> Dict[str, Any]:
    """Load all configuration files."""
    config_dir = Path("configs")
    config_files = {
        "model": config_dir / "model.yml",
        "training": config_dir / "training.yml",
        "environment": config_dir / "environment.yml",
    }

    cfg = {}
    for name, path in config_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            cfg[name] = yaml.safe_load(f) or {}

    return cfg


class Config:
    """Main configuration class with Laravel-like interface."""

    def __init__(self) -> None:
        self._config = _load_all_configs()

        # Initialize configurations
        self.model = ModelConfig(**self._get_config_with_env("model", "storage"))
        self.training = TrainingConfig(**self._get_config_with_env("training", "training"))
        self.data = DataConfig(**self._get_config_with_env("data", "data"))
        self.monitoring = MonitoringConfig(**self._get_config_with_env("monitoring", "monitoring"))

    def _get_config_with_env(self, config_file: str, section: str) -> dict[str, Any]:
        """Get configuration with environment variable overrides."""
        cfg = dict(self._config.get(config_file, {}).get(section, {}))

        # Override with environment variables
        env_prefix = "ML_"
        for key in cfg.keys():
            env_key = f"{env_prefix}{key.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                # Convert value to appropriate type
                if isinstance(cfg[key], bool):
                    cfg[key] = value.lower() in ("true", "1", "yes")
                elif isinstance(cfg[key], int):
                    cfg[key] = int(value)
                elif isinstance(cfg[key], float):
                    cfg[key] = float(value)
                else:
                    cfg[key] = value

        return cfg

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get configuration value using dot notation."""
        try:
            config_file, section, *parts = key.split(".")
            value = self._config[config_file][section]
            for part in parts:
                value = value[part]
            return str(value)
        except (KeyError, TypeError):
            return default


@lru_cache()
def config() -> Config:
    """Get the global configuration instance."""
    return Config()
