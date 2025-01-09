"""Settings package initialization."""

from src.settings.config import config as _load_config

config = _load_config()
