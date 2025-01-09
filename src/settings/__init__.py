"""Settings package initialization."""

from src.settings.config import config as _load_config
from src.settings.tensorflow import configure_tensorflow

# Load global configuration
config = _load_config()

# Configure TensorFlow
configure_tensorflow()
