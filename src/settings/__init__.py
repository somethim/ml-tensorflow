"""Settings package initialization."""

from .config import config as _load_config
from .tensorflow import configure_tensorflow

# Load global configuration
config = _load_config()

# Configure TensorFlow
configure_tensorflow()
