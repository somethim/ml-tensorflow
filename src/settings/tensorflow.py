"""TensorFlow configuration module."""

import logging
import os

import tensorflow as tf

from src.settings.config import TensorFlowConfig, config

logger = logging.getLogger(__name__)


def _configure_gpu(cfg: TensorFlowConfig) -> None:
    """Configure GPU settings based on configuration."""
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        logger.info("No GPU devices found, falling back to CPU")
        return

    try:
        # Configure visible devices
        if cfg.gpu_devices:
            visible_gpus = [gpus[i] for i in cfg.gpu_devices if i < len(gpus)]
            tf.config.set_visible_devices(visible_gpus, "GPU")

        # Configure memory growth
        if cfg.memory_growth:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        # Configure memory limits
        if cfg.gpu_memory_limit > 0:
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=cfg.gpu_memory_limit)],
                )

        logger.info(f"GPU configuration complete. Using {len(gpus)} GPU(s)")
    except RuntimeError as e:
        logger.warning(f"GPU configuration error: {e}")


def _configure_cpu(cfg: TensorFlowConfig) -> None:
    """Configure CPU settings based on configuration."""
    if cfg.parallel_threads > 0:
        tf.config.threading.set_inter_op_parallelism_threads(cfg.parallel_threads)
        tf.config.threading.set_intra_op_parallelism_threads(cfg.parallel_threads)

    if cfg.enable_mkl:
        os.environ["TF_ENABLE_MKL_NATIVE_FORMAT"] = "1"

    logger.info("CPU configuration complete")


def _configure_logging(cfg: TensorFlowConfig) -> None:
    """Configure TensorFlow logging based on configuration."""
    # Base logging level
    log_level = "0" if cfg.debug_mode else "2"

    # Conditional warning suppressions
    if cfg.suppress_warnings.onednn:
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    if cfg.suppress_warnings.cuda_missing:
        os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"  # Suppress CUDA-related warnings

    if cfg.suppress_warnings.registration_conflicts:
        # Suppress registration conflict warnings
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # Configure logging format and outputs
    if cfg.logging.file_output:
        tf.get_logger().addHandler(logging.FileHandler("tensorflow.log"))

    # Set final logging level
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = log_level


def configure_tensorflow() -> None:
    """Configure TensorFlow environment based on settings."""
    cfg = config().tensorflow

    # Configure logging first
    _configure_logging(cfg)

    # Configure compute mode
    if cfg.compute_mode == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        _configure_cpu(cfg)
        logger.info("Using CPU for computation")
    elif cfg.compute_mode in ["gpu", "multi_gpu"]:
        if not tf.test.is_built_with_cuda():
            if not cfg.suppress_warnings.cuda_missing:
                logger.warning("CUDA not available despite GPU mode being selected")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            _configure_cpu(cfg)
        else:
            _configure_gpu(cfg)
            _configure_cpu(cfg)
        logger.info(f"Using {cfg.compute_mode.upper()} for computation")
    else:  # "auto"
        if tf.test.is_built_with_cuda():
            _configure_gpu(cfg)
            _configure_cpu(cfg)
            logger.info("Automatically selected available compute devices")
        else:
            if not cfg.suppress_warnings.cuda_missing:
                logger.info("CUDA not available, using CPU for computation")
            _configure_cpu(cfg)

    # Configure optimization flags conditionally
    if cfg.enable_onednn_opts and not cfg.suppress_warnings.onednn:
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

    if cfg.xla_acceleration:
        tf.config.optimizer.set_jit(True)

    if cfg.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    if cfg.enable_op_determinism:
        tf.config.experimental.enable_op_determinism()

    # Configure distribution strategy if needed
    if cfg.distribution_strategy != "auto" and cfg.num_workers > 1:
        if cfg.distribution_strategy == "mirrored":
            tf.distribute.MirroredStrategy()
        elif cfg.distribution_strategy == "multi_worker":
            tf.distribute.MultiWorkerMirroredStrategy()

    # Log device placement if requested
    tf.debugging.set_log_device_placement(cfg.log_device_placement)

    logger.info("TensorFlow configuration complete")
