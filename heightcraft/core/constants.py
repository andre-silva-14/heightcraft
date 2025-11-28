"""
Constants for Heightcraft.

This module defines constants used throughout the Heightcraft application.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple


# Application information
APP_NAME = "Heightcraft"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "A powerful and flexible height map generator for 3D models"
APP_AUTHOR = "Heightcraft Team"

# File extensions
SUPPORTED_MODEL_EXTENSIONS = [".obj", ".stl", ".ply", ".glb", ".gltf"]
SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".tif"]

# Defaults
DEFAULT_MAX_RESOLUTION = 256
DEFAULT_BIT_DEPTH = 16
DEFAULT_NUM_SAMPLES = 100000
DEFAULT_NUM_THREADS = 4
DEFAULT_CACHE_DIR = os.path.join(os.getcwd(), ".cache")
DEFAULT_OUTPUT_PATH = "height_map.png"
DEFAULT_CHUNK_SIZE = 1000000
DEFAULT_MAX_MEMORY = 0.8  # 80% of available memory

# GPU settings
GPU_MEMORY_SAFETY_MARGIN = 0.1  # 10% safety margin for GPU memory

# Threading settings
DEFAULT_THREAD_TIMEOUT = 600  # 10 minutes
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds

# Performance settings
PROFILE_ENABLED_BY_DEFAULT = True

# Mesh processing
MIN_VERTICES_FOR_LARGE_MODEL = 1000000
MIN_FACES_FOR_LARGE_MODEL = 500000

# Height map generation
MIN_HEIGHT_MAP_DIMENSION = 32
MAX_HEIGHT_MAP_DIMENSION = 16384

# Upscaling
SUPPORTED_UPSCALE_FACTORS = [2, 3, 4]
DEFAULT_UPSCALE_FACTOR = 2

# Cache settings
DEFAULT_CACHE_MAX_AGE = 86400  # 1 day in seconds

# Logging
DEFAULT_LOG_LEVEL = "info"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
DEFAULT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Command-line interface
CLI_COMMAND_GROUPS = {
    "Output Options": [
        "output_path",
        "max_resolution",
        "bit_depth",
        "split",
    ],
    "Processing Options": [
        "use_gpu",
        "num_samples",
        "num_threads",
    ],
    "Large Model Options": [
        "large_model",
        "chunk_size",
        "max_memory",
        "cache_dir",
    ],
    "Upscaling Options": [
        "upscale",
        "upscale_factor",
        "pretrained_model",
    ],
    "Miscellaneous Options": [
        "test",
        "verbose",
    ],
}

# File paths
def get_default_paths() -> Dict[str, str]:
    """
    Get default paths for the application.
    
    Returns:
        Dictionary of default paths
    """
    base_dir = os.getcwd()
    return {
        "cache_dir": os.path.join(base_dir, ".cache"),
        "output_dir": base_dir,
        "config_file": os.path.join(base_dir, "heightcraft.config"),
        "log_file": os.path.join(base_dir, "heightcraft.log"),
    } 