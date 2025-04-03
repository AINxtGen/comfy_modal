"""
Constants needed for comfy_image build script.
"""

from typing import NamedTuple
from enum import Enum
import modal # Added for get_config_path
import logging # Added for get_config_path

logger = logging.getLogger(__name__) # Added for get_config_path

class PathConfig(NamedTuple):
    """Base path configuration for different components"""
    base: str
    config: str = None
    output: str = None
    input: str = None
    workflows: str = None

class Paths:
    """Namespace for path-related constants relevant to ComfyUI build"""
    # Cache paths
    CACHE = PathConfig(
        base="/root/.cache/huggingface"
    )
    # Inference paths (ComfyUI)
    INFERENCE = PathConfig(
        base="/root/comfy", # Base directory where ComfyUI is installed by comfy-cli
        output="/root/comfy/ComfyUI/output",
        input="/root/comfy/ComfyUI/input",
        workflows="/root/comfy/ComfyUI/user/default/workflows",
    )

def get_config_path() -> str:
    """Returns the appropriate path to config.toml based on the execution environment."""
    if modal.is_local():
        config_path = "config.toml" # Local execution relative to project root
        logger.debug("Running locally, using config_path: %s", config_path)
    else:
        config_path = "/root/comfy/config.toml" # Modal execution (inside container)
        logger.debug("Running in Modal container, using config_path: %s", config_path)
    return config_path
