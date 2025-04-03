# comfy_ui.py
"""
ComfyUI web interface
"""

import subprocess
import modal
import logging
import toml

# Use local utilities and image definition
from _utils.constants import Paths, get_config_path # Added get_config_path
from build import comfy_image
from _utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# Determine config path using the helper function
config_path = get_config_path()

# Load configuration from TOML file
try:
    config = toml.load(config_path)
    modal_settings = config.get('modal_settings', {})
    ALLOW_CONCURRENT_INPUTS = modal_settings.get('allow_concurrent_inputs', 100)
    CONTAINER_IDLE_TIMEOUT = modal_settings.get('container_idle_timeout', 300)
    TIMEOUT = modal_settings.get('timeout', 1200)
    GPU_CONFIG = modal_settings.get('gpu', "A10G")
    logger.info(f"Loaded Modal settings from config: Concurrent={ALLOW_CONCURRENT_INPUTS}, IdleTimeout={CONTAINER_IDLE_TIMEOUT}, Timeout={TIMEOUT}, GPU={GPU_CONFIG}")
except FileNotFoundError:
    logger.error(f"Configuration file not found at {config_path}. Using default Modal settings.")
    ALLOW_CONCURRENT_INPUTS = 10
    CONTAINER_IDLE_TIMEOUT = 300
    TIMEOUT = 900
    GPU_CONFIG = "A10G"
except Exception as e:
    logger.error(f"Error loading configuration from {config_path}: {e}. Using default Modal settings.")
    ALLOW_CONCURRENT_INPUTS = 10
    CONTAINER_IDLE_TIMEOUT = 300
    TIMEOUT = 1800
    GPU_CONFIG = "A10G"


app = modal.App(name="comfy")

# Define necessary volumes directly
cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
comfy_output_vol = modal.Volume.from_name("comfy-output", create_if_missing=True)
comfy_input_vol = modal.Volume.from_name("comfy-input", create_if_missing=True)
comfy_workflows_vol = modal.Volume.from_name("comfy-workflows", create_if_missing=True)
@app.function(
    image=comfy_image,
    allow_concurrent_inputs=ALLOW_CONCURRENT_INPUTS,
    max_containers=1,
    scaledown_window=CONTAINER_IDLE_TIMEOUT,
    timeout=TIMEOUT,
    gpu=GPU_CONFIG,
    volumes={
        f"{Paths.CACHE.base}": cache_vol,
        f"{Paths.INFERENCE.output}": comfy_output_vol,
        f"{Paths.INFERENCE.input}": comfy_input_vol,
    }
)
@modal.web_server(8000, startup_timeout=300)
def ui():
    """Start ComfyUI web interface"""

    comfy_script_path = "/root/comfy/ComfyUI/main.py"
    comfy_command = (
        f"python {comfy_script_path} "
        f"--listen 0.0.0.0 --port 8000 "
        f"--verbose DEBUG"
    )

    logger.info(f"Starting ComfyUI with command: {comfy_command}")
    subprocess.Popen(
        comfy_command,
        shell=True,
    )

    # subprocess.Popen(
    #     "comfy launch -- --listen 0.0.0.0 --port 8000",
    #     shell=True,
    #     env={"PYTHONIOENCODING": "utf-8"}
    # )