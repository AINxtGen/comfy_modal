"""
Build and register ComfyUI image
"""
from typing import List, Optional
from modal import Image, Secret, NotFoundError
from pathlib import Path
import sys
import modal
import subprocess
import logging
import os
import toml
from _utils.constants import Paths, get_config_path
from _utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def get_common_apt_packages() -> List[str]:
    """Get list of common apt packages"""
    logger.debug("Getting common apt packages")
    return [
        "git",
        "wget",
        "curl",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
        "cmake", # Added for WAS Node Suite dependencies
    ]

def get_common_pip_packages() -> List[str]:
    """Get list of common pip packages"""
    logger.debug("Getting common pip packages")
    return [
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers==4.49.0",
        "diffusers==0.32.2",
        "accelerate==1.3.0",
        "xformers",
        "safetensors==0.5.2",
        "huggingface_hub[hf_transfer]>=0.26.2",
        "pillow==11.1.0",
        "tqdm==4.67.1",
        "toml"
    ]

def get_common_env_vars() -> dict:
    """Get common environment variables"""
    logger.debug("Setting up common environment variables")
    return {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": Paths.CACHE.base, # Assumes Paths.CACHE.base is defined correctly
        "HF_HUB_CACHE": Paths.CACHE.base, # Assumes Paths.CACHE.base is defined correctly
        "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "PYTHONIOENCODING": "utf-8"
    }

# Build base image
logger.debug("Building base image for ComfyUI")
base_image = (
    Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(get_common_apt_packages())
    .pip_install(get_common_pip_packages())
    .env(get_common_env_vars())
    )
logger.debug("Base image build configuration complete")

# Removed add_common_files function as it's no longer needed

def _download_inference_models(model_type: Optional[str] = None) -> bool:
    """Download models for inference using config from /root/comfy/config
    
    Args:
        model_type: Optional model type to limit download to specific models
        
    Returns:
        bool: True if download succeeded, False otherwise
    """
    # Config file is now expected at /root/comfy/config.toml
    config_path_in_image = "/root/comfy/config.toml" # Updated path
    # _utils should be importable directly as the working directory is /app

    try:
        # Re-import utils now that they are in the path relative to /app
        # These imports should work because the workdir is /app
        from _utils.constants import Paths # Removed ModelTypes, COMFY_FOLDERS
        from _utils.model_utils import download_and_link_inference_model
        from _utils.logging_config import configure_logging
        configure_logging() # Reconfigure logging inside the function if needed

        logger.debug(f"=== Starting _download_inference_models (model_type={model_type}) ===")
        if not os.path.exists(config_path_in_image):
             logger.error(f"Config file not found inside image at {config_path_in_image}")
             return False

        config = toml.load(config_path_in_image)
        logger.debug(f"Loaded config from {config_path_in_image}")
        
        # Ensure 'models' key exists in the config
        if "models" not in config:
            logger.error("Config file is missing the 'models' section.")
            return False

        for model_spec in config["models"]:
            try:
                # Check ignore flag value (case-insensitive boolean check, default to False)
                should_ignore = str(model_spec.get('ignore', 'false')).lower() == 'true'
                if should_ignore:
                    logger.info(f"Ignoring model spec because ignore=true: {model_spec.get('link', 'Link missing')}")
                    continue

                # Check for mandatory fields
                if 'link' not in model_spec or 'type' not in model_spec:
                    logger.error(f"Skipping model spec due to missing 'link' or 'type': {model_spec}")
                    continue

                model_url = model_spec["link"]
                # Keep original case from config.toml for model type
                spec_model_type = model_spec["type"] 

                # Handle optional fields with logging
                custom_filename = model_spec.get("filename")
                if custom_filename is None:
                    logger.debug(f"No 'filename' provided for {model_url}. Using default name from URL.")

                model_for = model_spec.get("for")
                if model_for is None:
                     logger.debug(f"No 'for' field provided for {model_url}.") # Log if 'for' is missing

                # Skip if model_type filter is active and 'for' doesn't match (or is missing)
                if model_type and model_for != model_type:
                    logger.debug(f"Skipping model {model_url} as it's for '{model_for}' not '{model_type}'")
                    continue
                
                logger.debug(f"Processing model: {model_url}, type: {spec_model_type}, custom filename: {custom_filename}")
                # Ensure download_and_link_inference_model is correctly imported and available
                if not download_and_link_inference_model(model_url, spec_model_type, custom_filename):
                    logger.error(f"Failed to download and link model: {model_url}")
                    # Decide if one failure should stop the whole process or just log and continue
                    # return False # Uncomment to stop on first failure
                else:
                    logger.debug(f"Successfully downloaded and linked model: {model_url}")
                    
            except KeyError as e:
                 logger.error(f"Missing key {e} in model spec: {model_spec}", exc_info=True)
                 continue # Continue with the next model spec
            except Exception as e:
                logger.error(f"Error processing model spec {model_spec}: {str(e)}", exc_info=True) # Log traceback
                continue # Continue with the next model spec

        logger.debug("Finished processing all specified inference models.")
        return True # Return True even if some models failed, adjust if needed

    except FileNotFoundError:
        logger.error(f"Comfy config file not found at {config_path_in_image}")
        return False
    except Exception as e:
        logger.error(f"Error downloading inference models: {str(e)}", exc_info=True) # Log traceback
        return False

def _install_comfy_nodes():
    """Install ComfyUI nodes using config from /app"""
    config_path_in_image = "/root/comfy/config.toml" # Updated path
    # _utils should be importable directly as the working directory is /app

    try:
        # Re-import utils now that they are in the path relative to /app
        from _utils.constants import Paths
        from _utils.logging_config import configure_logging
        configure_logging() # Reconfigure logging inside the function if needed

        logger.debug("=== Starting ComfyUI nodes installation ===")

        if not os.path.exists(config_path_in_image):
             logger.error(f"Config file not found inside image at {config_path_in_image}")
             return "Failed: config.toml not found"

        logger.debug(f"Loading ComfyUI nodes configuration from {config_path_in_image}")
        config = toml.load(config_path_in_image)
        if 'nodes' not in config or 'nodes' not in config['nodes']:
             logger.error("Invalid config format: missing [nodes] section or 'nodes' list")
             return "Failed: Invalid config format"
        nodes = config['nodes']['nodes']

    except Exception as e:
        logger.error(f"Failed to load or parse config file {config_path_in_image}: {e}", exc_info=True)
        return f"Failed: Error loading config {e}"

    logger.debug(f"Found {len(nodes)} nodes to install")
    
    successful_installs = 0
    failed_installs = 0
    
    # Ensure Paths.INFERENCE.base is correctly defined and accessible
    try:
        comfy_base_path = Paths.INFERENCE.base # Use the constant
    except AttributeError:
        logger.error("Paths.INFERENCE.base is not defined in constants.")
        return "Failed: Paths.INFERENCE.base not defined."

    custom_nodes_dir = os.path.join(comfy_base_path, 'ComfyUI', 'custom_nodes')
    os.makedirs(custom_nodes_dir, exist_ok=True) # Ensure the custom_nodes directory exists

    for i, node_url in enumerate(nodes, 1):
        try:
            if '@' in node_url:
                repo_url, commit_hash = node_url.split('@', 1)
            else:
                repo_url = node_url
                commit_hash = None
                
            # Extract node name robustly
            node_name = Path(repo_url).stem # Use pathlib for safer name extraction
            if not node_name: # Handle cases like URLs ending with /
                 node_name = Path(repo_url).parent.stem
            
            logger.debug(f"[{i}/{len(nodes)}] Processing node: {node_name} from {repo_url}" + (f" @{commit_hash}" if commit_hash else " (latest)"))
                
            target_dir = os.path.join(custom_nodes_dir, node_name)

            # Check if directory already exists (e.g., from a previous failed run)
            if os.path.exists(target_dir):
                 logger.warning(f"Target directory {target_dir} already exists. Skipping clone.")
                 # Optionally, you could try to update or just assume it's correct
                 successful_installs += 1 # Count as success if already exists
                 continue

            clone_cmd = f'git clone {repo_url} {target_dir}'
            logger.debug(f"Cloning: {repo_url} to {target_dir}")
            
            # Use subprocess.run with better error handling
            result = subprocess.run(clone_cmd, shell=True, check=False, encoding="utf-8", capture_output=True) # check=False
            if result.returncode != 0:
                 logger.error(f"Error cloning node {node_name}: {result.stderr or result.stdout}")
                 failed_installs += 1
                 continue # Skip to next node on clone failure
            logger.debug(f"Successfully cloned {node_name}")
            
            if commit_hash:
                checkout_cmd = f'git -C {target_dir} checkout {commit_hash}' # Use -C for cwd
                logger.debug(f"Checking out commit: {commit_hash} in {target_dir}")
                result_checkout = subprocess.run(checkout_cmd, shell=True, check=False, encoding="utf-8", capture_output=True)
                if result_checkout.returncode != 0:
                     logger.error(f"Error checking out commit {commit_hash} for {node_name}: {result_checkout.stderr or result_checkout.stdout}")
                     failed_installs += 1
                     # Decide if you want to remove the failed checkout dir or leave it
                     continue # Skip to next node
                logger.debug(f"Successfully checked out commit {commit_hash} for {node_name}")
            
            # Requirements installation is skipped as dependencies are handled globally

            successful_installs += 1
                
        except Exception as e:
             logger.error(f"Unexpected error installing node from URL {node_url}: {str(e)}", exc_info=True)
             failed_installs += 1
             continue
    
    logger.debug(f"=== Completed ComfyUI nodes installation: {successful_installs} successful, {failed_installs} failed ===")
    if failed_installs > 0:
        return f"Warning: Completed node installation with {failed_installs} failures."
    return "Installed all ComfyUI nodes successfully"


def _install_extra_dependencies():
    """Install extra dependencies specified in config.toml"""
    # Determine config path based on environment
    config_path = get_config_path()
    logger.debug("=== Starting extra dependencies installation ===")
    try:
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found at {config_path}, skipping extra dependencies.")
            return "Skipped: config.toml not found"

        config = toml.load(config_path)
        extra_deps = config.get('extra_dependencies', {}).get('packages', [])

        if not extra_deps:
            logger.debug("No extra dependencies specified in config.toml.")
            return "Skipped: No extra dependencies found"

        logger.info(f"Installing extra dependencies: {', '.join(extra_deps)}")
        install_cmd = [sys.executable, "-m", "pip", "install"] + extra_deps
        
        result = subprocess.run(install_cmd, check=False, encoding="utf-8", capture_output=True)
        
        if result.returncode != 0:
            logger.error(f"Error installing extra dependencies: {result.stderr or result.stdout}")
            # Decide if failure should stop the build. Returning a warning for now.
            return f"Warning: Failed to install some extra dependencies: {result.stderr or result.stdout}"
        else:
            logger.debug(f"Successfully installed extra dependencies: {result.stdout}")
            return "Successfully installed extra dependencies"

    except Exception as e:
        logger.error(f"Error during extra dependencies installation: {e}", exc_info=True)
        return f"Failed: Error installing extra dependencies: {e}"


# Build ComfyUI image with model-specific configuration
def build_comfy_image(model_type: Optional[str] = None) -> modal.Image:
    """Build ComfyUI image

    Args:
        model_type: Optional model type to limit model downloads

    Returns:
        modal.Image: Built ComfyUI image
    """
    logger.info(f"Building ComfyUI image")

    # Stage 1: Install all Python dependencies
    logger.info("Stage 1: Installing Python dependencies...")
    dependency_image = (
        base_image
        # Install ComfyUI specific packages in logical groups for better caching
        # Group 1: Core FastAPI/Modal related
        .pip_install([
            "python-slugify==8.0.4",
            "python-dotenv==1.0.1",
            "python-multipart==0.0.9",
            "modal==0.67.43",
            "pydantic==2.7.3",
            "httpx==0.27.0",
            "comfy-cli==1.3.7",
            "toml==0.10.2",
        ])
        # Group 2: Common ML/Data libraries
        .pip_install([
            "numpy",
            "pyyaml",
            "omegaconf",
            "yacs",
            "addict",
            "cachetools",
            "python-dateutil",
            "filelock",
            "importlib_metadata",
            "scipy",
        ])
        # Group 3: Image/Vision related
        .pip_install([
            "opencv-python",
            "imageio-ffmpeg>=0.6.0",
            "matplotlib",
            "mss",
            "albumentations",
            "scikit-image",
            "scikit-learn",
            "colour-science",
            "color-matcher",
            "mediapipe",
            "pixeloe",
            "rembg",
            "transparent-background",
            "qrcode",
            "svglib",
        ])
        # Group 4: Text/NLP related
        .pip_install([
            "ftfy",
            "sentencepiece>=0.2.0",
            "clip_interrogator",
            "lark",
        ])
        # Group 5: Model/Framework specific
        .pip_install([
            "jax>=0.4.28",
            "timm>=1.0.15",
            "einops>=0.8.0",
            "protobuf>=4.25.3,<5", # Adjusted version for mediapipe compatibility
            "onnxruntime-gpu",
            "peft",
            "spandrel",
            "numba",
            "fvcore",
            "trimesh",
            "gguf",
        ])
        # Group 6: Utilities & Platform specific
        .pip_install([
            "rich",
            "rich-argparse",
            "yapf",
            "rich",
            "rich-argparse",
            "yapf",
            "triton>=3.0.0 ; sys_platform == 'linux'",
            "sageattention",
        ])
        # Group 7: Dependencies for specific custom nodes (WAS Suite, Crystools)
        .pip_install([
            "fairscale>=0.4.4",
            "git+https://github.com/WASasquatch/img2texture.git",
            "git+https://github.com/WASasquatch/cstr",
            "gitpython",
            "imageio", # Explicitly add imageio
            "joblib",
            "pilgram",
            "git+https://github.com/WASasquatch/ffmpy.git",
            "deepdiff",
            "pynvml",
            "py-cpuinfo",
            "piexif",
            "olefile",
        ])
    )
    logger.info("Stage 1: Python dependencies installation complete.")

    # Stage 2: Install ComfyUI and create directories
    logger.info("Stage 2: Installing ComfyUI and creating directories...")
    comfy_install_image = (
        dependency_image
        # Install ComfyUI using comfy-cli
        # Assumes Paths.INFERENCE.base is correctly defined for the target location
        .run_commands([
            # f"comfy --skip-prompt install  --nvidia --version 0.3.27",
            f"comfy --skip-prompt install  --nvidia",
            f"rm -rf {Paths.INFERENCE.base}/ComfyUI/output && mkdir -p {Paths.INFERENCE.base}/ComfyUI/output",
            f"rm -rf {Paths.INFERENCE.base}/ComfyUI/input && mkdir -p {Paths.INFERENCE.base}/ComfyUI/input",
            f"rm -rf {Paths.INFERENCE.base}/ComfyUI/user/default/workflows && mkdir -p {Paths.INFERENCE.base}/ComfyUI/user/default/workflows",
            ])
    )
    logger.info("Stage 2: ComfyUI installation complete.")

    # Stage 3: Add local files and install extra dependencies
    logger.info("Stage 3: Adding local files and installing extra dependencies...")
    setup_image = (
        comfy_install_image
        # Add local files AFTER ComfyUI installation
        .add_local_python_source("_utils", "comfy_ui", "build", copy=True)
        .add_local_file("config.toml", remote_path="/root/comfy/config.toml", copy=True)
        .add_local_dir("workflows_example", remote_path="/root/comfy/workflows_example", copy=True)
        .run_function(_install_extra_dependencies)
    )
    logger.info("Stage 3: Local files added and extra dependencies installed.")


    # Stage 4: Install custom nodes
    logger.info("Stage 4: Installing ComfyUI custom nodes...")
    nodes_installed_image = setup_image.run_function(
        _install_comfy_nodes,
        timeout=1800 # Increase timeout for cloning
    )
    logger.info("Stage 4: Custom nodes installation complete.")


    # Stage 5: Download inference models
    # Stage 5: Download inference models
    logger.info(f"Stage 5: Downloading inference models for model_type: {model_type}...")

    # Check if Hugging Face token secret exists and include it conditionally
    hf_secret = None
    try:
        hf_secret = Secret.from_name("huggingface-token")
        logger.info("Hugging Face token secret found, will use it for downloads.")
        secrets_list = [hf_secret]
    except NotFoundError:
        logger.warning("Hugging Face token secret ('huggingface-token') not found. Downloads might fail for gated models.")
        secrets_list = [] # Pass empty list if secret doesn't exist

    final_image = nodes_installed_image.run_function(
        _download_inference_models,
        args=[model_type],
        volumes={
            f"{Paths.CACHE.base}": modal.Volume.from_name("hf-cache", create_if_missing=True)
        },
        secrets=secrets_list, # Use the conditionally defined list
        timeout=3600, # Increase timeout for downloads
        # Consider force_build=True during development/debugging
        # force_build=True
    )

    logger.info("Stage 5: Inference models download complete.")

    logger.info("ComfyUI image build process finished.")
    return final_image


# Pre-build and export the image (can be run directly: python build.py)
if __name__ == "__main__":
    logger.info("Building ComfyUI image directly...")
    # Build without specific model type filtering if run directly
    built_image = build_comfy_image()
    logger.info("ComfyUI image build process finished.")
else:
    # Build for import, potentially filtering models if needed later
    comfy_image = build_comfy_image()
    logger.info("ComfyUI image pre-built and ready for import.")
