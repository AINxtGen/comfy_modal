"""
Utility functions for downloading and managing models for comfy_image build.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from huggingface_hub import snapshot_download, hf_hub_download
import logging
import shutil
import toml

# Import local constants and logging config
try:
    from .constants import Paths # Removed ModelTypes, COMFY_FOLDERS
    from .logging_config import configure_logging
except ImportError:
    # Fallback for direct execution or testing
    from constants import Paths
    from logging_config import configure_logging


# Configure logging but only once
configure_logging()
logger = logging.getLogger(__name__)

def is_file_url(url: str) -> bool:
    """Check if a URL points to a file"""
    if not url or not isinstance(url, str):
        return False
    model_file_extensions = [
        ".safetensors", ".pt", ".bin", ".ckpt", ".pth", ".onnx",
        ".h5", ".pb", ".tflite", ".pkl", ".model", ".weights"
    ]
    return any(url.endswith(ext) for ext in model_file_extensions) or "/blob/" in url or "/resolve/" in url

def parse_hf_url(url: str) -> Dict:
    """Parse HuggingFace URL to get repo info"""
    try:
        if not url or not isinstance(url, str):
            raise ValueError(f"Invalid URL: {url}")
        is_file = is_file_url(url)
        parts = url.split("/")
        if len(parts) < 5:
            raise ValueError(f"Invalid HuggingFace URL format: {url}")
        repo_id = f"{parts[3]}/{parts[4]}"
        branch = "main"
        file_path = None
        filename = None
        if is_file:
            if "/blob/" in url or "/resolve/" in url:
                try:
                    indicator = "blob" if "/blob/" in url else "resolve"
                    indicator_index = parts.index(indicator)
                    if len(parts) > indicator_index + 2:
                        branch = parts[indicator_index + 1]
                        file_path = "/".join(parts[indicator_index + 2:])
                        filename = parts[-1]
                except ValueError:
                    logger.warning(f"Could not parse branch and file path from URL: {url}")
            else:
                path_parts = url.split("/")[5:]
                file_path = "/".join(path_parts)
                filename = path_parts[-1]
        return {
            "repo_id": repo_id,
            "is_file": is_file,
            "branch": branch,
            "file_path": file_path,
            "filename": filename
        }
    except Exception as e:
        logger.error(f"Error parsing URL {url}: {str(e)}")
        raise

def create_symlink(source: str, target: str, is_dir: bool = False):
    """Create symlink with proper handling"""
    source = os.path.abspath(os.path.expanduser(source))
    target = os.path.abspath(os.path.expanduser(target))
    if os.path.islink(target) and os.readlink(target) == source:
        logger.debug(f"Symlink already exists and points to correct target: {target} -> {source}")
        return
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        if target_path.is_dir() and not target_path.is_symlink():
            # Be cautious removing directories, maybe log warning instead?
            logger.warning(f"Target path exists and is a directory, not removing: {target}")
            return # Avoid removing potentially important directories
        else:
            target_path.unlink()
    logger.debug(f"Attempting to create symlink: {target} -> {source} (is_dir={is_dir})") # Added debug log
    os.symlink(source, target, target_is_directory=is_dir)
    logger.debug(f"Created symlink: {target} -> {source}")

def get_cache_path(repo_id: str) -> str:
    """Get HuggingFace cache path for repo"""
    repo_parts = repo_id.split("/")
    cache_folder = f"models--{repo_parts[0]}--{repo_parts[1]}"
    # Use the constant defined in the local constants.py
    return os.path.join(Paths.CACHE.base, cache_folder)

def download_hf_model(url_info: Dict, cache_path: str) -> Tuple[str, bool]:
    """Download model from HuggingFace and return path"""
    is_file = url_info["is_file"]
    repo_id = url_info["repo_id"]
    hf_token = os.environ.get("HF_TOKEN") # Get token from environment

    if is_file and url_info["file_path"]:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=url_info["file_path"],
            cache_dir=Paths.CACHE.base, # Use constant
            token=hf_token,
            revision=url_info["branch"]
        )
        return file_path, True
    else:
        try:
            # Ensure cache_path exists before download if needed by snapshot_download
            os.makedirs(cache_path, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=cache_path, # Download directly into the specific repo cache path
                token=hf_token,
                revision=url_info["branch"],
                local_dir_use_symlinks=False # Avoid symlinks within cache if problematic
            )
            # The downloaded content should be directly in cache_path
            # Verify snapshot directory structure if needed
            snapshots_dir = os.path.join(cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                 commit_hashes = sorted([d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))])
                 if commit_hashes:
                     latest_commit_dir = os.path.join(snapshots_dir, commit_hashes[-1])
                     logger.debug(f"Using latest snapshot from cache: {latest_commit_dir}")
                     return latest_commit_dir, False # Return path to snapshot dir
            logger.warning(f"Could not find snapshot directory structure in {cache_path}, returning base cache path.")
            return cache_path, False # Return the base cache path for the repo

        except Exception as e:
            logger.error(f"Error downloading repo {repo_id}: {str(e)}")
            raise

def download_and_link_inference_model(
    model_url: str,
    model_type: str, # Can be a single type or multiple types separated by | or ,
    custom_filename: Optional[str] = None
) -> bool:
    """
    Download inference model and create appropriate symlinks in ComfyUI structure
    for one or multiple specified model types.
    """
    try:
        # Parse URL and get repo info
        url_info = parse_hf_url(model_url)
        repo_id = url_info["repo_id"]
        owner, repo_name = repo_id.split("/")

        # Download model (only once)
        logger.debug(f"Downloading model {repo_id} from {model_url} to cache...")
        downloaded_path, is_file = download_hf_model(
            url_info,
            get_cache_path(repo_id) # Use helper to get cache path
        )
        logger.debug(f"Model downloaded to: {downloaded_path}, is_file: {is_file}")

        # Parse model types (handle separators and whitespace)
        types_raw = model_type.replace(',', '|').split('|')
        model_types = [t.strip() for t in types_raw if t.strip()]
        if not model_types:
            logger.error(f"No valid model types found in '{model_type}' for {model_url}")
            return False
        logger.debug(f"Parsed model types for {repo_id}: {model_types}")

        # Create symlink(s) for each specified type
        for current_type in model_types:
            logger.debug(f"Processing type: {current_type} for model {repo_id}")
            # Determine target directory within ComfyUI models folder for the current type
            target_base_dir = os.path.join(
                Paths.INFERENCE.base, # Use constant for Comfy base
                "ComfyUI", # Standard ComfyUI subfolder
                "models",
                current_type # Use the current type string
            )
            # Note: target_base_dir might not exist yet if current_type is new.

            if is_file:
                # Determine the final filename, potentially including subdirectories
                filename = custom_filename or url_info["filename"] or os.path.basename(downloaded_path)
                # Target path including potential subdirectories specified in filename
                target_path = os.path.join(target_base_dir, filename)

                # Ensure the target directory exists (including subdirs from filename)
                target_dir = os.path.dirname(target_path)
                os.makedirs(target_dir, exist_ok=True)
                logger.debug(f"Ensured target directory exists for type '{current_type}': {target_dir}")

                logger.debug(f"Creating file symlink for type '{current_type}': {target_path} -> {downloaded_path}")
                create_symlink(downloaded_path, target_path, is_dir=False)
            else:
                # Target path for directory symlink (using repo_name as the link name)
                target_path = os.path.join(target_base_dir, repo_name)

                # Ensure the target base directory exists
                os.makedirs(target_base_dir, exist_ok=True)
                logger.debug(f"Ensured target directory exists for type '{current_type}': {target_base_dir}")

                logger.debug(f"Creating directory symlink for type '{current_type}': {target_path} -> {downloaded_path}")
                create_symlink(downloaded_path, target_path, is_dir=True)

        return True

    except Exception as e:
        logger.error(f"Error downloading/linking inference model {model_url} (types: {model_type}): {str(e)}", exc_info=True)
        return False
