import os
import yaml
import torch
import inspect
import sys
import subprocess
from IPython import get_ipython
from pathlib import Path
from typing import Dict, Any, Union

from nanoplm.utils.logger import logger

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

def is_flash_attention_available():
    if not torch.cuda.is_available():
        return False

    # small dummy tensors just to test
    q = k = v = torch.randn(1, 1, 1, 1, device="cuda", dtype=torch.float16)
    try:
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return True
    except RuntimeError:
        return False


def read_yaml(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as file:
        try:
            yaml_content = yaml.safe_load(file)
            return yaml_content
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {e}")

def get_device():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    return device

def create_dirs(path: Union[str, Path]):
    dir_path = Path(path)
    if dir_path.suffix:  # If path has file extension, get the parent directory
        dir_path = dir_path.parent
        
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def get_caller_dir() -> Path:
    """
    Get the directory of the script that's using this function.
    
    For .py files, traverses the call stack to find the first frame outside
    the pymmseqs package, presumed to be the user's code. For .ipynb files
    (Jupyter notebooks), returns the current working directory, which is
    typically the directory containing the notebook unless changed.
    
    Returns:
        Path: Absolute path to the directory containing the calling script
    """
    # Check if running in a Jupyter notebook
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook detected; return current working directory
            return Path(os.getcwd())
    except NameError:
        # Not in an IPython environment; proceed with stack traversal
        pass
    
    # Get the full call stack
    frame = inspect.currentframe()
    try:
        # Get package path to identify frames within nanoplm
        nanoplm_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Start from the immediate caller
        caller_frame = frame.f_back
        
        # Traverse up the stack until we find a frame outside nanoplm
        while caller_frame:
            caller_file = caller_frame.f_code.co_filename
            
            # If the frame is not from within nanoplm package or standard library
            if (not caller_file.startswith(nanoplm_path) and 
                not caller_file.startswith(sys.prefix) and
                not caller_file == '<string>'):  # Ignore REPL or eval frames
                # Found a frame outside nanoplm - likely the user's code
                return Path(os.path.dirname(os.path.abspath(caller_file)))
            
            # Move up to the next frame
            caller_frame = caller_frame.f_back
        
        # If no suitable frame is found, return current working directory
        return Path(os.getcwd())
    finally:
        # Clean up the frame to prevent memory leaks
        del frame

def inside_git_repo(work_dir: Path) -> bool:
    """
    Check if inside a Git work tree.
    
    Args:
        work_dir: Directory path to check
        
    Returns:
        bool: True if inside a Git repository, False otherwise
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(work_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

def is_git_subdir(work_dir: Path) -> bool:
    """
    Check if work_dir is a subdirectory of a Git repository (not at root).
    
    Args:
        work_dir: Directory path to check
        
    Returns:
        bool: True if work_dir is a subdirectory of a Git repo, False otherwise
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if result.returncode == 0:
            git_root = Path(result.stdout.strip()).resolve()
            return git_root != work_dir.resolve()
        return False
    except FileNotFoundError:
        return False