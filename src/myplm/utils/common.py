import os
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Union

from myplm.utils.logger import logger

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
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def create_dirs(path: Union[str, Path]):
    dir_path = Path(path)
    if dir_path.suffix:  # If path has file extension, get the parent directory
        dir_path = dir_path.parent
        
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
