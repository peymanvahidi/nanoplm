import os
import yaml
import torch
from typing import Dict, Any

def read_yaml(file_path: str) -> Dict[str, Any]:
    """
    Read a YAML file and return its contents as a dictionary.
    
    Args:
        file_path: Path to the YAML file to read
        
    Returns:
        Dictionary containing the YAML file contents
        
    Raises:
        FileNotFoundError: If the file does not exist
        yaml.YAMLError: If the file is not valid YAML
    """
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
    elif torch.backends.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
