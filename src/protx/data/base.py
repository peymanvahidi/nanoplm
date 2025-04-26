from pathlib import Path
from ..utils.logger import logger

class BaseProcessor:
    """Base class with common functionality for data processing."""
    
    def __init__(self):
        self.total_seqs = 0
        self.train_count = 0
        self.val_count = 0
    
    def create_dirs(self, path):
        dir_path = Path(path)
        if dir_path.suffix:  # If path has file extension, get the parent directory
            dir_path = dir_path.parent
            
        if dir_path.exists():
            logger.info(f"Directory already exists: {dir_path}")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
