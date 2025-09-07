import os
import time
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import wandb

from nanoplm.utils import logger


class TrainingSessionManager:
    """
    Handles checkpoint resumption and training session setup.
    
    Separates the resume logic from the main training pipeline to keep
    the code organized and testable.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        wandb_dir: str,
        project_name: str,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.wandb_dir = wandb_dir
        self.project_name = project_name
    
    def setup_session(self, training_config: Optional[Dict[str, Any]] = None) -> Tuple[str, Path, bool]:
        """
        Set up training session (new or resumed).
        
        Args:
            training_config: Configuration dictionary to save for new sessions
        
        Returns:
            Tuple of (run_name, output_dir, is_resuming)
        """
        is_resuming = self._is_valid_checkpoint_dir(self.checkpoint_dir)
        
        if is_resuming:
            logger.info(f"Resuming training from checkpoint: {self.checkpoint_dir}")
            run_name = self._get_original_run_name(self.checkpoint_dir)
            output_dir = Path(self.checkpoint_dir).parent
        else:
            logger.info("Starting new training session")
            timestamp = int(time.time())
            run_name = f"run-{timestamp}"
            output_dir = Path(self.wandb_dir) / run_name
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_run_name(output_dir, run_name)
            
            # Save training configuration for future resuming
            if training_config:
                self._save_training_config(output_dir, training_config)
        
        return run_name, output_dir, is_resuming
    
    def setup_wandb_config(
        self,
        run_name: str,
        training_args: Any,
        is_resuming: bool,
    ) -> Dict[str, Any]:
        """
        Set up wandb configuration for new or resumed runs.
        
        Args:
            run_name: Name of the training run
            training_args: Training arguments object
            is_resuming: Whether this is a resumed run
            
        Returns:
            Dictionary with wandb configuration
        """
        wandb_config = {
            "project": self.project_name,
            "name": run_name,
            "config": training_args.to_dict(),
            "settings": wandb.Settings(start_method="fork"),
            "id": run_name  # Use run name as ID for consistency
        }
        
        if is_resuming:
            wandb_config["resume"] = "allow"
        
        return wandb_config
    
    @staticmethod
    def _is_valid_checkpoint_dir(checkpoint_dir: str) -> bool:
        """Check if the checkpoint directory is valid and contains required files."""
        if not checkpoint_dir or not os.path.exists(checkpoint_dir):
            if checkpoint_dir:
                logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            return False
            
        checkpoint_path = Path(checkpoint_dir)
        logger.info(f"Checking for valid checkpoint in: {checkpoint_path}")

        required_files = [
            "trainer_state.json",
            "training_args.bin",
        ]
        
        # Check for model files (either model.safetensors or pytorch_model.bin)
        model_files = ["model.safetensors", "pytorch_model.bin"]
        has_model = any((checkpoint_path / f).exists() for f in model_files)
        if not has_model:
            logger.warning(f"Model file (e.g., model.safetensors) not found in {checkpoint_path}")

        # Check for required files
        has_required = True
        for f in required_files:
            if not (checkpoint_path / f).exists():
                logger.warning(f"Required file not found: {f} in {checkpoint_path}")
                has_required = False
        
        is_valid = has_model and has_required
        if is_valid:
            logger.info(f"Valid checkpoint found at {checkpoint_path}")
        else:
            logger.warning(f"Invalid or incomplete checkpoint at {checkpoint_path}")

        return is_valid

    @staticmethod
    def _get_original_run_name(checkpoint_dir: str) -> str:
        """Get the original run name from the checkpoint directory."""
        checkpoint_path = Path(checkpoint_dir)
        run_name_file = checkpoint_path.parent / "run_name.txt"
        
        if run_name_file.exists():
            return run_name_file.read_text().strip()
        
        # Fallback: try to infer from directory structure
        parent_dir = checkpoint_path.parent
        if parent_dir.name.startswith("run-"):
            return parent_dir.name
            
        # Last resort: generate a new run name
        logger.warning("Could not find original run name, generating new one")
        return f"resumed-{int(time.time())}"

    @staticmethod
    def _save_run_name(output_dir: Path, run_name: str):
        """Save the run name to a file for future resuming."""
        with open(output_dir / "run_name.txt", "w") as f:
            f.write(run_name)
    
    @staticmethod
    def _save_training_config(output_dir: Path, config: Dict[str, Any]):
        """Save the training configuration to a file for future resuming."""
        config_file = output_dir / "training_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved training configuration to {config_file}")
    
    @staticmethod
    def load_training_config(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
        """
        Load training configuration from a checkpoint directory.
        
        Args:
            checkpoint_dir: Path to the checkpoint directory
            
        Returns:
            Dictionary with training configuration or None if not found
        """
        if not TrainingSessionManager._is_valid_checkpoint_dir(checkpoint_dir):
            return None
        
        checkpoint_path = Path(checkpoint_dir)
        config_file = checkpoint_path.parent / "training_config.json"
        
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                logger.info(f"Loaded training configuration from {config_file}")
                return config
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load training config from {config_file}: {e}")
                return None
        else:
            logger.warning(f"Training configuration file not found: {config_file}")
            return None 