import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

class DistillConfig:
    def __init__(self, config: Optional[Union[Dict, str]] = None):
        # Load params from provided config_dict or params.yaml
        if isinstance(config, dict):
            self._params = config
        elif isinstance(config, str):
            if os.path.exists(config):
                with open(config, "r") as f:
                    self._params = yaml.safe_load(f)
            else:
                raise FileNotFoundError(f"Config file not found: {config}")
        else:
            self._params = {}
            if os.path.exists("params.yaml"):
                with open("params.yaml", "r") as f:
                    self._params = yaml.safe_load(f)
            else:
                raise FileNotFoundError("No config provided and default 'params.yaml' not found")
        
        self.on_the_fly = bool(self._get_param("distill_params.on_the_fly"))
        self.base_dir = Path(self._get_param("data_dirs.base_dir"))
        # self.trained_protx_dir = self._get_param("data_dirs.protx_dataset_dir")
        self.checkpoint_dir = Path(self._get_param("data_dirs.wandb_checkpoints_dir"))
        self.plots_dir = Path(self._get_param("data_dirs.plots_dir"))

        self.num_epochs = int(self._get_param("distill_params.num_epochs"))
        self.batch_size = int(self._get_param("distill_params.batch_size"))

        # Scheduler
        ## warmup
        self.warmup_steps = int(self._get_param("distill_params.warmup_steps"))
        self.start_lr = float(self._get_param("distill_params.start_lr"))

        ## cosine annealing
        self.min_lr = float(self._get_param("distill_params.min_lr"))
        self.max_lr = float(self._get_param("distill_params.max_lr"))
        self.T_0 = int(self._get_param("distill_params.T_0"))
        self.T_mult = int(self._get_param("distill_params.T_mult"))
        self.gamma = float(self._get_param("distill_params.gamma"))
        self.max_cycle_length = self._get_param("distill_params.max_cycle_length")

        # student
        self.student_embed_dim = int(self._get_param("distill_params.student_embed_dim"))
        self.student_num_layers = int(self._get_param("distill_params.student_num_layers"))
        self.student_num_heads = int(self._get_param("distill_params.student_num_heads"))

        self.dataloader_num_workers = int(self._get_param("distill_params.dataloader_num_workers", 0))

    def override(self, kwargs):
        """Override configuration attributes with provided kwargs."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Config does not have attribute '{key}'")

    def _get_param(self, key):
        if "." in key:
            parts = key.split(".")
            current = self._params
            for part in parts:
                if part not in current:
                    raise KeyError(f"Key '{key}' not found in params")
                current = current[part]
            return current
        return self._params[key]
