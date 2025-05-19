import os
import yaml
from pathlib import Path

class DistillConfig:
    def __init__(self):
        self._params = {}
        if os.path.exists("params.yaml"):
            with open("params.yaml", "r") as f:
                self._params = yaml.safe_load(f)
        
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
