from pathlib import Path
import yaml
import os
from typing import Dict, Any, Optional, Union

class DataConfig:
    def __init__(self, config: Optional[Union[Dict, str]] = None):
        # Load params from provided config_dict or params.yaml
        if isinstance(config, dict):
            self._params = config
        elif isinstance(config, str):
            if os.path.exists(config):
                self._params = {}
                with open(Path(config), "r") as f:
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
        
        self.base_dir = self._get_param("data_dirs.base_dir")
        self.raw_dir = self._get_param("data_dirs.raw_dir")
        
        self.val_ratio = self._get_param("data_params.val_ratio")
        self.max_seqs_num = self._get_param("data_params.max_seqs_num")
        self.min_seq_len = self._get_param("data_params.min_seq_len")
        self.max_seq_len = self._get_param("data_params.max_seq_len")
        self.filter_skip_n = self._get_param("data_params.filter_skip_n")
        self.shuffle = self._get_param("data_params.shuffle")
        self.shuffle_seed = self._get_param("data_params.shuffle_seed")
        # This is batch size for embedding calculation
        self.embed_calc_batch_size = self._get_param("data_params.embed_calc_batch_size")
        
        self.filter_split_dir = self._get_param("data_dirs.filter_split_dir")
        self.protx_dataset_dir = self._get_param("data_dirs.protx_dataset_dir")
        
        self.uniref50_url = self._get_param("data_dirs.uniref50_url")
        self.uniref50_fasta_gz = self._get_param("data_dirs.uniref50_fasta_gz")
        self.uniref50_fasta = self._get_param("data_dirs.uniref50_fasta")
        self.shuffled_fasta_file = self._get_param("data_dirs.shuffled_fasta_file")
        
        self.filtered_seqs = self._get_param("data_dirs.uniref50_filtered_fasta")
        self.train_file = self._get_param("data_dirs.train_fasta")
        self.val_file = self._get_param("data_dirs.val_fasta")
        self.info_file = self._get_param("data_dirs.info_file")

        # These will be used if you want to precompute the embeddings
        # If they are gonna be calculated on-the-fly (suggested), these won't be used
        self.protx_train_prefix = self._get_param("data_dirs.protx_train_prefix")
        self.protx_val_prefix = self._get_param("data_dirs.protx_val_prefix")
        
        self.validate()
    
    def override(self, kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Config does not have attribute '{key}'")
        
        self.validate()
        
    def validate(self):
        self._validate_val_ratio()
        self._validate_seq_len_params()
        self._validate_max_seqs_num()
        self._validate_path_url()
        
    def _validate_val_ratio(self):
        if not 0 <= self.val_ratio <= 1:
            raise ValueError(f"val_ratio must be between 0 and 1, got {self.val_ratio}")
            
    def _validate_seq_len_params(self):
        for param_name in ["min_seq_len", "max_seq_len"]:
            value = getattr(self, param_name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{param_name} must be a positive integer, got {value}")
        
        if self.min_seq_len > self.max_seq_len:
            raise ValueError(
                f"min_seq_len ({self.min_seq_len}) cannot be greater than "
                f"max_seq_len ({self.max_seq_len})"
            )
            
    def _validate_max_seqs_num(self):
        if not isinstance(self.max_seqs_num, int) or self.max_seqs_num <= 0:
            raise ValueError(f"max_seqs_num must be a positive integer, got {self.max_seqs_num}")
            
    def _validate_path_url(self):
        for attr, value in vars(self).items():
            if isinstance(value, (str, Path)) and not str(value):
                raise ValueError(f"{attr} must be a non-empty string or Path, got {value}")

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

if __name__ == "__main__":
    dc = DataConfig()
    print(dc.uniref50_url)