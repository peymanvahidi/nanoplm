from pathlib import Path

class DataConfig:
    def __init__(self):
        
        # Only change these variables
        self.val_ratio = 0.1
        self.max_seqs_num = 200
        self.min_seq_len = 20
        self.max_seq_len = 256
        
        # Hardcoded constants (DO NOT CHANGE)
        self.base_dir = Path("output")
        self.raw_dir = self.base_dir / "data/raw"
        self.filter_split_dir = self.base_dir / "data/filter_split"
        self.protx_dataset_dir = self.base_dir / "data/protx_dataset"
        
        self.uniref50_url = "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz"
        self.uniref50_fasta_gz = self.raw_dir / "uniref50.fasta.gz"
        self.uniref50_fasta = self.raw_dir / "uniref50.fasta"
        
        self.filtered_seqs = self.filter_split_dir / "uniref50_filtered.fasta"
        self.train_file = self.filter_split_dir / "train.fasta"
        self.val_file = self.filter_split_dir / "val.fasta"
        self.info_file = self.filter_split_dir / "dataset_info.txt"

        # These will be used if you want to precompute the embeddings
        # If they are gonna be calculated on-the-fly (suggested), these won't be used
        self.protx_train = self.protx_dataset_dir / "train.h5"
        self.protx_val = self.protx_dataset_dir / "val.h5"
        
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
