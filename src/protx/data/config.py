import yaml
from pathlib import Path

class Config:
    def __init__(self, params_file: Path = None):
        """Initialize configuration from a YAML file and set default constants."""
        # Load parameters from YAML
        if params_file is None:
            yaml_path = Path(__file__).parent / "params.yaml"
        else:
            yaml_path = params_file
            
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        
        # Parameters from YAML
        self.val_ratio = data["val_ratio"]
        self.max_seqs_num = data["max_seqs_num"]
        self.min_seq_len = data["min_seq_len"]
        self.max_seq_len = data["max_seq_len"]
        self.batch_size = data["batch_size"]
        
        # Hardcoded constants
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
