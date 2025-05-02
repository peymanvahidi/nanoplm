from pathlib import Path

class DataConfig:
    def __init__(self):
        
        # Only change these variables
        self.val_ratio = 0.1
        self.max_seqs_num = 5000
        self.min_seq_len = 20
        self.max_seq_len = 512
        
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
