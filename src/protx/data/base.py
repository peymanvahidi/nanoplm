import yaml
from pathlib import Path
from ..utils.logger import logger

class BaseProcessor:
    """Base class with common functionality for data processing."""
    
    def __init__(
        self,
        pipeline_output_dir: Path = Path("pipeline_output"),
        params_file="src/protx/data/params.yaml"
    ):
        # Set up paths
        self.pipeline_output_dir = pipeline_output_dir
        self.raw_data_dir = self.pipeline_output_dir / "data/raw"
        self.processed_data_dir = self.pipeline_output_dir / "data/processed"
        self.uniref50_url = "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz"
        
        self.uniref50_fasta_gz = self.raw_data_dir / "uniref50.fasta.gz"
        self.uniref50_fasta = self.raw_data_dir / "uniref50.fasta"
        self.filtered_seqs = self.processed_data_dir / "uniref50_filtered.fasta"
        self.train_file = self.processed_data_dir / "train.fasta"
        self.val_file = self.processed_data_dir / "val.fasta"
        self.info_file = self.processed_data_dir / "dataset_info.txt"
        
        # Load parameters
        self.params = self._load_params(params_file)
        self.val_ratio = self.params['val_ratio']
        self.max_seqs_num = self.params['max_seqs_num']
        self.min_seq_len = self.params['min_seq_len']
        self.max_seq_len = self.params['max_seq_len']
        self.batch_size = self.params['batch_size']
        # Initialize counters
        self.total_seqs = 0
        self.train_count = 0
        self.val_count = 0
    
    def _load_params(self, params_file):
        """Load processing parameters from params.yaml."""
        with open(params_file, 'r') as f:
            params = yaml.safe_load(f)
        return params
    
    def create_dirs(self, dir_name):
        """Create necessary directories if they don't exist."""
        dir_path = Path(dir_name)
        if dir_path.exists():
            logger.info(f"Directory already exists: {dir_path}")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
