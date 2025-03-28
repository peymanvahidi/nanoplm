import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseProcessor:
    """Base class with common functionality for data processing."""
    
    def __init__(self, params_file="src/data/params.yaml"):
        """
        Initialize the processor with paths and parameters.
        
        Args:
            params_file (str): Path to the parameters file
        """
        # Set up paths
        self.raw_data_dir = Path("data/raw")
        self.processed_data_dir = Path("data/processed")
        self.uniref50_url = "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz"
        self.uniref50_fasta_gz = self.raw_data_dir / "uniref50.fasta.gz"
        self.uniref50_fasta = self.raw_data_dir / "uniref50.fasta"
        self.processed_sequences = self.processed_data_dir / "uniref50_processed.fasta"
        self.train_file = self.processed_data_dir / "train.fasta"
        self.val_file = self.processed_data_dir / "val.fasta"
        self.info_file = self.processed_data_dir / "dataset_info.txt"
        
        # Load parameters
        self.params = self._load_params(params_file)
        self.val_ratio = self.params['val_ratio']
        self.max_seqs_number = self.params['max_seqs_number']
        self.min_sequence_length = self.params['min_sequence_length']
        self.max_sequence_length = self.params['max_sequence_length']
        
        # Initialize counters
        self.total_sequences = 0
        self.train_count = 0
        self.val_count = 0
    
    def _load_params(self, params_file):
        """Load processing parameters from params.yaml."""
        with open(params_file, 'r') as f:
            params = yaml.safe_load(f)
        return params
    
    def create_dirs(self):
        """Create necessary directories if they don't exist."""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directories: {self.raw_data_dir} and {self.processed_data_dir}") 