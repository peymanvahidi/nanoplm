from .downloader import Downloader
from .extractor import Extractor
from .filter_splitor import FilterSplitor
from .dataset import ProtXDataProcessor

from ..utils.logger import logger, log_stage

class DataPipeline:
    
    def __init__(self, config, **kwargs):
        """
        Initialize the pipeline with a config object and optional overrides.
        
        Args:
            config: Instance of Config class containing pipeline parameters.
            **kwargs: Optional keyword arguments to override config attributes.
        """
        self.config = config
        
        # Apply any runtime overrides from kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Config does not have attribute '{key}'")
        
        self.downloader = Downloader(
            url=self.config.uniref50_url,
            output_file=self.config.uniref50_fasta_gz
        )
        self.extractor = Extractor(
            input_file=self.config.uniref50_fasta_gz,
            output_file=self.config.uniref50_fasta
        )
        self.filter_splitor = FilterSplitor(
            input_file=self.config.uniref50_fasta,
            output_dir=self.config.filter_split_dir,
            min_seq_len=self.config.min_seq_len,
            max_seq_len=self.config.max_seq_len,
            max_seqs_num=self.config.max_seqs_num,
            val_ratio=self.config.val_ratio,
            info_file=self.config.info_file,
        )

        # self.protx_data_processor = ProtXDataProcessor(pipeline_output_dir=self.config.filter_split_dir)
    
    def download(self):
        log_stage("DOWNLOAD")
        self.downloader.download()
    
    def extract(self):
        log_stage("EXTRACT")
        self.extractor.extract()
    
    def filter_split(self):
        log_stage("FILTER & SPLIT")
        self.filter_splitor.filter(
            output_file=self.config.filtered_seqs
        )
        self.filter_splitor.split(
            train_file=self.config.train_file,
            val_file=self.config.val_file
        )
    
    # def generate_protx_training_data(self):
    #     log_stage("ProtX Training Data Gen")
    #     self.protx_data_processor.process_dataset()
    
    def run_all(self):
        """Run the complete pipeline."""
        self.download()
        self.extract()
        self.filter_split()
        # self.generate_protx_training_data()
        logger.info("Complete pipeline execution finished!")
