from pathlib import Path

from .downloader import Downloader
from .extractor import Extractor
from .filter_splitor import FilterSplitor
from .dataset import ProtXDataProcessor

from ..models.teacher import ProtT5
from ..config import DataConfig

from ..utils import logger, log_stage, get_device

class DataPipeline:
    
    def __init__(
        self,
        config: DataConfig,
        **kwargs
    ):
        """
        Initialize the pipeline with a config object and optional overrides.
        
        Args:
            config: Instance of Config class containing pipeline parameters.
            **kwargs: Optional keyword arguments to override config attributes.
        """
        self.config = config
        
        if kwargs:
            self.config.override(kwargs)
        
        
    def download(self):
        downloader = Downloader(
            url=self.config.uniref50_url,
            output_file=self.config.uniref50_fasta_gz
        )
        try:
            log_stage("DOWNLOAD")
            downloader.download()
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise
    
    def extract(self):
        extractor = Extractor(
            input_file=self.config.uniref50_fasta_gz,
            output_file=self.config.uniref50_fasta
        )
        try:
            log_stage("EXTRACT")
            extractor.extract()
        except Exception as e:
            logger.error(f"Extract failed: {e}")
            raise
    
    def filter_split(self):
        filter_splitor = FilterSplitor(
            input_file=self.config.uniref50_fasta,
            output_dir=self.config.filter_split_dir,
            min_seq_len=self.config.min_seq_len,
            max_seq_len=self.config.max_seq_len,
            max_seqs_num=self.config.max_seqs_num,
            val_ratio=self.config.val_ratio,
            info_file=self.config.info_file,
        )
        try:
            log_stage("FILTER & SPLIT")
            filter_splitor.filter(
                output_file=self.config.filtered_seqs
            )
            filter_splitor.split(
                    train_file=self.config.train_file,
                    val_file=self.config.val_file
            )
        except Exception as e:
            logger.error(f"Filter & Split failed: {e}")
            raise
    
    def save_protx_train_dataset(self):

        protx_train_data = ProtXDataProcessor(
            data_path=self.config.train_file,
            teacher_model=ProtT5(),
            max_seq_len=self.config.max_seq_len,
            seqs_num_per_file=self.config.seqs_num_per_file,
            batch_size=self.config.embed_calc_batch_size,
            device=get_device()
        )

        try:
            log_stage("ProtX Training Data Gen")
            train_files = protx_train_data.process_dataset(
                save_path=Path(self.config.protx_train_prefix)
            )
            logger.info(f"Created {len(train_files)} training dataset files")
            return train_files
        except Exception as e:
            logger.error(f"Save ProtX Train Dataset failed: {e}")
            raise
    

    def save_protx_val_dataset(self):

        protx_val_data = ProtXDataProcessor(
            data_path=self.config.val_file,
            teacher_model=ProtT5(),
            max_seq_len=self.config.max_seq_len,
            seqs_num_per_file=self.config.seqs_num_per_file,
            batch_size=self.config.embed_calc_batch_size,
            device=get_device()
        )
        
        try:
            log_stage("ProtX Validation Data Gen")
            val_files = protx_val_data.process_dataset(
                save_path=Path(self.config.protx_val_prefix)
            )
            logger.info(f"Created {len(val_files)} validation dataset files")
            return val_files
        except Exception as e:
            logger.error(f"Save ProtX Val Dataset failed: {e}")
            raise
    
    def run_all(self, save_protx_dataset=False):
        self.download()
        self.extract()
        self.filter_split()
        if save_protx_dataset:
            self.save_protx_train_dataset()
            self.save_protx_val_dataset()
        logger.info("Complete pipeline execution finished!")
