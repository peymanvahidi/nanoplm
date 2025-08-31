from pathlib import Path

from myplm.data.downloader import Downloader
from myplm.data.extractor import Extractor
from myplm.data.shuffler import FastaShuffler
from myplm.data.filter_splitor import FilterSplitor
from myplm.data.dataset import ProtXDataProcessor

from myplm.models.teacher import ProtT5
from myplm.config import DataConfig

from myplm.utils import logger, log_stage, get_device

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
        """Download data - same logic as CLI download-data command"""
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
        """Extract data - same logic as CLI extract-data command"""
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
    
    def shuffle_fasta(self):
        """Shuffle the main FASTA file."""
        shuffler = FastaShuffler(
            input_file=self.config.uniref50_fasta,
            output_file=self.config.shuffled_fasta_file,
            seed=self.config.shuffle_seed
        )
        try:
            log_stage("SHUFFLE FASTA")
            shuffler.shuffle()
        except Exception as e:
            logger.error(f"FASTA shuffling failed: {e}")
            raise

    def filter_split(self):
        """Filter and split data - same logic as CLI filter-split-data command"""
        filter_splitor = FilterSplitor(
            input_file=self.config.shuffled_fasta_file,
            output_file=self.config.filtered_seqs,
            min_seq_len=self.config.min_seq_len,
            max_seq_len=self.config.max_seq_len,
            max_seqs_num=self.config.max_seqs_num,
            val_ratio=self.config.val_ratio,
            info_file=self.config.info_file,
            skip_n=self.config.filter_skip_n
        )
        try:
            log_stage("FILTER & SPLIT")
            filter_splitor.filter()
            filter_splitor.split(
                    train_file=self.config.train_file,
                    val_file=self.config.val_file
            )
        except Exception as e:
            logger.error(f"Filter & Split failed: {e}")
            raise

    def save_protx_train_dataset(self, n_files: int = 1):
        """Save training dataset - same logic as CLI save-protx-dataset command"""
        protx_train_data = ProtXDataProcessor(
            data_path=self.config.train_file,
            teacher_model=ProtT5(),
            max_seq_len=self.config.max_seq_len,
            batch_size=self.config.embed_calc_batch_size,
            device=get_device(),
            skip_n=0,
            n_files=n_files
        )

        try:
            log_stage("ProtX Training Data Gen")
            result = protx_train_data.process_dataset(
                save_path=Path(self.config.protx_train_prefix)
            )
            
            if n_files > 1:
                train_files = result
                logger.info(f"Created {len(train_files)} training dataset shard files")
            else:
                train_files = [result] if result else []
                logger.info(f"Created training dataset file: {result}")
            
            return train_files
        except Exception as e:
            logger.error(f"Save ProtX Train Dataset failed: {e}")
            raise
    

    def save_protx_val_dataset(self, n_files: int = 1):
        """Save validation dataset - same logic as CLI save-protx-dataset command"""
        protx_val_data = ProtXDataProcessor(
            data_path=self.config.val_file,
            teacher_model=ProtT5(),
            max_seq_len=self.config.max_seq_len,
            batch_size=self.config.embed_calc_batch_size,
            device=get_device(),
            skip_n=0,
            n_files=n_files
        )
        
        try:
            log_stage("ProtX Validation Data Gen")
            result = protx_val_data.process_dataset(
                save_path=Path(self.config.protx_val_prefix)
            )
            
            if n_files > 1:
                val_files = result
                logger.info(f"Created {len(val_files)} validation dataset shard files")
            else:
                val_files = [result] if result else []
                logger.info(f"Created validation dataset file: {result}")
            
            return val_files
        except Exception as e:
            logger.error(f"Save ProtX Val Dataset failed: {e}")
            raise
    
    def run_all(self, save_protx_dataset=False, n_files: int = 1):
        """Run the complete pipeline - equivalent to running all CLI commands in sequence"""
        self.download()
        self.extract()
        self.shuffle_fasta()
        self.filter_split()
        if save_protx_dataset:
            self.save_protx_train_dataset(n_files=n_files)
            self.save_protx_val_dataset(n_files=n_files)
        logger.info("Complete pipeline execution finished!")
