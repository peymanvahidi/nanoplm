from ..utils.logger import logger, log_stage
from .download import Downloader
from .extract import Extractor
from .preprocess import Preprocessor

class DataPipeline:
    """Complete data processing pipeline combining all steps."""
    
    def __init__(self, pipeline_output_dir=None, uniref50_url=None):
        self.downloader = Downloader(pipeline_output_dir=pipeline_output_dir)
        self.extractor = Extractor(pipeline_output_dir=pipeline_output_dir)
        self.preprocessor = Preprocessor(pipeline_output_dir=pipeline_output_dir)

        # Set URL if provided
        if uniref50_url:
            self.downloader.uniref50_url = uniref50_url
    
    def run_download(self):
        """Run only the download step."""
        log_stage("DOWNLOAD")
        self.downloader.download()
    
    def run_extract(self):
        """Run only the extract step."""
        log_stage("EXTRACT")
        self.extractor.extract()
    
    def run_filter(self):
        """Run only the preprocess step."""
        log_stage("FILTER")
        self.preprocessor.filter_sequences()
    
    def run_split(self):
        """Run only the split step."""
        log_stage("SPLIT")
        self.preprocessor.split()

    def run_tokenize(self):
        """Run only the tokenization step."""
        log_stage("TOKENIZE")
        self.preprocessor.tokenize_sequences()
    
    def run_all(self):
        """Run the complete pipeline."""
        self.run_download()
        self.run_extract()
        self.run_filter()
        self.run_split()
        self.run_tokenize()
        logger.info("Complete pipeline execution finished!")
