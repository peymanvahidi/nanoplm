from ..utils.logger import logger, log_stage
from .download import Downloader
from .extract import Extractor
from .preprocess import Preprocessor
from .split import Splitter

class DataPipeline:
    """Complete data processing pipeline combining all steps."""
    
    def __init__(self):
        self.downloader = Downloader()
        self.extractor = Extractor()
        self.preprocessor = Preprocessor()
        self.splitter = Splitter()
    
    def run_download(self):
        """Run only the download step."""
        log_stage("DOWNLOAD")
        self.downloader.download()
    
    def run_extract(self):
        """Run only the extract step."""
        log_stage("EXTRACT")
        self.extractor.extract()
    
    def run_preprocess(self):
        """Run only the preprocess step."""
        log_stage("PREPROCESS")
        self.preprocessor.preprocess()
    
    def run_split(self):
        """Run only the split step."""
        log_stage("SPLIT")
        self.splitter.split()
    
    def run_all(self):
        """Run the complete pipeline."""
        self.run_download()
        self.run_extract()
        self.run_preprocess()
        self.run_split()
        logger.info("Complete pipeline execution finished!")

if __name__ == "__main__":
    # For direct execution of the full pipeline
    data_pipeline = DataPipeline()
    data_pipeline.run_all() 