import argparse
from src.data import DataPipeline

def main():
    # Create parser
    parser = argparse.ArgumentParser(description="ProtT5-s pipeline runner")
    
    # Data processing arguments
    parser.add_argument("--download-data", action="store_true", help="Run only the data download step")
    parser.add_argument("--extract-data", action="store_true", help="Run only the data extraction step")
    parser.add_argument("--filter-data", action="store_true", help="Run only the data filtering step")
    parser.add_argument("--split-data", action="store_true", help="Run only the data splitting step")
    parser.add_argument("--tokenize-data", action="store_true", help="Run only the data tokenization step")
    
    # Add more arguments for other pipeline steps as needed
    # parser.add_argument("--train", action="store_true", help="Run only the training step")
    # parser.add_argument("--evaluate", action="store_true", help="Run only the evaluation step")
    
    args = parser.parse_args()
    
    # Initialize the pipeline
    data_pipeline = DataPipeline()
    
    # Run individual steps
    if args.download_data:
        data_pipeline.run_download()
    
    if args.extract_data:
        data_pipeline.run_extract()
        
    if args.filter_data:
        data_pipeline.run_filter()
        
    if args.split_data:
        data_pipeline.run_split()
        
    if args.tokenize_data:
        data_pipeline.run_tokenize()

if __name__ == "__main__":
    main()
