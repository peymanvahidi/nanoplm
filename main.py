import argparse

from src.protx.config import DataConfig, DistillConfig
from src.protx.data import DataPipeline
from src.protx.distillation.pipeline import DistillPipeline

def main():
    # Create parser
    parser = argparse.ArgumentParser(description="ProtX pipeline runner")
    
    # Data processing arguments
    parser.add_argument("--download-data", action="store_true", help="Run data download step")
    parser.add_argument("--extract-data", action="store_true", help="Run data extraction step")
    parser.add_argument("--filter-split-data", action="store_true", help="Run data filtering and splitting step")
    parser.add_argument("--save-protx-dataset", action="store_true", help="Saves train/val datasets ready to be used for student model")
    parser.add_argument("--train-model", action="store_true", help="Train the model with the specified parameters")
    
    args = parser.parse_args()
    
    # Initialize the pipeline
    data_config = DataConfig()
    data_pipeline = DataPipeline(data_config)
    
    if args.download_data:
        data_pipeline.download()
    
    if args.extract_data:
        data_pipeline.extract()
        
    if args.filter_split_data:
        data_pipeline.filter_split()
    
    if args.save_protx_dataset:
        data_pipeline.save_protx_train_dataset()
        data_pipeline.save_protx_val_dataset()
    
    if args.train_model:
        distill_config = DistillConfig()
        distill_pipeline = DistillPipeline(data_config, distill_config)
        distill_pipeline.train()

if __name__ == "__main__":
    main()
