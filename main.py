import argparse
from pathlib import Path

from src.protx.data.config import Config
from src.protx.data import DataPipeline
from src.protx.models.teacher import TeacherModel
from src.protx.distillation.teacher_embed import TeacherEmbedder

def main():
    # Create parser
    parser = argparse.ArgumentParser(description="ProtX pipeline runner")
    
    # Data processing arguments
    parser.add_argument("--download-data", action="store_true", help="Run data download step")
    parser.add_argument("--extract-data", action="store_true", help="Run data extraction step")
    parser.add_argument("--filter-split-data", action="store_true", help="Run data filtering and splitting step")
    parser.add_argument("--generate-teacher-embeddings", action="store_true", help="Generate embeddings using the teacher model")
    
    # Add more arguments for other pipeline steps as needed
    # parser.add_argument("--train", action="store_true", help="Run training step")
    # parser.add_argument("--evaluate", action="store_true", help="Run evaluation step")
    
    args = parser.parse_args()
    
    # Initialize the pipeline
    config = Config()
    data_pipeline = DataPipeline(config)
    
    # Run individual steps
    if args.download_data:
        data_pipeline.download()
    
    if args.extract_data:
        data_pipeline.extract()
        
    if args.filter_split_data:
        data_pipeline.filter_split()
    
    # Generate teacher embeddings if requested
    if args.generate_teacher_embeddings:
        embedder = TeacherEmbedder()
        embedder.process_all()

if __name__ == "__main__":
    main()
