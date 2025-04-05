import torch
import os
import logging
from pathlib import Path
from tqdm import tqdm
from ..models.teacher import TeacherModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeacherEmbedder:
    """
    Class for generating and saving embeddings from the teacher model.
    Uses only the encoder part of the teacher model to generate embeddings.
    """
    
    def __init__(
        self, 
        model_name_or_path="Rostlab/prot_t5_xl_uniref50",
        device="mps" if torch.backends.mps.is_available() else "auto",  # Prioritize MPS on Mac
        output_dir="pipeline_output/data/processed/teacher_embeddings"
    ):
        """
        Initialize the teacher embedder.
        
        Args:
            model_name_or_path: Name or path of the teacher model
            device: Device to load the model on ("auto", "cpu", "cuda", "mps", etc.)
            output_dir: Directory to save the generated embeddings
        """
        self.model_name_or_path = model_name_or_path
        self.output_dir = Path(output_dir)
        self.device = device
        
        logger.info(f"Initializing TeacherEmbedder with device: {device}")
        if device == "auto" and torch.backends.mps.is_available():
            logger.info(f"MPS is available but not explicitly selected. Consider using 'mps' for Mac GPU acceleration.")
        
        # Load the teacher model (only encoder will be used)
        logger.info(f"Loading teacher model from {model_name_or_path}")
        self.teacher = TeacherModel(model_name_or_path=model_name_or_path, device=device)
        
        # Create the output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Embeddings will be saved to {self.output_dir}")
    
    def process_batch_file(self, batch_file):
        """
        Process a single batch file to generate embeddings.
        
        Args:
            batch_file: Path to the batch file
            
        Returns:
            Dictionary with embeddings and sequence_ids
        """
        logger.info(f"Processing batch file: {batch_file}")
        
        # Load the batch data
        batch_data = torch.load(batch_file)
        input_ids = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        sequence_ids = batch_data['sequence_ids']
        
        # Log tensor device information for debugging
        logger.info(f"Input tensor device before processing: {input_ids.device}")
        
        # Generate embeddings using only the encoder
        with torch.no_grad():
            embeddings = self.teacher.get_encoder_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        logger.info(f"Generated embeddings tensor device: {embeddings.device}")
        
        # Return embeddings and sequence IDs
        return {
            'embeddings': embeddings,
            'sequence_ids': sequence_ids
        }
    
    def process_dataset(self, input_dir="pipeline_output/data/processed/train_tokenized"):
        """
        Process all batch files in the dataset to generate embeddings.
        
        Args:
            input_dir: Directory containing the tokenized batch files
        """
        input_dir = Path(input_dir)
        logger.info(f"Processing dataset from {input_dir}")
        
        # Load metadata to get information about the dataset
        metadata_file = input_dir / "metadata.pt"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        metadata = torch.load(metadata_file)
        num_batches = metadata['num_batches']
        
        logger.info(f"Dataset contains {metadata['num_sequences']} sequences in {num_batches} batches")
        
        # Create output directory for this dataset
        dataset_name = input_dir.name
        output_subdir = self.output_dir / dataset_name
        output_subdir.mkdir(exist_ok=True, parents=True)
        
        # Save metadata in the output directory
        torch.save(metadata, output_subdir / "metadata.pt")
        
        # Process each batch file
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            batch_file = input_dir / f"batch_{batch_idx:05d}.pt"
            if not batch_file.exists():
                logger.warning(f"Batch file not found: {batch_file}")
                continue
            
            # Process the batch
            result = self.process_batch_file(batch_file)
            
            # Save the embeddings and sequence IDs
            output_file = output_subdir / f"embed_{batch_idx:05d}.pt"
            torch.save(result, output_file)
        
        logger.info(f"Finished processing dataset. Embeddings saved to {output_subdir}")
        
        # Return the path to the output directory
        return output_subdir

    def process_all(self):
        """
        Process all datasets (train and validation) to generate embeddings.
        """
        logger.info("Processing all datasets")
        logger.info(f"Using device: {self.teacher.device}")
        
        # Process train and validation datasets
        train_dir = "pipeline_output/data/processed/train_tokenized"
        val_dir = "pipeline_output/data/processed/val_tokenized"
        
        # Process train dataset if it exists
        if os.path.exists(train_dir):
            self.process_dataset(train_dir)
        
        # Process validation dataset if it exists
        if os.path.exists(val_dir):
            self.process_dataset(val_dir)
        
        logger.info("Finished processing all datasets")

