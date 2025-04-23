import torch
import os
import logging
import h5py
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np

from ..utils.common import get_device
from ..models.teacher import TeacherModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeacherEmbedder:
    """
    Only using the encoder part of the teacher model to generate embeddings.
    """
    
    def __init__(
        self, 
        model_name="Rostlab/prot_t5_xl_uniref50",
        output_dir="pipeline_output"
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir) / "data/processed/teacher_embeddings"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = get_device()
        
        self.teacher = TeacherModel(model_name=model_name)
    
    def process_hdf5_batch(self, h5_file, batch_idx):
        """
        Process a single batch from an HDF5 file to generate embeddings.
        """
        batch_group_name = f"batch_{batch_idx:05d}"
        
        batch_group = h5_file[batch_group_name]
        input_ids = torch.tensor(batch_group['input_ids'][()])
        attention_mask = torch.tensor(batch_group['attention_mask'][()])
        seq_ids = list(batch_group['seq_ids'][()])
        
        with torch.no_grad():
            embeddings = self.teacher.get_encoder_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        return {
            'embeddings': embeddings,
            'seq_ids': seq_ids
        }
    
    def process_dataset(self, dataset_path: Path):
        """
        Process all batches in the dataset to generate embeddings.
        """
        logger.info(f"Processing dataset from {dataset_path}")
        
        if 'train' in dataset_path.name:
            dataset_name = 'train'
        elif 'val' in dataset_path.name:
            dataset_name = 'val'
        else:
            raise ValueError(f"Unknown dataset type: {dataset_path.name}")
        
        # Define output HDF5 file path
        output_h5_file = self.output_dir / f"{dataset_name}.h5"
        logger.info(f"Embeddings will be saved to {output_h5_file}")
        
        # Define string datatype for HDF5
        string_dt = h5py.string_dtype(encoding='utf-8')
        
        # Open the input HDF5 file and create the output HDF5 file
        with h5py.File(dataset_path, 'r') as in_hf, h5py.File(output_h5_file, 'w') as out_hf:
            # Extract metadata from HDF5 attributes
            metadata = {
                'num_seqs': in_hf.attrs['num_seqs'],
                'num_batches': in_hf.attrs['num_batches'],
                'batch_size': in_hf.attrs['batch_size'],
                'tokenizer': in_hf.attrs['tokenizer'],
                'teacher_model': self.model_name
            }
            
            # Store metadata as attributes in the output HDF5 file
            for key, value in metadata.items():
                # Convert Path objects or complex objects to strings
                if key == 'teacher_model' and not isinstance(value, (str, int, float, bool)):
                    out_hf.attrs[key] = str(value)
                else:
                    out_hf.attrs[key] = value
            
            # Also save legacy metadata.pt for backward compatibility
            torch.save(metadata, self.output_dir / "metadata.pt")
            
            num_batches = metadata['num_batches']
            logger.info(f"Dataset contains {metadata['num_seqs']} sequences in {num_batches} batches")
            
            # Process each batch
            for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
                # Process the batch from HDF5
                result = self.process_hdf5_batch(in_hf, batch_idx)
                
                # Create a group for this batch in the output HDF5 file
                batch_group = out_hf.create_group(f"batch_{batch_idx:05d}")
                
                # Save embeddings and sequence IDs in the batch group
                batch_group.create_dataset('embeddings', data=result['embeddings'].cpu().numpy())
                batch_group.create_dataset('seq_ids', data=result['seq_ids'], dtype=string_dt)
        
        logger.info(f"Finished processing dataset. Embeddings saved to {output_h5_file}")
        
        # Return the path to the output directory
        return self.output_dir

    def process_all(self):
        """
        Process all datasets (train and validation) to generate embeddings.
        """
        logger.info("Processing all datasets")
        logger.info(f"Using device: {self.teacher.device}")
        
        # Get base processed directory from output_dir
        base_dir = self.output_dir.parent / "tokenized"
        
        logger.info(f"Looking for train and val data in: {base_dir}")
        
        for dataset_name in ["train", "val"]:
            dataset_path = base_dir / f"{dataset_name}.h5"
            if dataset_path.exists():
                self.process_dataset(dataset_path)
            else:
                logger.warning(f"{dataset_name} file not found: {dataset_path}")
        
        # Create a file_structure.json file to document the HDF5 file structure
        file_structure = {
            "description": "#### This file is just for DEMONSTRATION of the .h5 file structure ####",
            "attributes": {
                "num_seqs": "integer - Total number of sequences in the dataset",
                "num_batches": "integer - Total number of batches",
                "batch_size": "integer - Size of each batch",
                "tokenizer": "string - Name of the tokenizer used",
                "teacher_model": "string - Name of the teacher model used"
            },
            "groups": {
                "batch_00000": {
                    "description": "Group containing data for a single batch",
                    "datasets": {
                        "embeddings": "numpy array - Shape: [batch_size, sequence_length, embedding_dim] - Model embeddings for sequences",
                        "seq_ids": "string array - Sequence identifiers"
                    }
                },
                "batch_00001": "... and so on for each batch"
            }
        }
        
        # Save the file structure to JSON
        with open(self.output_dir / "file_structure.json", 'w') as f:
            json.dump(file_structure, f, indent=2)
        
        logger.info(f"File structure documentation saved to {self.output_dir / 'file_structure.json'}")
        logger.info("Finished processing all datasets")
