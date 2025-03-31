import torch
import argparse
from pathlib import Path

def inspect_batch(batch_file, max_sequences=3):
    """
    Load and inspect a batch file.
    
    Args:
        batch_file: Path to the .pt batch file
        max_sequences: Maximum number of sequences to display
    """
    print(f"\nInspecting batch file: {batch_file}")
    
    # Load the batch data
    batch_data = torch.load(batch_file)
    
    # Get batch information
    num_sequences = len(batch_data['sequences'])
    input_ids_shape = batch_data['input_ids'].shape
    attention_mask_shape = batch_data['attention_mask'].shape
    
    print(f"Batch contains {num_sequences} sequences")
    print(f"Input IDs tensor shape: {input_ids_shape}")
    print(f"Attention mask tensor shape: {attention_mask_shape}")
    
    # Display samples
    print(f"\nDisplaying up to {max_sequences} samples:")
    for i in range(min(max_sequences, num_sequences)):
        print(f"\nSample {i+1}:")
        print(f"Sequence ID: {batch_data['sequence_ids'][i]}")
        print(f"Original sequence (with spaces): {batch_data['sequences'][i][:60]}...")
        print(f"Input IDs: {batch_data['input_ids'][i][:20].tolist()}...")
        print(f"Attention mask: {batch_data['attention_mask'][i][:20].tolist()}...")

def inspect_dataset(dataset_dir, num_batches=2):
    """
    Inspect a tokenized dataset.
    
    Args:
        dataset_dir: Path to the dataset directory
        num_batches: Number of batch files to inspect
    """
    dataset_path = Path(dataset_dir)
    
    # Check if metadata exists
    metadata_file = dataset_path / "metadata.pt"
    if metadata_file.exists():
        metadata = torch.load(metadata_file)
        print(f"\nDataset metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    # Find all batch files
    batch_files = sorted(list(dataset_path.glob("batch_*.pt")))
    
    if not batch_files:
        print(f"No batch files found in {dataset_path}")
        return
    
    print(f"\nFound {len(batch_files)} batch files")
    
    # Inspect a subset of batches
    for i, batch_file in enumerate(batch_files[:num_batches]):
        inspect_batch(batch_file)
        
        if i < len(batch_files[:num_batches]) - 1:
            print("\n" + "-" * 80)  # Separator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect tokenized batches")
    parser.add_argument("dataset_dir", type=str, help="Path to the tokenized dataset directory")
    parser.add_argument("--num-batches", type=int, default=2, help="Number of batches to inspect")
    parser.add_argument("--batch-file", type=str, help="Inspect a specific batch file")
    
    args = parser.parse_args()
    
    if args.batch_file:
        inspect_batch(args.batch_file)
    else:
        inspect_dataset(args.dataset_dir, args.num_batches) 