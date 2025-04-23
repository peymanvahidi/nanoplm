import random
import h5py
import json
from pathlib import Path
from Bio import SeqIO

def analyze_embedding(output_dir: Path = Path("testing_dir")):

    # Define paths
    fasta_file = output_dir / "data/processed/train.fasta"
    embedding_file = output_dir / "data/processed/teacher_embeddings/train.h5"

    # Read all sequences from the FASTA file
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    if not sequences:
        raise ValueError(f"No sequences found in {fasta_file}")

    # Select a random sequence
    random_seq = random.choice(sequences)
    seq_id = random_seq.id
    seq_str = str(random_seq.seq)

    print(f"Selected random sequence ID: {seq_id}")
    print(f"Sequence length (amino acids): {len(seq_str)}")
    print(f"First 50 amino acids: {seq_str[:50]}...")

    # Open the embedding file and find the matching sequence
    with h5py.File(embedding_file, 'r') as f:
        # Search in all batches for the matching sequence ID
        found = False
        for batch_name in f.keys():
            if batch_name.startswith("batch_"):
                # Get sequence IDs in this batch
                seq_ids = f[batch_name]['seq_ids'][()]
                seq_ids = [s.decode('utf-8') if isinstance(s, bytes) else s for s in seq_ids]
                
                # Check if our sequence ID is in this batch
                if seq_id in seq_ids:
                    idx = seq_ids.index(seq_id)
                    
                    # Get embedding
                    embedding = f[batch_name]['embeddings'][idx]
                    embedding_shape = embedding.shape

                    # Now need to find the attention mask and input_ids
                    # Since they're not directly in the embedding file, we need to look at the original tokenized file
                    tokenized_file = output_dir / "data/processed/tokenized/train.h5"
                    
                    with h5py.File(tokenized_file, 'r') as tok_f:
                        # Get attention mask and input_ids from the same batch
                        attention_mask = tok_f[batch_name]['attention_mask'][idx]
                        input_ids = tok_f[batch_name]['input_ids'][idx]
                    
                    # Count zeros and ones in attention mask
                    zeros = (attention_mask == 0).sum()
                    ones = (attention_mask == 1).sum()
                    
                    # Protein sequence tokenization can be different from the raw sequence length,
                    # so let's calculate the expected length
                    tokenized_length = min(len(seq_str) + 1, 512)  # +1 for special tokens, max 512
                    
                    # Verify assertions
                    print("\nVerification Results:")
                    
                    # Assertion 1: embedding shape
                    print(f"1. Embedding shape should be (ones+zeros in attention mask, 1024)")
                    condition_1 = embedding_shape[0] == ones+zeros
                    print(f"   - Match with ones+zeros in attention mask: {condition_1}")
                    
                    # Assertion 2: number of ones in attention mask
                    print(f"\n2. Number of ones in attention_mask should be equal to min(sequence_length, 512)")
                    print(f"   - Expected ones: ~{tokenized_length}")
                    print(f"   - Actual ones: {ones}")
                    condition_2 = ones == tokenized_length
                    print(f"   - Match: {condition_2}")
                    
                    # Assertion 3: ones + zeros = total length
                    print(f"\n3. Number of ones + zeros in attention_mask should equal the token sequence length")
                    print(f"   - Ones: {ones}")
                    print(f"   - Zeros: {zeros}")
                    total_length = ones + zeros
                    print(f"   - Total length (ones + zeros): {total_length}")
                    condition_3 = ones + zeros == len(attention_mask)
                    print(f"   - Ones + zeros match attention mask length: {condition_3}")
                    condition_4 = len(seq_str) + 1 == ones
                    print(f"   - Sequence length matches ones: {condition_4}")
                    
                    # Assertion 4: zeros in attention mask -> zeros in input_ids
                    mask_zeros_indices = [i for i, x in enumerate(attention_mask) if x == 0]
                    input_at_mask_zeros = [input_ids[i] for i in mask_zeros_indices]
                    condition_5 = all(token == 0 for token in input_at_mask_zeros)
                    print(f"   - All zeros match: {condition_5}")
                    
                    print(f"\n4. Zeros in attention mask should correspond to zeros in tokenized sequence")
                    print(f"   - All zeros match: {condition_5}")
                    if not condition_5:
                        # Show some examples where they don't match
                        mismatches = [(i, input_ids[i]) for i in mask_zeros_indices if input_ids[i] != 0]
                        print(f"   - Mismatches (up to 5): {mismatches[:5]}")
                    
                    found = True
                    break
        
        if not found:
            raise ValueError(f"Sequence ID {seq_id} not found in embedding file")
        
    # Based on all the conditions return boolean
    return (condition_1, condition_2, condition_3, condition_4, condition_5)