import torch
import torch.nn as nn
import numpy as np
import csv
import os
from typing import Dict, Tuple


class FeatureEmbedding(nn.Module):
    """
    Enhanced embedding layer that converts token IDs to multi-feature representation:
    - Token embeddings (learnable)
    - 20 PCA features (computed from sliding window)
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int, 
        window_size: int = 3,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        unk_token_id: int = 2
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id
        
        # Token embeddings for discrete token IDs
        self.token_embeddings = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_token_id
        )
        
        # Linear projection for 20 PCA features
        self.continuous_projection = nn.Linear(20, embed_dim)
        
        # Fusion layer to combine token and continuous features  
        self.feature_fusion = nn.Linear(embed_dim * 2, embed_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Mapping from token IDs to amino acids (reverse of tokenizer vocab)
        self.id_to_aa = self._create_id_to_aa_mapping()
        
        # Load PCA features from CSV file
        self.pca_features = self._load_pca_features()
        
        # Convert to tensors for efficient GPU computation
        self._create_pca_tensors()
    
    def _create_id_to_aa_mapping(self) -> Dict[int, str]:
        """Create mapping from token IDs to amino acid letters"""
        # Standard ProtX tokenizer vocabulary
        vocab = {
            "A": 3, "L": 4, "G": 5, "V": 6, "S": 7, "R": 8, "E": 9, "D": 10,
            "T": 11, "I": 12, "P": 13, "K": 14, "F": 15, "Q": 16, "N": 17,
            "Y": 18, "M": 19, "H": 20, "W": 21, "C": 22, "X": 23, "B": 24,
            "O": 25, "U": 26, "Z": 27
        }
        return {v: k for k, v in vocab.items()}
    
    def _load_pca_features(self) -> Dict[str, np.ndarray]:
        """Load PCA features from CSV file"""
        pca_file_path = os.path.join(os.path.dirname(__file__), 'pca.csv')
        pca_features = {}
        
        with open(pca_file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header row
            
            for row in reader:
                aa = row[0].strip()
                # Convert the 20 PCA values to float array
                features = np.array([float(val.strip()) for val in row[1:21]])
                pca_features[aa] = features
        
        return pca_features
    
    def _create_pca_tensors(self):
        """Create tensors for efficient PCA feature lookup"""
        # Create lookup tensor for each token ID (20 features per amino acid)
        pca_values = torch.zeros(self.vocab_size, 20)
        
        for token_id, aa in self.id_to_aa.items():
            if aa in self.pca_features:
                pca_values[token_id] = torch.tensor(self.pca_features[aa], dtype=torch.float32)
            else:
                # Use zeros for unknown amino acids
                pca_values[token_id] = torch.zeros(20)
        
        # Register as buffer so it moves with the model to GPU/CPU
        self.register_buffer('pca_lookup', pca_values)
    
    def compute_sliding_window_features(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PCA features with sliding window.
        
        Args:
            input_ids: [batch_size, seq_len] tensor of token IDs
            attention_mask: [batch_size, seq_len] tensor of attention mask
            
        Returns:
            pca_features: [batch_size, seq_len, 20] tensor of PCA feature values
        """
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids to have 2 dimensions, got {input_ids.dim()} with shape {input_ids.shape}")
        
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(f"Expected attention_mask to have 2 dimensions, got {attention_mask.dim()} with shape {attention_mask.shape}")
            
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize output tensor for 20 PCA features
        pca_features = torch.zeros(batch_size, seq_len, 20, device=device)
        
        # Get base PCA values for all positions
        base_pca = self.pca_lookup[input_ids]  # [batch_size, seq_len, 20]
        
        # Apply sliding window averaging
        half_window = self.window_size // 2
        
        for i in range(seq_len):
            # Define window boundaries
            start_pos = max(0, i - half_window)
            end_pos = min(seq_len, i + half_window + 1)
            
            # Extract window values
            window_pca = base_pca[:, start_pos:end_pos, :]  # [batch_size, window_len, 20]
            window_mask = attention_mask[:, start_pos:end_pos]  # [batch_size, window_len]
            
            # Mask out padded positions
            window_mask_expanded = window_mask.unsqueeze(-1).float()  # [batch_size, window_len, 1]
            window_pca = window_pca * window_mask_expanded
            
            # Compute average (avoiding division by zero)
            window_lengths = window_mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]
            window_lengths = torch.clamp(window_lengths, min=1.0)  # Avoid division by zero
            
            # Sum over window and divide by window length
            pca_features[:, i, :] = window_pca.sum(dim=1) / window_lengths
        
        return pca_features
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with enhanced PCA features.
        
        Args:
            input_ids: [batch_size, seq_len] tensor of token IDs
            attention_mask: [batch_size, seq_len] tensor of attention mask
            
        Returns:
            embeddings: [batch_size, seq_len, embed_dim] tensor of enhanced embeddings
        """
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids to have 2 dimensions, got {input_ids.dim()} with shape {input_ids.shape}")
        
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(f"Expected attention_mask to have 2 dimensions, got {attention_mask.dim()} with shape {attention_mask.shape}")
        
        # Get token embeddings
        token_embeds = self.token_embeddings(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # Compute PCA features
        pca_features = self.compute_sliding_window_features(
            input_ids, attention_mask
        )  # [batch_size, seq_len, 20]
        
        # Project PCA features to embedding dimension
        continuous_embeds = self.continuous_projection(pca_features)  # [batch_size, seq_len, embed_dim]
        
        # Combine token and continuous embeddings
        combined = torch.cat([
            token_embeds, continuous_embeds
        ], dim=-1)  # [batch_size, seq_len, embed_dim * 2]
        
        # Fuse features back to original embedding dimension
        fused_embeds = self.feature_fusion(combined)  # [batch_size, seq_len, embed_dim]
        
        # Apply layer normalization
        embeddings = self.layer_norm(fused_embeds)
        
        return embeddings 