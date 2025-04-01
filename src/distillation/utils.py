import torch
from torch import nn
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path
import logging

from ..models.teacher import TeacherModel


def initialize_student_from_teacher(
    student_model: nn.Module,
    teacher_model: TeacherModel,
    layer_mapping: Dict[str, str] = None,
    layer_stride: int = None
) -> nn.Module:
    """
    Initialize student model weights from teacher model by copying parameters
    from specific layers according to a mapping.
    
    Args:
        student_model: Student model to initialize
        teacher_model: Teacher model to copy weights from
        layer_mapping: Dictionary mapping student layer names to teacher layer names
                      If None, attempts to use mapping based on layer_stride or identical layer names
        layer_stride: Take every Nth layer (e.g., 4 means take layers 0,4,8,12,16,20)
                      
    Returns:
        Initialized student model
    """
    # Get teacher model weights
    teacher_weights = teacher_model.get_layer_weights()
    student_state = student_model.state_dict()
    
    # Create layer mapping if not provided
    if not layer_mapping:
        if layer_stride:
            layer_mapping = create_t5_layer_mapping(teacher_weights, student_state, layer_stride)
        else:
            layer_mapping = {k: k for k in student_state.keys() if k in teacher_weights}
    
    # Initialize matched layers
    initialized_layers = []
    with torch.no_grad():  # Prevent in-place gradient tracking errors
        for student_key, teacher_key in layer_mapping.items():
            if student_key in student_state and teacher_key in teacher_weights:
                # Check if shapes are compatible
                if student_state[student_key].shape == teacher_weights[teacher_key].shape:
                    student_state[student_key].copy_(teacher_weights[teacher_key])
                    initialized_layers.append(student_key)
                else:
                    logging.warning(
                        f"Shape mismatch for {student_key} ({student_state[student_key].shape}) "
                        f"and {teacher_key} ({teacher_weights[teacher_key].shape})"
                    )
    
    # Load the updated state dict
    student_model.load_state_dict(student_state)
    
    logging.info(f"Initialized {len(initialized_layers)} layers from teacher model")
    return student_model


def create_t5_layer_mapping(teacher_state_dict, student_state_dict, layer_stride):
    """
    Create a custom mapping for T5 architecture, selecting every Nth attention layer
    
    Args:
        teacher_state_dict: Dictionary of teacher model state
        student_state_dict: Dictionary of student model state
        layer_stride: Take every Nth layer (e.g., 4 means take layers 0,4,8,12,16,20)
        
    Returns:
        Dictionary mapping student layer names to teacher layer names
    """
    mapping = {}
    
    # Find all teacher attention layers
    teacher_attn_layers = [k for k in teacher_state_dict.keys() 
                           if 'SelfAttention' in k]
    
    # Group by block number
    teacher_blocks = {}
    for layer in teacher_attn_layers:
        # Extract block number - assumes format like "encoder.block.{X}.layer.0.SelfAttention.q"
        parts = layer.split('.')
        block_idx = int(parts[2])  # Get the X from "encoder.block.{X}"
        
        if block_idx % layer_stride == 0:  # Only select every Nth block
            if block_idx not in teacher_blocks:
                teacher_blocks[block_idx] = []
            teacher_blocks[block_idx].append(layer)
    
    # Sort blocks and create sequential mapping to student
    sorted_blocks = sorted(teacher_blocks.keys())
    student_block_idx = 0
    
    for teacher_block_idx in sorted_blocks:
        for layer in teacher_blocks[teacher_block_idx]:
            # Get corresponding student layer by replacing block number
            student_layer = layer.replace(f"block.{teacher_block_idx}", f"block.{student_block_idx}")
            
            # Only add mapping if student actually has this layer
            if student_layer in student_state_dict:
                mapping[student_layer] = layer
        
        student_block_idx += 1  # Increment student block index
    
    return mapping


def extract_encoder_embeddings(
    teacher_model: TeacherModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_ids: Union[int, List[int], None] = None
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Extract embeddings from specific encoder layers of the teacher model.
    
    Args:
        teacher_model: Teacher model to extract embeddings from
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        layer_ids: Layer index/indices to extract (None means only the last layer)
        
    Returns:
        If layer_ids is int or None: Tensor of shape [batch_size, seq_len, hidden_size]
        If layer_ids is list: List of tensors, one for each requested layer
    """
    # Get all hidden states
    _, all_hidden_states = teacher_model.get_encoder_embeddings(
        input_ids, 
        attention_mask, 
        output_hidden_states=True
    )
    
    # Return specific layers
    if layer_ids is None:
        return all_hidden_states[-1]  # Last layer
    elif isinstance(layer_ids, int):
        return all_hidden_states[layer_ids]
    else:
        return [all_hidden_states[i] for i in layer_ids]


def get_attention_matrices(
    teacher_model: TeacherModel, 
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor
) -> List[torch.Tensor]:
    """
    Extract attention matrices from the full model for analysis or 
    attention-based knowledge distillation.
    
    Args:
        teacher_model: Teacher model to extract attention matrices from
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        
    Returns:
        List of attention matrices, one per layer
        Each matrix has shape [batch_size, num_heads, seq_len, seq_len]
    """
    with torch.no_grad():
        # Run forward pass with output_attentions=True
        outputs = teacher_model.full_model(
            input_ids=input_ids.to(teacher_model.device),
            attention_mask=attention_mask.to(teacher_model.device),
            output_attentions=True
        )
        
        # Extract and return attention matrices
        return outputs.encoder_attentions 