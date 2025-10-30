from dataclasses import dataclass
from typing import Union, List, Optional
import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput

from transformers import ModernBertConfig, ModernBertForMaskedLM, ModernBertModel
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
from nanoplm.models.student.triangular_attention import PairwiseTriangularBlock, create_triangular_attention_layer

@dataclass
class ProtModernBertMLMConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int = 29
    mlp_activation: str = "swiglu"
    mlp_dropout: float = 0.0
    mlp_bias: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    classifier_activation: str = "gelu"
    # Triangular Attention parameters (optional, only used if use_triangular_attention=True)
    use_triangular_attention: bool = False
    triangular_layers: Optional[Union[List[int], str]] = None
    triangular_pair_dim: Optional[int] = None
    triangular_heads: Optional[int] = None
    triangular_dropout: Optional[float] = None

class ProtModernBertMLM(nn.Module):
    """
    Clean implementation: either standard ModernBERT OR modular segments
    No inheritance confusion, no duplicate parameters
    """

    def __init__(
        self,
        config: ProtModernBertMLMConfig
    ):
        super().__init__()
        
        self.tokenizer = ProtModernBertTokenizer()
        self.use_triangular_attention = config.use_triangular_attention
        
        # Create ModernBERT config
        self.config = ModernBertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            max_position_embeddings=1024,
            mlp_dropout=config.mlp_dropout,
            mlp_bias=config.mlp_bias,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            classifier_activation=config.classifier_activation,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=None,
            unk_token_id=self.tokenizer.unk_token_id,
            mask_token_id=self.tokenizer.mask_token_id
        )
        
        if self.use_triangular_attention:
            print("🔺 Building MODULAR architecture with triangular attention")
            self._setup_modular_architecture(config)
        else:
            print("🔧 Building STANDARD ModernBERT architecture")
            self._setup_standard_architecture()
        
        # Print model parameter count
        self._print_parameter_count()

    def _setup_standard_architecture(self):
        """Setup standard single ModernBERT - clean and simple"""
        self.bert_model = ModernBertForMaskedLM(self.config)

    def _setup_modular_architecture(self, config):
        """Setup modular segments with triangular attention - clean separation"""
        # Set triangular attention parameters
        self.triangular_layers = config.triangular_layers or [3, 11]
        self.triangular_pair_dim = config.triangular_pair_dim or config.hidden_size
        self.triangular_heads = config.triangular_heads or 4
        self.triangular_dropout = config.triangular_dropout or 0.1
        
        print(f"   residue_dim: {config.hidden_size}")
        print(f"   triangular_layers: {self.triangular_layers}")
        print(f"   triangular_pair_dim: {self.triangular_pair_dim}")
        print(f"   triangular_heads: {self.triangular_heads}")
        print(f"   triangular_dropout: {self.triangular_dropout}")
        
        # Parse triangular layer indices
        if isinstance(self.triangular_layers, str):
            if self.triangular_layers.lower() == "all":
                triangular_indices = list(range(1, config.num_hidden_layers))
            else:
                triangular_indices = [int(x.strip()) for x in self.triangular_layers.split(',')]
        else:
            triangular_indices = self.triangular_layers
        
        triangular_indices = sorted(triangular_indices)
        print(f"   Triangular attention after layers: {triangular_indices}")
        
        # Calculate segment boundaries
        segment_boundaries = [0] + [idx + 1 for idx in triangular_indices] + [config.num_hidden_layers]
        self.segments = [(segment_boundaries[i], segment_boundaries[i+1]) for i in range(len(segment_boundaries)-1)]
        
        print(f"   ModernBERT segments: {self.segments}")
        
        # Create separate ModernBERT instances for each segment
        self.bert_segments = nn.ModuleList()
        self.triangular_blocks = nn.ModuleList()
        
        for i, (start_layer, end_layer) in enumerate(self.segments):
            segment_layers = end_layer - start_layer
            
            # Create ModernBERT config for this segment
            segment_config = ModernBertConfig(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_hidden_layers=segment_layers,
                num_attention_heads=config.num_attention_heads,
                max_position_embeddings=1024,
                mlp_dropout=config.mlp_dropout,
                mlp_bias=config.mlp_bias,
                attention_bias=config.attention_bias,
                attention_dropout=config.attention_dropout,
                classifier_activation=config.classifier_activation,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=None,
                unk_token_id=self.tokenizer.unk_token_id,
                mask_token_id=self.tokenizer.mask_token_id
            )
            
            # All segments are ModernBertModel (encoder only)
            bert_segment = ModernBertModel(segment_config)
            if i == 0:
                print(f"   📦 Segment {i}: ModernBERT layers {start_layer}-{end_layer-1} (with embeddings)")
            else:
                print(f"   📦 Segment {i}: ModernBERT layers {start_layer}-{end_layer-1} (encoder only)")
            
            self.bert_segments.append(bert_segment)
            
            # Add triangular attention after each segment except the last
            if i < len(self.segments) - 1:
                triangular_block = create_triangular_attention_layer(
                    residue_dim=config.hidden_size,
                    pair_dim=self.triangular_pair_dim,
                    num_heads=self.triangular_heads,
                    dropout=self.triangular_dropout
                )
                self.triangular_blocks.append(triangular_block)
                print(f"   🔺 Triangular attention block {i} created")
        if len(self.bert_segments) > 1:
            shared_embeddings = self.bert_segments[0].embeddings
            for seg in self.bert_segments[1:]:
                seg.embeddings = shared_embeddings
        # Create shared MLM head (like a normal ModernBERT)
        self.head = nn.Linear(config.hidden_size, config.hidden_size)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        # TODO: Tok_embeddings has high grad norm!
        
        print(f"   🎯 Shared MLM head created with stabilized initialization")
        
        # Add gradient hooks for debugging
        self._setup_gradient_hooks()

    def _print_parameter_count(self):
        """Print detailed parameter count for the model"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n📊 MODEL PARAMETER COUNT:")
        print(f"   Total parameters:       {total_params:,}")
        print(f"   Trainable parameters:   {trainable_params:,}")
        
        if self.use_triangular_attention:
            # Count parameters for each segment
            bert_segment_params = sum(sum(p.numel() for p in segment.parameters()) for segment in self.bert_segments)
            triangular_params = sum(sum(p.numel() for p in block.parameters()) for block in self.triangular_blocks)
            head_params = sum(p.numel() for p in self.head.parameters()) + sum(p.numel() for p in self.decoder.parameters())
            
            triangular_percentage = (triangular_params / total_params) * 100
            print(f"   ModernBERT segments:     {bert_segment_params:,}")
            print(f"   Triangular attention:   {triangular_params:,}")
            print(f"   MLM head:               {head_params:,}")
            print(f"   Triangular overhead:    {triangular_percentage:.2f}%")
            
            # Parameter breakdown per segment
            print(f"\n📦 BERT SEGMENT BREAKDOWN:")
            for i, bert_segment in enumerate(self.bert_segments):
                segment_params = sum(p.numel() for p in bert_segment.parameters())
                start_layer, end_layer = self.segments[i]
                print(f"   Segment {i} (layers {start_layer}-{end_layer-1}): {segment_params:,} parameters")
            
            # Parameter breakdown per triangular block
            print(f"\n🔺 TRIANGULAR ATTENTION BREAKDOWN:")
            for i, triangular_block in enumerate(self.triangular_blocks):
                block_params = sum(p.numel() for p in triangular_block.parameters())
                print(f"   Block {i}: {block_params:,} parameters")
        else:
            bert_params = sum(p.numel() for p in self.bert_model.parameters())
            print(f"   Standard ModernBERT:    {bert_params:,}")
        
        print(f"")  # Empty line for readability

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, 
                head_mask=None, inputs_embeds=None, labels=None, output_attentions=None, 
                output_hidden_states=None, return_dict=None, **kwargs):
        """
        Forward pass - standard ModernBERT or modular with triangular attention.
        """
        
        if self.use_triangular_attention:
            # Modular forward pass
            return self._forward_modular(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )
        else:
            # Standard ModernBERT forward pass - delegate to wrapped model
            return self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )

    def _forward_modular(self, input_ids, attention_mask=None, position_ids=None, 
                        inputs_embeds=None, labels=None, output_attentions=None, 
                        output_hidden_states=None, return_dict=None, **kwargs):
        """
        Modular forward pass: run separate ModernBERT segments with triangular attention
        """
        
        # Filter kwargs to only include parameters that ModernBertModel accepts
        bert_kwargs = {}
        valid_bert_params = {
            'head_mask', 'encoder_hidden_states', 'encoder_attention_mask',
            'past_key_values', 'use_cache', 'output_attentions', 'output_hidden_states', 
            'return_dict', 'training'
        }
        for key, value in kwargs.items():
            if key in valid_bert_params:
                bert_kwargs[key] = value
        
        all_hidden_states = []
        all_attentions = []
        current_hidden_state = None
        
        # Process each segment sequentially
        for segment_idx, bert_segment in enumerate(self.bert_segments):
            if segment_idx == 0:
                # First segment: use input_ids and get embeddings
                segment_outputs = bert_segment(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=True,
                    **bert_kwargs
                )
                current_hidden_state = segment_outputs.last_hidden_state
            else:
                # Subsequent segments: use inputs_embeds from previous segment
                segment_outputs = bert_segment(
                    inputs_embeds=current_hidden_state,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=True,
                    **bert_kwargs
                )
                current_hidden_state = segment_outputs.last_hidden_state
            
            # Collect hidden states and attentions
            if output_hidden_states:
                all_hidden_states.extend(segment_outputs.hidden_states)
            if output_attentions and segment_outputs.attentions:
                all_attentions.extend(segment_outputs.attentions)
            
            # Apply triangular attention after this segment (except the last)
            if segment_idx < len(self.triangular_blocks):
                triangular_block = self.triangular_blocks[segment_idx]
                enhanced_hidden_state, _ = triangular_block(
                    current_hidden_state, 
                    pair_repr=None, 
                    mask=attention_mask
                )
                current_hidden_state = enhanced_hidden_state
        
        # Use shared MLM head on final hidden state (simulates normal ModernBERT)
        prediction_scores = self.decoder(self.head(current_hidden_state))
        
        # Compute loss if labels are provided
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + (tuple(all_hidden_states) if output_hidden_states else (), tuple(all_attentions) if output_attentions else ())
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_attentions) if output_attentions else None,
        )
    
    def _setup_gradient_hooks(self):
        """Setup gradient hooks to debug gradient explosion"""
        
        def make_hook(name):
            def hook(grad):
                if grad is not None and torch.rand(1).item() < 0.01:
                    grad_norm = grad.norm().item()
                    grad_mean = grad.mean().item()
                    grad_std = grad.std().item()
                    print(f"🚨 GRAD {name}: norm={grad_norm:.2f}, mean={grad_mean:.6f}, std={grad_std:.6f}")
                    if grad_norm > 100:
                        print(f"⚠️  HIGH GRADIENT in {name}: {grad_norm:.2f}")
                return grad
            return hook
        
        if self.use_triangular_attention:
            # Hook ModernBERT segments
            for i, segment in enumerate(self.bert_segments):
                for name, param in segment.named_parameters():
                    if param.requires_grad:
                        param.register_hook(make_hook(f"Segment{i}.{name}"))
            
            # Hook triangular blocks
            for i, block in enumerate(self.triangular_blocks):
                for name, param in block.named_parameters():
                    if param.requires_grad:
                        param.register_hook(make_hook(f"Triangular{i}.{name}"))
            
            # Hook MLM head
            for name, param in self.head.named_parameters():
                if param.requires_grad:
                    param.register_hook(make_hook(f"MLMHead.{name}"))
            
            for name, param in self.decoder.named_parameters():
                if param.requires_grad:
                    param.register_hook(make_hook(f"MLMDecoder.{name}"))
