from dataclasses import dataclass
from typing import Union, List, Optional
import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput

from transformers import ModernBertConfig, ModernBertForMaskedLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
from nanoplm.pretraining.models.modern_bert.model_withTriangularAttention import ModernBertForMaskedLMWithTriangularAttention

@dataclass
class ProtModernBertMLMConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int = 32
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
            mask_token_id=self.tokenizer.mask_token_id,
            tie_word_embeddings=False,
            #_attn_implementation="eager"
        )

        # Manueller Fix wenn layer_types null ist:
        if not hasattr(self.config, 'layer_types') or self.config.layer_types is None:
            print("Fixing layer_types manually...")
            self.config.layer_types = [
                "sliding_attention" if bool(i % self.config.global_attn_every_n_layers) else "full_attention"
                for i in range(self.config.num_hidden_layers)
            ]
            print("layer_types nach Fix:", self.config.layer_types)

        if not hasattr(self.config, 'rope_parameters') or self.config.rope_parameters is None:
            print("Fixing rope_parameters manually...")
            # Manueller RoPE parameter fix für ältere Versionen
            self.config.rope_parameters = {
                "full_attention": {
                    "rope_theta": 160_000.0,
                    "rope_type": "default"
                },
                "sliding_attention": {
                    "rope_theta": 10_000.0, 
                    "rope_type": "default"
                }
            }
            print("rope_parameters nach Fix:", self.config.rope_parameters)
        
        if self.use_triangular_attention:
            print("🔺 Building MODULAR architecture with triangular attention")
            self.bert_model = ModernBertForMaskedLMWithTriangularAttention(self.config, triangular_attention_layers=config.triangular_layers, triangular_pair_dim=config.triangular_pair_dim, triangular_heads=config.triangular_heads, triangular_dropout=config.triangular_dropout)
            print("=== Weight Check ===")
            for name, param in self.bert_model.named_parameters():
                if 'weight' in name:
                    print(f"{name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")
                    if param.data.std() > 10.0 or param.data.mean().abs() > 1.0:
                        print(f"⚠️  PROBLEMATIC: {name}")
        else:
            print("🔧 Building STANDARD ModernBERT architecture")
            self.bert_model = ModernBertForMaskedLM(self.config)
        
        # Print model parameter count
        self._print_parameter_count()

    def _print_parameter_count(self):
        """Print detailed parameter count for the model"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n📊 MODEL PARAMETER COUNT:")
        print(f"   Total parameters:       {total_params:,}")
        print(f"   Trainable parameters:   {trainable_params:,}")
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], MaskedLMOutput]:
        return self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,            
            inputs_embeds=inputs_embeds,
            labels=labels,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )