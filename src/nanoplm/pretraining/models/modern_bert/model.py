from dataclasses import dataclass
from typing import Union, List, Optional
import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput

from transformers import ModernBertConfig, ModernBertForMaskedLM, ModernBertModel, ModernBertPredictionHead
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
            mask_token_id=self.tokenizer.mask_token_id,
            #_attn_implementation="eager"
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

    def _setup_modular_architecture(self, cfg: ProtModernBertMLMConfig):
            # parse triangular layers
            if isinstance(cfg.triangular_layers, str):
                if cfg.triangular_layers.lower() == "all":
                    tri_layers = list(range(1, cfg.num_hidden_layers))
                else:
                    tri_layers = [int(x.strip()) for x in cfg.triangular_layers.split(",")]
            else:
                tri_layers = cfg.triangular_layers or [3, 11]

            tri_layers = sorted(tri_layers)
            segment_boundaries = [0] + [i + 1 for i in tri_layers] + [cfg.num_hidden_layers]
            self.segments = [(segment_boundaries[i], segment_boundaries[i + 1])
                            for i in range(len(segment_boundaries) - 1)]

            print(f"   Segments: {self.segments}")

            # build submodules
            self.bert_segments = nn.ModuleList()
            self.triangular_blocks = nn.ModuleList()

            shared_embeddings = None

            for i, (start, end) in enumerate(self.segments):
                seg_cfg = self.config.to_dict()
                seg_cfg = ModernBertConfig(**seg_cfg)
                seg_cfg.num_hidden_layers = end - start
                bert_segment = ModernBertModel(seg_cfg)
                print("➡️ Embedding attributes:", dir(bert_segment.embeddings))

                if shared_embeddings is None:
                    shared_embeddings = bert_segment.embeddings
                else:
                    bert_segment.embeddings = shared_embeddings

                self.bert_segments.append(bert_segment)

                if i < len(self.segments) - 1:
                    tri_block = create_triangular_attention_layer(
                        residue_dim=cfg.hidden_size,
                        pair_dim=cfg.triangular_pair_dim or cfg.hidden_size,
                        num_heads=cfg.triangular_heads or 4,
                        dropout=cfg.triangular_dropout or 0.1,
                    )
                    self.triangular_blocks.append(tri_block)

            self.head = ModernBertPredictionHead(config)
            self.decoder = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=cfg.decoder_bias)


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
        Modular forward pass with gradient and hidden-state debugging
        """
        # Filter kwargs to only include parameters that ModernBertModel accepts
        bert_kwargs = {k: v for k, v in kwargs.items() if k in {
            'head_mask', 'encoder_hidden_states', 'encoder_attention_mask',
            'past_key_values', 'use_cache', 'output_attentions', 'output_hidden_states', 
            'return_dict', 'training'
        }}

        hidden_states = None
        all_hidden_states = [] if output_hidden_states else None

        for i, bert_segment in enumerate(self.bert_segments):
            if i == 0:
                seg_out = bert_segment(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )
            else:
                seg_out = bert_segment(
                    inputs_embeds=hidden_states,
                    attention_mask=attention_mask,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )

            hidden_states = seg_out.last_hidden_state
            hidden_states = self.bert_segments[i].embeddings.norm(hidden_states)

            # DEBUG: hidden states stats
            print(f"[Segment {i}] hidden_states min/max/mean/std:",
                hidden_states.min().item(), hidden_states.max().item(),
                hidden_states.mean().item(), hidden_states.std().item())

            if output_hidden_states:
                all_hidden_states.extend(seg_out.hidden_states)

            # Optional triangular block
            if i < len(self.triangular_blocks):
                tri_block = self.triangular_blocks[i]
                hidden_states, _ = tri_block(hidden_states, pair_repr=None, mask=attention_mask)

                print(f"[Segment {i}] after triangular min/max/mean/std:",
                    hidden_states.min().item(), hidden_states.max().item(),
                    hidden_states.mean().item(), hidden_states.std().item())

        # DEBUG: hook to capture gradients
        def grad_hook(name):
            def hook(grad):
                print(f"GRAD {name}: norm={grad.norm().item():.6f}, mean={grad.mean().item():.6f}, std={grad.std().item():.6f}")
            return hook

        # Register hooks for head and decoder
        self.head.weight.register_hook(grad_hook("head.weight"))
        self.decoder.weight.register_hook(grad_hook("decoder.weight"))

        logits = self.decoder(self.head(hidden_states))

        loss = None
        if labels is not None:
            loss = self.ForMaskedLMLoss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                ignore_index=-100
            )
            print(f"⚠️ loss (per token): {loss.item():.6f}")

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )
    
    @staticmethod
    def ForMaskedLMLoss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        vocab_size: int,
        num_items_in_batch: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        **kwargs,
    ):
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()

        # Flatten the tokens
        logits = logits.view(-1, vocab_size)
        labels = labels.view(-1)

        labels = labels.to(logits.device)
        loss = fixed_cross_entropy(logits, labels, num_items_in_batch, ignore_index, **kwargs)
        return loss
    
def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss