from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from transformers import ModernBertConfig, ModernBertForMaskedLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer


class SwiGLU(nn.Module):
    def forward(self, x, gate):
        return F.silu(gate) * x


class ModernBertMLPSwiGLU(nn.Module):
    """Replacement MLP that applies SwiGLU to each ModernBERT layer."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.Wi = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=config.mlp_bias)
        self.drop = nn.Dropout(config.mlp_dropout)
        self.act = SwiGLU()
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states):
        x, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(x, gate)))


class ModernBertNoOpMLP(nn.Module):
    def forward(self, hidden_states):
        return hidden_states.new_zeros(hidden_states.shape)


@dataclass
class ProtModernBertMLMConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_kv_heads: Optional[int] = None  # GQA: K,V head count (None = same as num_attention_heads = MHA)
    vocab_size: int = 32
    mlp_activation: str = "swiglu"
    mlp_dropout: float = 0.0
    mlp_bias: bool = False
    no_mlp_on_first_layer: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    classifier_activation: str = "gelu"
    use_resid_lambdas: bool = False
    use_x0_lambdas: bool = False
    use_qk_norm: bool = False
    use_canon_layers: bool = False
    canon_layers_mode: str = "abcd"
    canon_layers_kernel_size: Optional[int] = None
    resid_lambda_init: float = 1.0
    x0_lambda_init: float = 0.1
    use_repo: bool = False
    repo_after_n_layers: int = 3
    use_prores: bool = False
    prores_T: int = 1000
    gradient_checkpointing: bool = False
    gradient_checkpointing_mode: str = "layer"
    use_mhc_lite: bool = False
    mhc_n_streams: int = 4
    mhc_triton_fused: bool = False
    mhc_lite_wrapping_level: str = "layer"
    use_diff_attn_v2: bool = False
    attn_layer_pattern: Optional[str] = None


class ProtModernBertMLM(ModernBertForMaskedLM):

    def __init__(
        self,
        config: ProtModernBertMLMConfig
    ):
        if config.use_canon_layers:
            raise ValueError(
                "Canon layers are currently implemented only in the pure-torch path. "
                "Use --pure-torch with use_canon_layers=true."
            )

        self.tokenizer = ProtModernBertTokenizer()
        # Keep the original high-level config for checkpoint serialization.
        self.model_config = config

        self.config = ModernBertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            # Keep this comfortably above common dataset max_seq_len values.
            # RoPE frequencies are generated from this bound at runtime.
            max_position_embeddings=8192,
            mlp_dropout=config.mlp_dropout,
            mlp_bias=config.mlp_bias,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            classifier_activation=config.classifier_activation,
            # Set correct token IDs from our tokenizer
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=None,  # Not used in our tokenizer
            unk_token_id=self.tokenizer.unk_token_id,
            mask_token_id=self.tokenizer.mask_token_id,
            loss_type="ForMaskedLM",
        )

        super().__init__(self.config)
        # PreTrainedModel.__init__ auto-infers loss_type from class name via
        # regex against LOSS_MAPPING keys. "ProtModernBertMLM" doesn't contain
        # "ForMaskedLM", so it falls back to None → ForCausalLMLoss (which
        # shifts labels left by 1 — wrong for MLM). Override it here.
        self.loss_type = "ForMaskedLM"

        # Apply SwiGLU activation to MLP layers if specified
        if config.mlp_activation.lower() == "swiglu":
            for layer in self.model.layers:
                layer.mlp = ModernBertMLPSwiGLU(self.config)
        if config.no_mlp_on_first_layer and len(self.model.layers) > 0:
            first_layer = self.model.layers[0]
            if hasattr(first_layer, "mlp_norm"):
                first_layer.mlp_norm = nn.Identity()
            first_layer.mlp = ModernBertNoOpMLP()
