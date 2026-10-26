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
        self.Wi = nn.Linear(
            config.hidden_size, config.intermediate_size * 2, bias=config.mlp_bias
        )
        self.drop = nn.Dropout(config.mlp_dropout)
        self.act = SwiGLU()
        self.Wo = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

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
    num_kv_heads: Optional[int] = (
        None  # GQA: K,V head count (None = same as num_attention_heads = MHA)
    )
    vocab_size: int = 32
    num_register_tokens: int = 0  # 0, 1, or 2 — appended after EOS during tokenization
    norm_type: str = "layernorm"  # layernorm | rmsnorm
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
    reorder_RoPE_QKNorm: bool = False  # True = QK norm BEFORE RoPE
    global_rope_theta: float = 160_000.0  # RoPE base for full-attention layers
    local_rope_theta: float = 10_000.0  # RoPE base for sliding-window layers
    rope_dim_fraction: float = 1.0  # fraction of head_dim to apply RoPE (rest are NoPE content dims)
    use_canon_layers: bool = False
    canon_layers_mode: str = "abcd"
    canon_layers_kernel_size: Optional[int] = None
    resid_lambda_init: float = 1.0
    x0_lambda_init: float = 0.1
    use_repo: bool = False
    repo_after_n_layers: int = 3
    use_prores: bool = False
    prores_T: int = 1000
    activation_checkpointing: bool = False
    activation_checkpointing_mode: str = "layer"
    use_mhc_lite: bool = False
    mhc_n_streams: int = 4
    mhc_triton_fused: bool = False
    mhc_lite_wrapping_level: str = "layer"
    tie_word_embeddings: bool = True
    use_diff_attn_v2: bool = False
    use_paired_head_attention: bool = False
    attn_layer_pattern: Optional[str] = None
    fused_qkv: bool = True
    fused_up_gate: bool = True
    wo_init: str = "zero"
    # NOBLE
    use_noble: bool = False
    noble_rank: int = 64
    noble_alpha: float = 0.01
    noble_omega_range: tuple = (0.8, 1.2)
    noble_phi_std: float = 0.1
    noble_half_kaiming: bool = True
    noble_targets: str = "all"
    # MoE
    use_moe: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_use_bias_correction: bool = True
    moe_aux_loss_coef: float = 0.01
    moe_z_loss_coef: float = 5e-5
    moe_routed_scaling_factor: float = 1.0
    moe_n_group: int = 1  # group-limited routing: number of expert groups (1 = flat top-k)
    moe_topk_group: int = 1  # group-limited routing: number of groups kept per token
    moe_bias_update_rate: float = 1e-3  # Ling-style expert-bias update rate
    moe_leading_dense_layers: int = 0
    moe_kernel_backend: str = "cutlass"  # cutlass | sonicmoe (routed-expert compute)
    moe_nan_guard: bool = False  # skip-and-log a NaN step instead of crashing (see ModernBertConfig)
    # Huginn-style recycling (pure-torch path only). See ModernBertConfig for field semantics.
    use_huginn_looping: bool = False
    recycling_num_prelude_layers: int = 0
    recycling_num_coda_layers: int = 0
    recycling_loop_count: int = 1
    recycling_adapter_bias: bool = False
    recycling_s0_noise_sigma: float = 0.6324555320336759
    recycling_use_takase_init: bool = True
    # Loopie layer-loop (pure-torch path only). See ModernBertConfig.
    use_loopie: bool = False
    loopie_loop_count: int = 2
    loopie_bptt: Optional[int] = None


class ProtModernBertMLM(ModernBertForMaskedLM):
    def __init__(self, config: ProtModernBertMLMConfig):
        if config.recycling_loop_count != 1 or config.use_huginn_looping:
            raise ValueError(
                "Huginn-style recycling (recycling_loop_count > 1) is implemented only "
                "in the pure-torch path. Use --pure-torch with recycling_loop_count>1."
            )
        if config.use_loopie:
            raise ValueError(
                "Loopie layer looping (use_loopie=true) is implemented only in the "
                "pure-torch path. Use --pure-torch."
            )
        if config.use_noble:
            raise ValueError(
                "NOBLE is implemented only in the pure-torch path. "
                "Use --pure-torch with use_noble=true."
            )
        if not config.fused_qkv:
            raise ValueError(
                "fused_qkv=False is implemented only in the pure-torch path. "
                "Use --pure-torch with fused_qkv=false."
            )
        if not config.fused_up_gate:
            raise ValueError(
                "fused_up_gate=False is implemented only in the pure-torch path. "
                "Use --pure-torch with fused_up_gate=false."
            )
        if config.use_moe:
            raise ValueError(
                "MoE is implemented only in the pure-torch path. "
                "Use --pure-torch with use_moe=true."
            )
        if config.use_canon_layers:
            raise ValueError(
                "Canon layers are currently implemented only in the pure-torch path. "
                "Use --pure-torch with use_canon_layers=true."
            )
        if (
            config.num_kv_heads is not None
            and config.num_kv_heads != config.num_attention_heads
        ):
            raise ValueError(
                "GQA is not implemented in the HF ModernBERT path. "
                "Set num_kv_heads equal to num_attention_heads or use --pure-torch."
            )
        if str(config.classifier_activation).strip().lower() == "srelu":
            raise ValueError(
                "classifier_activation='srelu' is implemented only in the pure-torch "
                "path. Use --pure-torch or choose relu/gelu."
            )
        if config.use_paired_head_attention:
            raise ValueError(
                "Paired head attention is implemented only in the pure-torch path. "
                "Use --pure-torch or disable use_paired_head_attention."
            )

        self.tokenizer = ProtModernBertTokenizer(
            num_register_tokens=config.num_register_tokens
        )
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
            tie_word_embeddings=config.tie_word_embeddings,
            global_rope_theta=config.global_rope_theta,
            local_rope_theta=config.local_rope_theta,
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
