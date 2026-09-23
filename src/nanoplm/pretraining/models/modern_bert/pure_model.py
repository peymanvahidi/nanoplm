"""
Pure-torch protein ModernBERT MLM wrapper.

This is the pure-torch counterpart to ``model.py`` (which wraps HF's
``ModernBertForMaskedLM``).  It wraps the pure-torch
``ModernBertForMaskedLM`` from ``modeling.py`` instead.

The existing HF-based ``ProtModernBertMLM`` in ``model.py`` is left
completely untouched.
"""

from nanoplm.pretraining.models.modern_bert.modeling import (
    ModernBertConfig,
    ModernBertForMaskedLM,
)
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig


def modernbert_config_from(config: ProtModernBertMLMConfig) -> ModernBertConfig:
    """Map the user-facing model config onto the internal ModernBertConfig.

    Single source of truth. `nanoplm.hf` builds the exported model through this
    same function, so a newly added model flag reaches the exported model
    without a second edit.

    The tokenizer built below is not consulted for `num_register_tokens`
    (register tokens occupy fixed vocab slots and never move pad/eos/unk/mask);
    it exists solely to read those four fixed special-token ids into
    `ModernBertConfig`.

    Callers: a `moe_kernel_backend` of "sonicmoe" must be remapped to
    "cutlass" by the CALLER before calling this function — never inside it.
    This function forwards `moe_kernel_backend` faithfully by design, and the
    export path performs that remap upstream.
    """
    tokenizer = ProtModernBertTokenizer(num_register_tokens=config.num_register_tokens)
    return ModernBertConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_kv_heads=config.num_kv_heads,
        norm_type=config.norm_type,
        mlp_activation=config.mlp_activation,
        # Keep this comfortably above common dataset max_seq_len values.
        # Position embeddings are RoPE frequencies (not learned tables), so a
        # larger cap avoids runtime index asserts with minimal overhead.
        max_position_embeddings=8192,
        mlp_dropout=config.mlp_dropout,
        mlp_bias=config.mlp_bias,
        no_mlp_on_first_layer=config.no_mlp_on_first_layer,
        attention_bias=config.attention_bias,
        attention_dropout=config.attention_dropout,
        classifier_activation=config.classifier_activation,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=None,
        unk_token_id=tokenizer.unk_token_id,
        mask_token_id=tokenizer.mask_token_id,
        use_resid_lambdas=config.use_resid_lambdas,
        use_x0_lambdas=config.use_x0_lambdas,
        use_qk_norm=config.use_qk_norm,
        reorder_RoPE_QKNorm=config.reorder_RoPE_QKNorm,
        global_rope_theta=config.global_rope_theta,
        local_rope_theta=config.local_rope_theta,
        rope_dim_fraction=config.rope_dim_fraction,
        use_canon_layers=config.use_canon_layers,
        canon_layers_mode=config.canon_layers_mode,
        canon_layers_kernel_size=config.canon_layers_kernel_size,
        resid_lambda_init=config.resid_lambda_init,
        x0_lambda_init=config.x0_lambda_init,
        use_repo=config.use_repo,
        repo_after_n_layers=config.repo_after_n_layers,
        use_prores=config.use_prores,
        prores_T=config.prores_T,
        activation_checkpointing=config.activation_checkpointing,
        activation_checkpointing_mode=str(
            getattr(config, "activation_checkpointing_mode", "layer")
        )
        .strip()
        .lower(),
        use_mhc_lite=config.use_mhc_lite,
        mhc_n_streams=config.mhc_n_streams,
        mhc_triton_fused=config.mhc_triton_fused,
        mhc_lite_wrapping_level=str(config.mhc_lite_wrapping_level).strip().lower(),
        tie_word_embeddings=config.tie_word_embeddings,
        use_diff_attn_v2=config.use_diff_attn_v2,
        use_paired_head_attention=config.use_paired_head_attention,
        attn_layer_pattern=config.attn_layer_pattern,
        fused_qkv=config.fused_qkv,
        fused_up_gate=config.fused_up_gate,
        wo_init=config.wo_init,
        # NOBLE
        use_noble=config.use_noble,
        noble_rank=config.noble_rank,
        noble_alpha=config.noble_alpha,
        noble_omega_range=config.noble_omega_range,
        noble_phi_std=config.noble_phi_std,
        noble_half_kaiming=config.noble_half_kaiming,
        noble_targets=config.noble_targets,
        use_moe=config.use_moe,
        moe_num_experts=config.moe_num_experts,
        moe_top_k=config.moe_top_k,
        moe_use_bias_correction=config.moe_use_bias_correction,
        moe_aux_loss_coef=config.moe_aux_loss_coef,
        moe_z_loss_coef=config.moe_z_loss_coef,
        moe_routed_scaling_factor=config.moe_routed_scaling_factor,
        moe_n_group=config.moe_n_group,
        moe_topk_group=config.moe_topk_group,
        moe_bias_update_rate=config.moe_bias_update_rate,
        moe_leading_dense_layers=config.moe_leading_dense_layers,
        moe_kernel_backend=config.moe_kernel_backend,
        use_huginn_looping=config.use_huginn_looping,
        recycling_num_prelude_layers=config.recycling_num_prelude_layers,
        recycling_num_coda_layers=config.recycling_num_coda_layers,
        recycling_loop_count=config.recycling_loop_count,
        recycling_adapter_bias=config.recycling_adapter_bias,
        recycling_s0_noise_sigma=config.recycling_s0_noise_sigma,
        recycling_use_takase_init=config.recycling_use_takase_init,
        use_loopie=config.use_loopie,
        loopie_loop_count=config.loopie_loop_count,
    )


class PureProtModernBertMLM(ModernBertForMaskedLM):
    """Pure-torch ``ProtModernBertMLM`` (no HF ``transformers`` dependency)."""

    def __init__(self, config: ProtModernBertMLMConfig):
        self.tokenizer = ProtModernBertTokenizer(
            num_register_tokens=config.num_register_tokens
        )
        # Keep the original high-level config for checkpoint serialization.
        self.model_config = config

        super().__init__(modernbert_config_from(config))
