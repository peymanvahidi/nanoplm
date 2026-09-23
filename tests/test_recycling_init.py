"""Tests for Takase spike initialization applied to the recurrent block.

When recycling_use_takase_init=True AND recycling_loop_count>1:
  - recurrent block Wo (attention + MLP) gets std = sqrt(1/(5·h·l))
  - recurrent block Wqkv, Wi get std = sqrt(2/(5·h))
  - adapter Linear(2h, h) gets std = sqrt(2/(5·h))
  - prelude / coda layers are unchanged from wo_init (e.g., 'zero' stays zero)

l = P + r_bar * R + C where r_bar = max(1, recycling_loop_count - 1).
"""

import math

import torch

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM


def _cfg(tokenizer, **overrides):
    kwargs = dict(
        hidden_size=128, intermediate_size=256, num_hidden_layers=6,
        num_attention_heads=4, vocab_size=tokenizer.vocab_size,
        recycling_num_prelude_layers=2, recycling_num_coda_layers=2,
        use_huginn_looping=True,
        recycling_loop_count=4,  # r_bar = 3
        recycling_use_takase_init=True,
        wo_init="zero",
    )
    kwargs.update(overrides)
    return ProtModernBertMLMConfig(**kwargs)


def test_takase_effective_depth_formula(tokenizer):
    """l = P + r_bar * R + C  with r_bar = max(1, loop_count - 1)."""
    cfg = _cfg(tokenizer)
    P, C = cfg.recycling_num_prelude_layers, cfg.recycling_num_coda_layers
    L = cfg.num_hidden_layers
    R = L - P - C
    r_bar = max(1, cfg.recycling_loop_count - 1)
    expected_l = P + r_bar * R + C
    assert expected_l == 2 + 3 * 2 + 2 == 10


def test_recurrent_wo_std_matches_takase(tokenizer):
    cfg = _cfg(tokenizer)
    model = PureProtModernBertMLM(cfg)
    inner = model.model
    h = cfg.hidden_size
    P, C = cfg.recycling_num_prelude_layers, cfg.recycling_num_coda_layers
    L = cfg.num_hidden_layers
    R = L - P - C
    r_bar = max(1, cfg.recycling_loop_count - 1)
    l = P + r_bar * R + C
    expected_std = math.sqrt(1.0 / (5.0 * h * l))

    wo_attn = inner.recurrent_layers[0].attn.Wo.weight
    actual_std = wo_attn.std().item()
    assert actual_std < expected_std * 3, (
        f"recurrent attn Wo std {actual_std:.6f} too large; expected ≈ {expected_std:.6f}"
    )
    assert actual_std > expected_std / 3, (
        f"recurrent attn Wo std {actual_std:.6f} too small; expected ≈ {expected_std:.6f}"
    )


def test_prelude_wo_unchanged_by_takase_when_wo_init_zero(tokenizer):
    """Prelude/coda follow wo_init; with wo_init='zero', their Wo must stay zero."""
    cfg = _cfg(tokenizer, wo_init="zero")
    model = PureProtModernBertMLM(cfg)
    inner = model.model
    if len(inner.prelude_layers) > 0:
        wo_attn_prelude = inner.prelude_layers[0].attn.Wo.weight
        assert wo_attn_prelude.abs().sum().item() == 0.0


def test_adapter_init_matches_general_takase(tokenizer):
    cfg = _cfg(tokenizer)
    model = PureProtModernBertMLM(cfg)
    inner = model.model
    h = cfg.hidden_size
    expected_std = math.sqrt(2.0 / (5.0 * h))
    actual_std = inner.adapter.weight.std().item()
    assert actual_std < expected_std * 3
    assert actual_std > expected_std / 3


def test_takase_disabled_when_flag_off(tokenizer):
    """recycling_use_takase_init=False → recurrent Wo follows wo_init."""
    cfg = _cfg(tokenizer, recycling_use_takase_init=False, wo_init="zero")
    model = PureProtModernBertMLM(cfg)
    inner = model.model
    wo_attn = inner.recurrent_layers[0].attn.Wo.weight
    assert wo_attn.abs().sum().item() == 0.0


def test_recurrent_mlp_wo_std_matches_takase(tokenizer):
    """MLP Wo must get the same std_out spike as attention Wo."""
    cfg = _cfg(tokenizer)
    inner = PureProtModernBertMLM(cfg).model
    h, P, C, L = cfg.hidden_size, cfg.recycling_num_prelude_layers, cfg.recycling_num_coda_layers, cfg.num_hidden_layers
    l = P + max(1, cfg.recycling_loop_count - 1) * (L - P - C) + C
    expected_std = math.sqrt(1.0 / (5.0 * h * l))
    mlp_wo = inner.recurrent_layers[0].mlp.Wo.weight
    actual = mlp_wo.std().item()
    assert expected_std / 3 < actual < expected_std * 3, \
        f"recurrent mlp.Wo std {actual:.6f} vs expected {expected_std:.6f}"


def test_recurrent_attn_wqkv_std_matches_takase_general(tokenizer):
    """Attention Wqkv (fused) must get std_general."""
    cfg = _cfg(tokenizer)
    inner = PureProtModernBertMLM(cfg).model
    h = cfg.hidden_size
    expected_std = math.sqrt(2.0 / (5.0 * h))
    wqkv = inner.recurrent_layers[0].attn.Wqkv.weight
    actual = wqkv.std().item()
    assert expected_std / 3 < actual < expected_std * 3, \
        f"recurrent attn.Wqkv std {actual:.6f} vs expected {expected_std:.6f}"


def test_coda_wo_unchanged_by_takase(tokenizer):
    """Coda layers must NOT be Takase-touched; wo_init='zero' leaves their Wo at zero."""
    cfg = _cfg(tokenizer, wo_init="zero")
    inner = PureProtModernBertMLM(cfg).model
    if len(inner.coda_layers) > 0:
        assert inner.coda_layers[0].attn.Wo.weight.abs().sum().item() == 0.0
