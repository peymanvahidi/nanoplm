"""Tests for ALBERT-style weight-tied looping (num_loops > 1).

The looped encoder applies the same L-layer stack T times with tied weights.
These tests verify:
  1. num_loops=1 is bit-identical to the standard (non-looped) forward pass.
  2. A looped forward is deterministic across repeated invocations
     (no leaked state between loop iterations).
  3. ModernBertConfig.__post_init__ guards incompatible feature combinations.
"""

import pytest
import torch

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.modeling import ModernBertConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM


def _base_cfg(tokenizer, **overrides):
    kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        vocab_size=tokenizer.vocab_size,
    )
    kwargs.update(overrides)
    return ProtModernBertMLMConfig(**kwargs)


def _build_model(tokenizer, seed=0, **cfg_overrides):
    torch.manual_seed(seed)
    return PureProtModernBertMLM(_base_cfg(tokenizer, **cfg_overrides)).eval()


def test_num_loops_default_is_one(tokenizer):
    """Default num_loops on ProtModernBertMLMConfig is 1."""
    cfg = _base_cfg(tokenizer)
    assert cfg.num_loops == 1


def test_num_loops_one_is_identity(tokenizer):
    """num_loops=1 must be bit-identical to the baseline (default) model."""
    m_default = _build_model(tokenizer, seed=0)
    m_loop1 = _build_model(tokenizer, seed=0, num_loops=1)

    ids = torch.randint(5, tokenizer.vocab_size, (2, 16))
    am = torch.ones_like(ids)
    with torch.no_grad():
        out_default = m_default(input_ids=ids, attention_mask=am)
        out_loop1 = m_loop1(input_ids=ids, attention_mask=am)

    diff = (out_default["logits"] - out_loop1["logits"]).abs().max().item()
    assert diff == 0.0, (
        f"num_loops=1 must be bit-identical to baseline; max abs diff = {diff}"
    )


def test_looped_forward_is_deterministic(tokenizer):
    """A num_loops=2 model must return the same logits on repeated forward
    calls with the same input. Catches state leaks (e.g. a buffer mutated
    during the loop body that affects the next call)."""
    m = _build_model(tokenizer, seed=0, num_loops=2)
    ids = torch.randint(5, tokenizer.vocab_size, (2, 16))
    am = torch.ones_like(ids)
    with torch.no_grad():
        a = m(input_ids=ids, attention_mask=am)["logits"]
        b = m(input_ids=ids, attention_mask=am)["logits"]
    diff = (a - b).abs().max().item()
    assert diff == 0.0, (
        f"Looped forward leaked state between calls; max abs diff = {diff}"
    )


def test_looped_differs_from_single_loop(tokenizer):
    """Sanity check: num_loops=2 produces different logits than num_loops=1
    with otherwise-identical config and init. Confirms the outer loop is
    actually re-running the stack rather than collapsing to a no-op.

    Note: must override wo_init from the default "zero" — with zero-initialized
    out-projections (attn.Wo, mlp.Wo) every layer is a pure identity at random
    init, so num_loops=1 and num_loops=2 trivially agree."""
    m1 = _build_model(tokenizer, seed=0, num_loops=1, wo_init="kaiming")
    m2 = _build_model(tokenizer, seed=0, num_loops=2, wo_init="kaiming")
    ids = torch.randint(5, tokenizer.vocab_size, (2, 16))
    am = torch.ones_like(ids)
    with torch.no_grad():
        a = m1(input_ids=ids, attention_mask=am)["logits"]
        b = m2(input_ids=ids, attention_mask=am)["logits"]
    diff = (a - b).abs().max().item()
    assert diff > 0.0, "num_loops=2 must produce different logits than num_loops=1"


def test_invalid_num_loops_rejected(tokenizer):
    """ModernBertConfig.__post_init__ rejects num_loops < 1 and non-int."""
    with pytest.raises(ValueError, match="num_loops must be int"):
        ModernBertConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_loops=0,
        )
    with pytest.raises(ValueError, match="num_loops must be int"):
        ModernBertConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_loops=-1,
        )


def test_moe_with_looping_rejected(tokenizer):
    """use_moe=True with num_loops>1 must raise (last_aux_loss /
    last_expert_counts are overwritten per forward call)."""
    with pytest.raises(ValueError, match="use_moe=True with num_loops>1"):
        ModernBertConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_loops=2,
            use_moe=True,
        )


def test_x0_lambdas_with_looping_rejected(tokenizer):
    """use_x0_lambdas=True with num_loops>1 must raise (x0 re-injection
    semantics under looping are not validated)."""
    with pytest.raises(ValueError, match="use_x0_lambdas=True with num_loops>1"):
        ModernBertConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_loops=2,
            use_x0_lambdas=True,
        )


def test_resid_lambdas_with_looping_rejected(tokenizer):
    """use_resid_lambdas=True with num_loops>1 must raise."""
    with pytest.raises(ValueError, match="use_resid_lambdas=True with num_loops>1"):
        ModernBertConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_loops=2,
            use_resid_lambdas=True,
        )
