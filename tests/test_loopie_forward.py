"""Tests for Loopie layer-loop recurrence (arXiv 2607.16051).

Each stored layer is applied ``loopie_loop_count`` times consecutively
(L1,L1,L2,L2,...), weight-tied. These tests cover the default full backprop;
``loopie_bptt`` truncation is tested in ``test_loopie_bptt.py``.
"""

import pytest
import torch

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM


def _cfg(**kwargs):
    base = dict(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=4,
        attn_layer_pattern="F",
        # non-zero Wo, else every layer is the identity at init and the
        # "repeat actually happened" assertions become vacuous.
        wo_init="kaiming",
        no_mlp_on_first_layer=False,
    )
    base.update(kwargs)
    return ProtModernBertMLMConfig(**base)


def _perturb(model):
    """Push every parameter off its init so no layer is accidentally identity."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)


def test_layer_repeats_reflects_config():
    plain = PureProtModernBertMLM(_cfg())
    assert all(l.repeats == 1 for l in plain.model.layers)
    m = PureProtModernBertMLM(_cfg(use_loopie=True, loopie_loop_count=3))
    assert all(l.repeats == 3 for l in m.model.layers)


def test_loopie_adds_no_parameters():
    """Weight tying means a Loopie model has exactly the plain model's weights."""
    plain = PureProtModernBertMLM(_cfg())
    loopie = PureProtModernBertMLM(_cfg(use_loopie=True, loopie_loop_count=2))
    assert set(plain.state_dict()) == set(loopie.state_dict())
    assert loopie.model.adapter is None  # no Huginn adapter


def test_loopie_matches_manual_double_application():
    torch.manual_seed(0)
    model = PureProtModernBertMLM(_cfg(use_loopie=True, loopie_loop_count=2)).eval()
    _perturb(model)
    enc = model.model
    input_ids = torch.randint(0, 32, (2, 8))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        got = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Reference: drive the same layer modules by hand, twice each.
        x = enc.embeddings(input_ids)
        cos_sin = enc.rotary_emb.forward_full(input_ids.shape[1], x.device, x.dtype)
        for layer in enc.layers:
            for _ in range(2):
                x = layer._forward_once(
                    x, attn_mask=None, cos_sin=cos_sin, position_ids=None,
                    token_mask=attention_mask, repo_active=False, prores_alpha=1.0)
        want = model.decoder(model.head(enc.final_norm(x)))

    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_loopie_differs_from_single_pass():
    """Guards against the repeat silently not happening."""
    torch.manual_seed(0)
    model = PureProtModernBertMLM(_cfg(use_loopie=True, loopie_loop_count=2)).eval()
    _perturb(model)
    enc = model.model
    input_ids = torch.randint(0, 32, (2, 8))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        looped = model(input_ids=input_ids, attention_mask=attention_mask).logits
        x = enc.embeddings(input_ids)
        cos_sin = enc.rotary_emb.forward_full(input_ids.shape[1], x.device, x.dtype)
        for layer in enc.layers:
            x = layer._forward_once(
                x, attn_mask=None, cos_sin=cos_sin, position_ids=None,
                token_mask=attention_mask, repo_active=False, prores_alpha=1.0)
        once = model.decoder(model.head(enc.final_norm(x)))

    assert (looped - once).abs().max() > 1e-4


def test_loopie_backprops_through_every_repeat():
    torch.manual_seed(0)
    model = PureProtModernBertMLM(_cfg(use_loopie=True, loopie_loop_count=2))
    _perturb(model)
    input_ids = torch.randint(0, 32, (2, 8))
    out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                labels=input_ids.clone())
    out.loss.backward()
    for i, layer in enumerate(model.model.layers):
        grads = [p.grad for p in layer.parameters() if p.grad is not None]
        assert grads, f"layer {i} received no gradients"
        assert any(g.abs().sum() > 0 for g in grads), f"layer {i} gradients are all zero"


@pytest.mark.parametrize("repeats", [2, 3])
def test_loopie_flops_scale_with_loop_count(repeats):
    """MFU accounting must charge R x the stack, with full (2x) backward."""
    from nanoplm.pretraining.pure_pipeline import (
        _estimate_model_flops_per_token,
        _recycling_flops_components,
        _recycling_flops_per_token,
    )
    from nanoplm.pretraining.models.modern_bert.modeling import ModernBertConfig

    seq = 128
    mb = ModernBertConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                          num_attention_heads=4, vocab_size=32,
                          use_loopie=True, loopie_loop_count=repeats)
    comp = _recycling_flops_components(mb, seq)
    assert _estimate_model_flops_per_token(mb, seq) == _recycling_flops_per_token(
        comp, repeats, bptt=None
    )
