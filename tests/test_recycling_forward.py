"""Tests for Huginn-style recycling forward pass and BPTT.

Covers:
  - Layer split into prelude/recurrent/coda ModuleLists
  - Adapter Linear(2h, h) module shape and bias settings
  - Forward output shape with recycling on/off
  - Truncated BPTT gradient window
  - Prelude gradient path (e not detached)
  - recycling_loop_count=1 produces identical output to non-recycled
"""

import torch

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM


def _recycling_cfg(tokenizer, **overrides):
    kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        vocab_size=tokenizer.vocab_size,
        recycling_num_prelude_layers=1,
        recycling_num_coda_layers=1,
        use_huginn_looping=True,
        recycling_loop_count=3,
        recycling_use_takase_init=False,  # off here for shape tests; tested in test_recycling_init.py
        wo_init="kaiming",  # non-zero Wo so gradients flow through attention blocks
    )
    kwargs.update(overrides)
    return ProtModernBertMLMConfig(**kwargs)


def test_layer_split_creates_three_module_lists(tokenizer):
    cfg = _recycling_cfg(tokenizer)
    model = PureProtModernBertMLM(cfg)
    inner = model.model
    assert len(inner.prelude_layers) == 1
    assert len(inner.recurrent_layers) == 4 - 1 - 1
    assert len(inner.coda_layers) == 1


def test_split_groupings_share_modules_with_self_layers(tokenizer):
    """The three groupings must reference the SAME module instances as self.layers;
    otherwise state_dict would carry duplicate keys."""
    cfg = _recycling_cfg(tokenizer)
    inner = PureProtModernBertMLM(cfg).model
    P, C, L = cfg.recycling_num_prelude_layers, cfg.recycling_num_coda_layers, cfg.num_hidden_layers
    assert inner.prelude_layers[0] is inner.layers[0]
    assert inner.recurrent_layers[0] is inner.layers[P]
    assert inner.coda_layers[0] is inner.layers[L - C]

    sd = inner.state_dict()
    assert not any(k.startswith("prelude_layers.") for k in sd), \
        "prelude_layers must not appear in state_dict (would duplicate self.layers keys)"
    assert not any(k.startswith("recurrent_layers.") for k in sd), \
        "recurrent_layers must not appear in state_dict (would duplicate self.layers keys)"
    assert not any(k.startswith("coda_layers.") for k in sd), \
        "coda_layers must not appear in state_dict (would duplicate self.layers keys)"


def test_adapter_none_raises_on_forward_kwarg_override(tokenizer, device):
    """Building with recycling_loop_count=1 leaves adapter=None; passing
    recycling_loop_count>1 via forward kwarg must raise a clean ValueError,
    not crash inside _recurrent_step with TypeError(NoneType not callable)."""
    cfg = _recycling_cfg(
        tokenizer,
        use_huginn_looping=False,
        recycling_loop_count=1,
        recycling_num_prelude_layers=0,
        recycling_num_coda_layers=0,
    )
    model = PureProtModernBertMLM(cfg).to(device)
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, 8), device=device)
    import pytest
    with pytest.raises(ValueError, match="adapter is None"):
        model(input_ids=input_ids, loop_count=4, recycling_bptt=2)


def test_adapter_module_shape_and_bias(tokenizer):
    cfg = _recycling_cfg(tokenizer)
    model = PureProtModernBertMLM(cfg)
    inner = model.model
    assert inner.adapter.in_features == 2 * cfg.hidden_size
    assert inner.adapter.out_features == cfg.hidden_size
    assert (inner.adapter.bias is None) == (not cfg.recycling_adapter_bias)


def test_forward_runs_and_output_shape(tokenizer, device):
    cfg = _recycling_cfg(tokenizer)
    model = PureProtModernBertMLM(cfg).to(device)
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert out.logits.shape == (batch_size, seq_len, tokenizer.vocab_size)
    assert out.loss.dim() == 0  # scalar


def test_recycling_off_produces_deterministic_forward(tokenizer, device):
    """recycling_loop_count=1 disables the recycling branch entirely.
    Forward must be deterministic across repeated calls (no leaked state, no
    surprise s_0 noise sampled when recycling is off).
    """
    torch.manual_seed(123)
    cfg_off = _recycling_cfg(
        tokenizer,
        recycling_loop_count=1,
        recycling_num_prelude_layers=0,
        recycling_num_coda_layers=0,
    )
    model_off = PureProtModernBertMLM(cfg_off).to(device)
    model_off.eval()  # disable dropout

    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)

    out1 = model_off(input_ids=input_ids, attention_mask=attention_mask).logits
    out2 = model_off(input_ids=input_ids, attention_mask=attention_mask).logits
    assert torch.equal(out1, out2), "recycling-off forward should be deterministic"


def test_prelude_receives_gradients_through_loop(tokenizer, device):
    """Critical bug-catch: e must not be detached. Prelude params must have non-zero grad
    after a backward pass even when most loop iterations are under no_grad (BPTT-truncated)."""
    cfg = _recycling_cfg(tokenizer, recycling_loop_count=4)
    model = PureProtModernBertMLM(cfg).to(device)
    inner = model.model

    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    out = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels,
        loop_count=4, recycling_bptt=2,
    )
    out.loss.backward()

    prelude_param = next(inner.prelude_layers[0].parameters())
    assert prelude_param.grad is not None
    assert prelude_param.grad.abs().sum().item() > 0.0, (
        "prelude parameter did not receive gradient — `e` is being detached somewhere"
    )


def test_bptt_window_limits_recurrent_grad_to_last_k_iters(tokenizer, device):
    """With recycling_loop_count=r and recycling_bptt=K, only the last K iterations
    of the recurrent block should be in the autograd graph. We verify this by
    checking grad magnitude scales with K, not with r."""
    cfg = _recycling_cfg(tokenizer, recycling_loop_count=8)
    model = PureProtModernBertMLM(cfg).to(device)
    inner = model.model
    recurrent_param = next(inner.recurrent_layers[0].parameters())

    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    grads = {}
    for K in (1, 4):
        if recurrent_param.grad is not None:
            recurrent_param.grad.zero_()
        torch.manual_seed(7)
        out = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            loop_count=8, recycling_bptt=K,
        )
        out.loss.backward()
        grads[K] = recurrent_param.grad.abs().sum().item()

    assert grads[4] > grads[1], (
        f"Expected grad with K=4 ({grads[4]}) > K=1 ({grads[1]}); BPTT window not working"
    )
