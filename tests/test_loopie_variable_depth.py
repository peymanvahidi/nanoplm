"""Loopie refinement depth as a forward-time kwarg.

``loop_count=r`` must reproduce, bit for bit, a model *built* to apply every
stored layer r times. That equivalence is what lets one kwarg serve both
capabilities: the training loop samples r per micro-batch, and an eval run
sweeps r over a checkpoint trained at some other depth.
"""

import pytest
import torch

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM


def _cfg(**overrides):
    kwargs = dict(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=4,
        attn_layer_pattern="F",
        wo_init="kaiming",
        no_mlp_on_first_layer=False,
    )
    kwargs.update(overrides)
    return ProtModernBertMLMConfig(**kwargs)


def _perturb(model):
    """Push every parameter off its init so no layer is accidentally identity."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)


def _reference_at(depth, weights_from):
    # ModernBertConfig rejects loopie_loop_count < 2, so depth 1 is only
    # reachable as an unlooped model -- which is what one pass per layer means.
    cfg = _cfg() if depth == 1 else _cfg(use_loopie=True, loopie_loop_count=depth)
    model = PureProtModernBertMLM(cfg).eval()
    model.load_state_dict(weights_from.state_dict())
    return model


def _logits(model, input_ids, attention_mask, **kwargs):
    with torch.no_grad():
        return model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        ).logits


@pytest.mark.parametrize("as_tensor", [False, True])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_forward_depth_matches_a_model_built_at_that_depth(depth, as_tensor):
    torch.manual_seed(0)
    model = PureProtModernBertMLM(_cfg(use_loopie=True, loopie_loop_count=2)).eval()
    _perturb(model)
    reference = _reference_at(depth, model)

    input_ids = torch.randint(0, 32, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    # A Python int is what eager eval passes; a 0-d tensor is what the compiled
    # training loop passes.
    loop_count = torch.tensor(depth, dtype=torch.int64) if as_tensor else depth

    got = _logits(model, input_ids, attention_mask, loop_count=loop_count)
    want = _logits(reference, input_ids, attention_mask)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_depths_produce_different_outputs():
    """Guards the equality above against every depth collapsing to the same value."""
    torch.manual_seed(0)
    model = PureProtModernBertMLM(_cfg(use_loopie=True, loopie_loop_count=2)).eval()
    _perturb(model)

    input_ids = torch.randint(0, 32, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    by_depth = {
        r: _logits(model, input_ids, attention_mask, loop_count=r) for r in (1, 2, 3)
    }
    for a, b in ((1, 2), (2, 3), (1, 3)):
        assert (by_depth[a] - by_depth[b]).abs().max() > 1e-4, (
            f"depth {a} and depth {b} produced the same logits"
        )


def test_omitting_the_kwarg_keeps_the_configured_depth():
    torch.manual_seed(0)
    model = PureProtModernBertMLM(_cfg(use_loopie=True, loopie_loop_count=2)).eval()
    _perturb(model)

    input_ids = torch.randint(0, 32, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    torch.testing.assert_close(
        _logits(model, input_ids, attention_mask),
        _logits(model, input_ids, attention_mask, loop_count=2),
        rtol=0, atol=0,
    )


def test_backprops_through_every_repeat_at_a_forward_time_depth():
    torch.manual_seed(0)
    model = PureProtModernBertMLM(_cfg(use_loopie=True, loopie_loop_count=2))
    _perturb(model)

    input_ids = torch.randint(0, 32, (2, 8))
    out = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids.clone(),
        loop_count=torch.tensor(3, dtype=torch.int64),
    )
    out.loss.backward()
    for i, layer in enumerate(model.model.layers):
        grads = [p.grad for p in layer.parameters() if p.grad is not None]
        assert grads, f"layer {i} received no gradients"
        assert any(g.abs().sum() > 0 for g in grads), f"layer {i} gradients are all zero"
