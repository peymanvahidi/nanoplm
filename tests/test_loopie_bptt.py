import pytest
import torch

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.modeling import ModernBertConfig
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
        wo_init="kaiming",
        no_mlp_on_first_layer=False,
    )
    base.update(kwargs)
    return ProtModernBertMLMConfig(**base)


def _perturb(model):
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)


def _grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


def test_every_stored_layer_learns_under_truncation():
    torch.manual_seed(0)
    model = PureProtModernBertMLM(
        _cfg(num_hidden_layers=3, use_loopie=True, loopie_loop_count=2, loopie_bptt=1)
    )
    _perturb(model)
    input_ids = torch.randint(0, 32, (2, 8))
    out = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids.clone(),
    )
    out.loss.backward()

    per_layer = [round(_grad_norm(l.parameters()), 4) for l in model.model.layers]
    untied_embedding_grad = model.model.embeddings.norm.weight.grad
    emb = None if untied_embedding_grad is None else round(float(untied_embedding_grad.norm()), 4)
    report = f"per-layer grad norms {per_layer}, untied embedding grad {emb}"

    assert all(g > 0 for g in per_layer), f"a stored layer got no gradient: {report}"
    assert untied_embedding_grad is not None, f"embeddings got no gradient at all: {report}"
    assert emb > 0, f"embeddings gradient is zero: {report}"


def _layer_and_inputs(loop_count=2, bptt=None):
    torch.manual_seed(0)
    model = PureProtModernBertMLM(
        _cfg(use_loopie=True, loopie_loop_count=loop_count, loopie_bptt=bptt)
    )
    _perturb(model)
    model.eval()
    enc = model.model
    torch.manual_seed(1234)
    input_ids = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        x = enc.embeddings(input_ids)
    x = x.detach().requires_grad_(True)
    cos_sin = enc.rotary_emb.forward_full(input_ids.shape[1], x.device, x.dtype)
    kw = dict(
        attn_mask=None, cos_sin=cos_sin, position_ids=None,
        token_mask=torch.ones_like(input_ids), repo_active=False, prores_alpha=1.0,
    )
    return enc.layers[0], x, kw


def _grads_and_reset(layer, x, out):
    out.pow(2).sum().backward()
    g = {n: p.grad.detach().clone() for n, p in layer.named_parameters() if p.grad is not None}
    assert g, "the layer received no parameter gradients at all"
    assert x.grad is not None, "no gradient reached the layer input"
    g["<input>"] = x.grad.detach().clone()
    layer.zero_grad(set_to_none=True)
    x.grad = None
    return g


def _max_diff(a, b):
    assert set(a) == set(b) and a
    return max((a[n] - b[n]).abs().max().item() for n in a)


def _reference_grads(layer, x, kw, frozen):
    """Gradients of the last application alone, stated without the re-attach.

    The ``frozen`` applications before it run once to produce its input and
    are then cut off: they contribute no parameter gradient, and in backward
    they are the identity, so dL/dx is dL/d(input of the last application).
    """
    h = x.detach()
    with torch.no_grad():
        for _ in range(frozen):
            h = layer._forward_once(h, **kw)
    h = h.detach().requires_grad_(True)
    loss = layer._forward_once(h, **kw).pow(2).sum()
    names, params = zip(*layer.named_parameters())
    grads = torch.autograd.grad(loss, [*params, h], allow_unused=True)
    want = {n: g for n, g in zip(names, grads[:-1]) if g is not None}
    want["<input>"] = grads[-1]
    return want


def test_bptt_1_matches_the_frozen_reattached_reference():
    layer, x, kw = _layer_and_inputs(loop_count=2, bptt=1)
    got = _grads_and_reset(layer, x, layer(x, **kw))
    want = _reference_grads(layer, x, kw, frozen=1)

    assert _max_diff(got, want) < 1e-6


def test_bptt_1_forward_is_bit_identical_but_gradients_are_not():
    trunc_layer, x, kw = _layer_and_inputs(loop_count=2, bptt=1)
    full_layer, full_x, full_kw = _layer_and_inputs(loop_count=2, bptt=None)
    for (n, p), (_, q) in zip(trunc_layer.named_parameters(), full_layer.named_parameters()):
        assert torch.equal(p, q), f"test precondition: {n} differs between the two builds"
    assert torch.equal(x, full_x)

    trunc_out = trunc_layer(x, **kw)
    full_out = full_layer(full_x, **full_kw)
    assert torch.equal(trunc_out, full_out), (
        "loopie_bptt changed the forward value; the re-attach must add exactly zero"
    )

    trunc = _grads_and_reset(trunc_layer, x, trunc_out)
    full = _grads_and_reset(full_layer, full_x, full_out)
    assert _max_diff(trunc, full) > 1e-6


def test_bptt_none_still_backprops_through_every_application():
    layer, x, kw = _layer_and_inputs(loop_count=2, bptt=None)
    got = _grads_and_reset(layer, x, layer(x, **kw))

    h = layer._forward_once(x, **kw)
    want = _grads_and_reset(layer, x, layer._forward_once(h, **kw))

    assert _max_diff(got, want) < 1e-6


def test_bptt_1_on_the_variable_depth_path():
    layer, x, kw = _layer_and_inputs(loop_count=2, bptt=1)
    got = _grads_and_reset(layer, x, layer(x, loop_count=3, **kw))
    want = _reference_grads(layer, x, kw, frozen=2)

    assert _max_diff(got, want) < 1e-6


def test_bptt_larger_than_the_executed_depth_is_full_backprop():
    layer, x, kw = _layer_and_inputs(loop_count=2, bptt=2)
    got = _grads_and_reset(layer, x, layer(x, loop_count=1, **kw))
    want = _grads_and_reset(layer, x, layer._forward_once(x, **kw))

    assert _max_diff(got, want) < 1e-6


def _mb_cfg(**kwargs):
    return ModernBertConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, vocab_size=32, use_loopie=True, loopie_loop_count=2,
        **kwargs,
    )


def test_bptt_shrinks_the_flops_estimate():
    from nanoplm.pretraining.pure_pipeline import (
        _estimate_model_flops_per_token,
        _recycling_flops_components,
        _recycling_flops_per_token,
    )

    seq = 128
    trunc = _mb_cfg(loopie_bptt=1)
    full = _mb_cfg(loopie_bptt=None)
    comp = _recycling_flops_components(trunc, seq)
    stack, tail, P, R, C, n_layers = comp
    assert (P, C, R) == (0, 0, n_layers), "loopie has no prelude or coda"

    def forward(applications):
        return applications * stack + tail

    assert _estimate_model_flops_per_token(trunc, seq) == _recycling_flops_per_token(
        comp, 2, bptt=1
    ) == forward(2) + 2 * forward(1)
    assert _estimate_model_flops_per_token(full, seq) == _recycling_flops_per_token(
        comp, 2, bptt=None
    ) == 3 * forward(2)


def test_config_validates_bptt():
    for bad in (0, -5, True, False, "abc", 1.0):
        with pytest.raises(ValueError, match="loopie_bptt"):
            _mb_cfg(loopie_bptt=bad)
    assert _mb_cfg(loopie_bptt=None).loopie_bptt is None
    assert _mb_cfg(loopie_bptt=1).loopie_bptt == 1
    assert _mb_cfg(loopie_bptt=3).loopie_bptt == 3


@pytest.mark.parametrize("bad", [-5, True, "abc"])
def test_config_validates_bptt_without_loopie(bad):
    with pytest.raises(ValueError, match="loopie_bptt"):
        ModernBertConfig(
            hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            num_attention_heads=4, vocab_size=32, loopie_bptt=bad,
        )


def test_yaml_true_is_rejected_not_read_as_1():
    import yaml

    from nanoplm.cli.pretrain import _load_model_config

    model_cfg = _load_model_config(yaml.safe_load(
        "vocab_size: 32\nhidden_size: 64\nintermediate_size: 64\n"
        "num_hidden_layers: 2\nnum_attention_heads: 4\n"
        "use_loopie: true\nloopie_bptt: true\n"
    ))
    with pytest.raises(ValueError, match="loopie_bptt"):
        PureProtModernBertMLM(model_cfg)


def test_bptt_without_loopie_is_rejected():
    with pytest.raises(ValueError, match="no effect"):
        PureProtModernBertMLM(_cfg(use_loopie=False, loopie_bptt=1))
