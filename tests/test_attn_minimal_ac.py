"""``activation_checkpointing_mode: "attn_minimal"``.

Same scope as ``attn`` (only the attention residual branch is checkpointed), but
with a selective-AC policy that saves the matmul outputs (Wqkv, Wo) and the
FlashAttention forward outputs, so the backward recomputes only the cheap
elementwise ops (norms, QK-norm, RoPE). The invariants:

* loss and gradients match no AC and plain ``attn``, eagerly and compiled;
* the backward runs fewer matmuls than ``attn`` (no projection GEMM is
  recomputed), and exactly as many as no AC;
* MoE accepts the mode (it rejects only ``attn+mlp``).
"""

import pytest
import torch
import torch._dynamo
from torch.utils._python_dispatch import TorchDispatchMode

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM

_VOCAB = 32


def _cfg(**overrides):
    kwargs = dict(
        vocab_size=_VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=4,
        attn_layer_pattern="F",
        wo_init="kaiming",
        no_mlp_on_first_layer=False,
    )
    kwargs.update(overrides)
    return ProtModernBertMLMConfig(**kwargs)


def _model(mode):
    """Identical weights for every mode (seeded), with wo != 0."""
    flags = (
        dict(activation_checkpointing=False)
        if mode is None
        else dict(activation_checkpointing=True, activation_checkpointing_mode=mode)
    )
    torch.manual_seed(0)
    model = PureProtModernBertMLM(_cfg(**flags))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    model.train()
    return model


def _kwargs(batch=2, seq=8):
    g = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, _VOCAB, (batch, seq), generator=g)
    return dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids.clone(),
    )


class _CountMM(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default):
            self.n += 1
        return func(*args, **(kwargs or {}))


def _assert_same_grads(model, ref):
    want = dict(ref.named_parameters())
    n = 0
    for name, param in model.named_parameters():
        if want[name].grad is None:
            assert param.grad is None, f"{name}: grad where the reference has none"
            continue
        assert param.grad is not None, f"{name}: gradient dropped"
        torch.testing.assert_close(
            param.grad,
            want[name].grad,
            rtol=1e-5,
            atol=1e-7,
            msg=lambda m, _n=name: f"grad {_n}: {m}",
        )
        n += 1
    assert n > 0, "no gradients were compared"


def test_attn_minimal_uses_a_selective_policy_only_in_that_mode():
    layers = lambda m: [l for l in m.modules() if hasattr(l, "_attn_sac")]
    assert all(l._attn_sac is not None for l in layers(_model("attn_minimal")))
    for mode in (None, "attn", "layer", "attn+mlp"):
        assert all(l._attn_sac is None for l in layers(_model(mode))), mode


@pytest.mark.parametrize("ref_mode", [None, "attn"], ids=["no_ac", "attn"])
def test_attn_minimal_matches_eagerly(ref_mode):
    kwargs = _kwargs()
    ref, sac = _model(ref_mode), _model("attn_minimal")
    loss_ref = ref(**kwargs).loss
    loss_ref.backward()
    loss_sac = sac(**kwargs).loss
    loss_sac.backward()
    torch.testing.assert_close(loss_sac, loss_ref, rtol=1e-5, atol=1e-7)
    _assert_same_grads(sac, ref)


def test_attn_minimal_matches_no_ac_compiled():
    kwargs = _kwargs()
    ref, sac = _model(None), _model("attn_minimal")
    torch._dynamo.reset()
    c_ref = torch.compile(ref, backend="aot_eager", dynamic=False)
    c_sac = torch.compile(sac, backend="aot_eager", dynamic=False)
    loss_ref = c_ref(**kwargs).loss
    loss_ref.backward()
    loss_sac = c_sac(**kwargs).loss
    loss_sac.backward()
    torch.testing.assert_close(loss_sac, loss_ref, rtol=1e-5, atol=1e-7)
    _assert_same_grads(sac, ref)


def test_attn_minimal_backward_recomputes_no_matmul():
    kwargs = _kwargs()
    counts = {}
    for mode in (None, "attn", "attn_minimal"):
        model = _model(mode)
        loss = model(**kwargs).loss
        with _CountMM() as c:
            loss.backward()
        counts[mode] = c.n
    assert counts["attn"] > counts[None], counts  # stock attn re-runs Wqkv/Wo
    assert counts["attn_minimal"] == counts[None], counts


def test_moe_accepts_attn_minimal():
    from nanoplm.pretraining.models.modern_bert.config import ModernBertConfig

    kw = dict(
        vocab_size=_VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        use_moe=True,
        activation_checkpointing=True,
    )
    ModernBertConfig(activation_checkpointing_mode="attn_minimal", **kw)
    with pytest.raises(ValueError, match="attn\\+mlp"):
        ModernBertConfig(activation_checkpointing_mode="attn+mlp", **kw)
