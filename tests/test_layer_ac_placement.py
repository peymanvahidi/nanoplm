"""Where layer-mode activation checkpointing sits, and what that costs.

``activation_checkpointing_mode: "layer"`` used to be applied from the encoder
loop in ``ModernBertModel.forward``, i.e. around ``layer(...)`` — around the
module call that FSDP2 has hooked. Those hooks are ``torch._dynamo.disable``d,
so the checkpoint body could not be traced: Dynamo hit "Attempted to call
function marked as skipped / Higher Order Operator:
torch.utils.checkpoint.checkpoint", could not resume, and dropped the whole
layer to eager. The eager recompute then re-materialised ~15 layers at once
(77 GB at micro-batch 128, R=3; experiments/loop-imp-v4/NOTES.md, Step 3 / R1).

``ModernBertEncoderLayer.forward`` now opens the checkpoint itself, inside the
FSDP unit — FSDP(AC(layer)) — and the encoder-level wrapper is skipped for that
block type only. The invariants below are the ones that keep that true:

* the recompute must not re-enter ``layer.__call__`` (that is what "inside the
  unit" means, and it needs no GPU to check);
* compiling with AC on must produce the same graph count as with it off, with
  no graph break naming the checkpoint HOP;
* loss and gradients must be unchanged, with full and truncated BPTT;
* block types that do *not* checkpoint themselves (MHCLiteBlock) must keep the
  encoder-level wrapper.

``dynamic=False`` matches production (``use_packing`` + ``use_static_inp_size``
resolve ``compile_dynamic`` to False in ``pure_pipeline``).
"""

import inspect

import pytest
import torch
import torch._dynamo
from torch._dynamo.utils import counters

from nanoplm.pretraining.models.modern_bert import modeling
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM

_NUM_LAYERS = 2
_VOCAB = 32

_AC_ON = dict(activation_checkpointing=True, activation_checkpointing_mode="layer")
_AC_OFF = dict(activation_checkpointing=False)


def _cfg(**overrides):
    kwargs = dict(
        vocab_size=_VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=4,
        num_kv_heads=4,
        attn_layer_pattern="F",
        wo_init="kaiming",
        no_mlp_on_first_layer=False,
    )
    kwargs.update(overrides)
    return ProtModernBertMLMConfig(**kwargs)


def _loopie_model(loop_count=2, **overrides):
    """A small looped model, identical every call (seeded), with wo != 0."""
    torch.manual_seed(0)
    model = PureProtModernBertMLM(
        _cfg(use_loopie=True, loopie_loop_count=loop_count, **overrides)
    )
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    model.train()
    return model


def _batch(batch=2, seq=8):
    input_ids = torch.randint(0, _VOCAB, (batch, seq))
    return input_ids, torch.ones_like(input_ids), input_ids.clone()


# `use_moe: true` is the production loop-imp geometry (16 of 20 layers), and
# MoELayer.forward rebinds `last_expert_counts` / `last_aux_loss` / `last_z_loss`
# / `last_aux_count` on itself. Inside the checkpoint HOP those rebinds are an
# "Unsafe side effect" unless the pipeline sets the Dynamo flag below, so the MoE
# arm is parametrized here and the flag is set exactly as `pure_pipeline`'s
# `_maybe_allow_side_effects_under_checkpoint` sets it.
_MOE = dict(
    use_moe=True,
    moe_num_experts=4,
    moe_top_k=2,
    moe_leading_dense_layers=1,
    mlp_activation="swiglu",
)
_ARCH = {"dense": {}, "moe": _MOE}


@pytest.fixture
def side_effect_flag():
    """Set/restore ``skip_fwd_side_effects_in_bwd_under_checkpoint``."""
    import torch._dynamo.config as dynamo_config

    previous = dynamo_config.skip_fwd_side_effects_in_bwd_under_checkpoint

    def _set(value: bool) -> None:
        dynamo_config.skip_fwd_side_effects_in_bwd_under_checkpoint = value

    yield _set
    dynamo_config.skip_fwd_side_effects_in_bwd_under_checkpoint = previous


@pytest.fixture
def count_checkpoints(monkeypatch):
    """Count the ``torch.utils.checkpoint.checkpoint`` calls modeling makes."""
    calls = []
    original = modeling._checkpoint

    def counting(fn, *args, **kwargs):
        calls.append(fn)
        return original(fn, *args, **kwargs)

    monkeypatch.setattr(modeling, "_checkpoint", counting)
    return calls


@pytest.mark.parametrize("arch", list(_ARCH), ids=list(_ARCH))
@pytest.mark.parametrize("r", [2, 3], ids=["R2", "R3"])
def test_layer_ac_adds_no_graphs_and_no_checkpoint_break(
    make_counting_backend, side_effect_flag, r, arch
):
    """AC on must compile into the same graphs as AC off, HOP break-free.

    The count is the load-bearing part: an un-resumable break at the checkpoint
    (what the encoder-level placement produced under FSDP2) splits the frame
    and shows up here as a different number of graphs and a new break reason.
    """
    side_effect_flag(True)
    input_ids, attention_mask, labels = _batch()

    counts = {}
    breaks = {}
    for name, flags in (("off", _AC_OFF), ("on", _AC_ON)):
        model = _loopie_model(loop_count=r, **_ARCH[arch], **flags)
        backend, n_graphs = make_counting_backend()
        torch._dynamo.reset()
        counters.clear()
        compiled = torch.compile(model, backend=backend, dynamic=False)
        compiled(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            loop_count=r,
        ).loss.backward()
        counts[name] = n_graphs[0]
        breaks[name] = dict(counters["graph_break"])

    assert counts["off"] > 0, "expected at least one compile without AC"
    assert counts["on"] == counts["off"], (
        f"layer AC compiled into {counts['on']} graphs vs {counts['off']} with it "
        "off; the checkpoint split the trace instead of becoming one HOP region."
    )
    assert breaks["on"] == breaks["off"], (
        "layer AC changed the graph breaks:\n"
        f"off: {sorted(breaks['off'])}\non: {sorted(breaks['on'])}"
    )
    offenders = [
        reason
        for reason in breaks["on"]
        if "checkpoint" in reason.lower() or "Higher Order Operator" in reason
    ]
    assert not offenders, f"the checkpoint HOP broke the graph:\n{offenders}"


def test_moe_needs_the_side_effect_flag_to_keep_the_hop(
    make_counting_backend, side_effect_flag
):
    """Without the pipeline's Dynamo flag, a MoE layer loses its HOP.

    This is the failure mode `_maybe_allow_side_effects_under_checkpoint` exists
    for, asserted so the test above cannot pass for the wrong reason: the MoE
    layer's telemetry rebinds break the HOP un-resumably, the layer falls back to
    eager and the graph count goes up.
    """
    side_effect_flag(False)
    model = _loopie_model(**_MOE, **_AC_ON)
    backend, n_graphs = make_counting_backend()
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(model, backend=backend, dynamic=False)
    input_ids, attention_mask, labels = _batch()
    compiled(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loop_count=2,
    ).loss.backward()

    reasons = [r for r in counters["graph_break"] if "Unsafe side effect" in r]
    assert reasons, (
        "expected the HOP side-effect break without the flag; got "
        f"{sorted(counters['graph_break'])}"
    )
    assert n_graphs[0] > 0


def test_moe_expert_counts_are_not_double_counted_under_ac(side_effect_flag):
    """Skipping the backward replay is also the semantics the bias update wants.

    ``last_expert_counts`` drives the Ling-style expert-bias nudge
    (``pure_pipeline._moe_count_accum``) and is read after the backward, so a
    replay during recompute would count every routed token twice.
    """
    side_effect_flag(True)
    r = 2
    batch, seq, top_k = 2, 8, _MOE["moe_top_k"]
    input_ids = torch.randint(0, _VOCAB, (batch, seq))
    attention_mask = torch.ones_like(input_ids)
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids.clone(),
        loop_count=r,
    )

    counts = {}
    for name, flags in (("off", _AC_OFF), ("on", _AC_ON)):
        model = _loopie_model(loop_count=r, **_MOE, **flags)
        torch._dynamo.reset()
        compiled = torch.compile(model, backend="aot_eager", dynamic=False)
        compiled(**kwargs).loss.backward()
        counts[name] = [
            layer.mlp.last_expert_counts
            for layer in model.model.layers
            if type(layer.mlp).__name__ == "MoELayer"
        ]

    assert counts["off"], "the test config produced no MoE layers"
    assert len(counts["on"]) == len(counts["off"])
    expected_total = batch * seq * top_k * r
    for got, want in zip(counts["on"], counts["off"]):
        assert got is not None and want is not None
        assert int(want.sum()) == expected_total, (
            f"reference itself is off: {int(want.sum())} != {expected_total}"
        )
        assert int(got.sum()) == expected_total, (
            f"layer AC counted {int(got.sum())} assignments instead of "
            f"{expected_total}; the recompute replayed the telemetry rebind"
        )


def test_layer_ac_becomes_one_checkpoint_hop_per_layer():
    """The checkpoint must reach the backend as a HOP, not as a graph break.

    ``tag_activation_checkpoint`` in the Dynamo graph is what lets AOTAutograd's
    min-cut partitioner own the recompute (the GPU-side acceptance criterion:
    partitioner reports for the layer graphs, no checkpoint-HOP break). A break
    instead of a HOP is the failure this step exists to fix.
    """
    targets = []

    def backend(gm, example_inputs):
        targets.extend(str(node.target) for node in gm.graph.nodes)
        return gm.forward

    model = _loopie_model(**_AC_ON)
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend, dynamic=False)
    input_ids, attention_mask, labels = _batch()
    compiled(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loop_count=2,
    ).loss.backward()

    hops = [t for t in targets if t == "tag_activation_checkpoint"]
    assert len(hops) == _NUM_LAYERS, (
        f"expected one checkpoint HOP per stored layer in the traced graphs; "
        f"got {len(hops)} (all HOP-ish targets: "
        f"{sorted({t for t in targets if 'checkpoint' in t})})"
    )


@pytest.mark.parametrize("bptt", [None, 1], ids=["full_bptt", "bptt1"])
def test_layer_ac_matches_no_ac_in_loss_and_grads(bptt):
    """Recompute must reproduce the forward exactly, bptt segments included.

    The truncated-BPTT parametrization is the interesting one: the first
    applications run under ``torch.no_grad()`` inside the checkpointed body and
    are stitched back with ``x = h + (x - x.detach())``, so the recompute has to
    re-enter that no_grad block and rebuild the same identity path.
    """
    r = 3
    input_ids, attention_mask, labels = _batch()
    plain = _loopie_model(loop_count=r, loopie_bptt=bptt, **_AC_OFF)
    checkpointed = _loopie_model(loop_count=r, loopie_bptt=bptt, **_AC_ON)

    ref = plain.state_dict()
    for key, value in checkpointed.state_dict().items():
        torch.testing.assert_close(
            value, ref[key], msg=lambda m, _k=key: f"test setup, {_k}: {m}"
        )

    torch._dynamo.reset()
    compiled_plain = torch.compile(plain, backend="aot_eager", dynamic=False)
    compiled_ckpt = torch.compile(checkpointed, backend="aot_eager", dynamic=False)

    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loop_count=r,
    )
    loss_plain = compiled_plain(**kwargs).loss
    loss_plain.backward()
    loss_ckpt = compiled_ckpt(**kwargs).loss
    loss_ckpt.backward()

    torch.testing.assert_close(loss_ckpt, loss_plain, rtol=1e-5, atol=1e-7)
    grads = dict(plain.named_parameters())
    n_compared = 0
    for name, param in checkpointed.named_parameters():
        want = grads[name].grad
        if want is None:
            assert param.grad is None, f"{name}: AC produced a grad where none was"
            continue
        assert param.grad is not None, f"{name}: AC dropped the gradient"
        torch.testing.assert_close(
            param.grad,
            want,
            rtol=1e-5,
            atol=1e-7,
            msg=lambda m, _n=name: f"grad {_n}: {m}",
        )
        n_compared += 1
    assert n_compared > 0, "no gradients were compared"


@pytest.mark.parametrize("bptt", [None, 1], ids=["full_bptt", "bptt1"])
def test_layer_ac_matches_no_ac_eagerly(bptt):
    """Same equality without torch.compile: the recompute path taken on CPU.

    Eager recompute runs the body a second time, so this also covers the
    non-reentrant checkpoint's RNG and no_grad bookkeeping, which the HOP
    version of the test above never executes.
    """
    r = 3
    input_ids, attention_mask, labels = _batch()
    plain = _loopie_model(loop_count=r, loopie_bptt=bptt, **_AC_OFF)
    checkpointed = _loopie_model(loop_count=r, loopie_bptt=bptt, **_AC_ON)
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loop_count=r,
    )
    loss_plain = plain(**kwargs).loss
    loss_plain.backward()
    loss_ckpt = checkpointed(**kwargs).loss
    loss_ckpt.backward()

    torch.testing.assert_close(loss_ckpt, loss_plain, rtol=1e-5, atol=1e-7)
    want = dict(plain.named_parameters())
    for name, param in checkpointed.named_parameters():
        if want[name].grad is None:
            continue
        torch.testing.assert_close(
            param.grad,
            want[name].grad,
            rtol=1e-5,
            atol=1e-7,
            msg=lambda m, _n=name: f"grad {_n}: {m}",
        )


def test_both_encoder_branches_share_the_gate():
    """The varlen/packed and SDPA loops must agree on when to wrap a block.

    Production runs the varlen branch, which needs flash-attention and cannot be
    exercised on CPU, so the guard here is structural: both branches call the
    same predicate with the same arguments.
    """
    source = inspect.getsource(modeling.ModernBertModel.forward)
    assert source.count("_block_owns_layer_checkpoint(layer, loop_count)") == 2, (
        "expected the layer-mode gate in both the varlen and the SDPA encoder "
        "loop; one branch was edited without the other"
    )


def test_recompute_stays_inside_the_module_call(count_checkpoints):
    """The FSDP(AC(layer)) ordering, checkable without a process group.

    FSDP2 all-gathers in a forward pre-hook on ``layer.__call__``. If the
    checkpoint wrapped that call, the backward recompute would enter it a second
    time — which is exactly what put FSDP's dynamo-disabled hooks inside the HOP
    body. One ``__call__`` and two body passes per layer is the invariant.
    """
    r = 2
    model = _loopie_model(loop_count=r, **_AC_ON)
    calls: dict[int, int] = {}
    bodies: dict[int, int] = {}
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_pre_hook(
            lambda module, args, _i=i: calls.__setitem__(_i, calls.get(_i, 0) + 1)
        )
        original = layer._forward_once

        def counted(*args, _original=original, _i=i, **kwargs):
            bodies[_i] = bodies.get(_i, 0) + 1
            return _original(*args, **kwargs)

        layer._forward_once = counted

    input_ids, attention_mask, labels = _batch()
    model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loop_count=r,
    ).loss.backward()

    assert len(count_checkpoints) == _NUM_LAYERS, (
        f"expected one checkpoint per stored layer; got {len(count_checkpoints)}"
    )
    assert calls == {i: 1 for i in range(_NUM_LAYERS)}, (
        f"each stored layer must be entered once per step (one all-gather, one "
        f"reshard), recompute included; got {calls}"
    )
    assert bodies == {i: 2 * r for i in range(_NUM_LAYERS)}, (
        f"expected {r} forward and {r} recomputed applications per layer, i.e. "
        f"the checkpoint is live; got {bodies}"
    )


def test_eval_mode_opens_no_checkpoint(count_checkpoints):
    """In-loop evaluation must not pay recompute (the `self.training` guard)."""
    model = _loopie_model(**_AC_ON)
    model.eval()
    input_ids, attention_mask, _ = _batch()
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask, loop_count=2)
    assert not count_checkpoints, (
        f"eval-mode forward opened {len(count_checkpoints)} checkpoints"
    )


def test_tensor_loop_count_keeps_the_encoder_level_wrapper(count_checkpoints):
    """The eager tensor-R driver does not checkpoint, so the encoder must.

    ``_forward_repeats`` is dynamo-disabled and has no checkpoint of its own;
    dropping the encoder-level wrapper for it would silently turn AC off.
    """
    model = _loopie_model(**_AC_ON)
    input_ids, attention_mask, labels = _batch()
    model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loop_count=torch.tensor(2, dtype=torch.int64),
    ).loss.backward()
    assert len(count_checkpoints) == _NUM_LAYERS, (
        f"expected one encoder-level checkpoint per layer on the tensor-R path; "
        f"got {len(count_checkpoints)}"
    )


def _mhc_config(**overrides):
    kwargs = dict(
        vocab_size=_VOCAB,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=4,
        use_mhc_lite=True,
        mhc_n_streams=3,
        mhc_triton_fused=False,
        mhc_lite_wrapping_level="layer",
        use_canon_layers=False,
        use_repo=False,
    )
    kwargs.update(overrides)
    return modeling.ModernBertConfig(**kwargs)


def test_mhc_lite_block_keeps_the_encoder_level_wrapper(count_checkpoints):
    """MHCLiteBlock has no internal layer-mode AC, so the encoder keeps wrapping.

    And the layer it wraps must have opted out, or the two checkpoints would
    nest and the layer body would be recomputed twice.
    """
    cfg = _mhc_config(**_AC_ON)
    model = modeling.ModernBertForMaskedLM(cfg)
    model.train()

    block = model.model.layers[0]
    assert isinstance(block, modeling.MHCLiteBlock)
    assert modeling._block_owns_layer_checkpoint(block) is False
    assert block.layer._layer_ac_internal is False
    assert block.layer._use_internal_layer_ac() is False

    input_ids = torch.randint(0, _VOCAB, (2, 8), dtype=torch.long)
    attention_mask = torch.ones((2, 8), dtype=torch.long)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids.clone(),
    )
    out.loss.backward()

    assert torch.isfinite(out.loss), "mHC-lite forward under layer AC diverged"
    assert len(count_checkpoints) == _NUM_LAYERS, (
        f"expected one encoder-level checkpoint per mHC-lite block; got "
        f"{len(count_checkpoints)}"
    )
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    ), "no finite gradients came back from the mHC-lite path"


def test_plain_encoder_layer_owns_its_checkpoint():
    """The predicate the encoder loop keys on, read straight off the layer."""
    model = _loopie_model(**_AC_ON)
    layer = model.model.layers[0]
    assert isinstance(layer, modeling.ModernBertEncoderLayer)
    assert modeling._block_owns_layer_checkpoint(layer) is True
    # A tensor trip count leaves `forward` for the eager driver, which does not
    # checkpoint, so the encoder-level wrapper has to stay for it.
    assert (
        modeling._block_owns_layer_checkpoint(layer, torch.tensor(2)) is False
    )
    model.eval()
    assert modeling._block_owns_layer_checkpoint(layer) is False

    off = _loopie_model(**_AC_OFF)
    assert modeling._block_owns_layer_checkpoint(off.model.layers[0]) is False


def test_layer_ac_changes_no_state_dict_keys():
    """Checkpointing inside ``forward`` adds no module, so nothing to strip.

    The wrapper alternative (``checkpoint_wrapper`` before ``fully_shard``)
    would insert ``_checkpoint_wrapped_module`` in every parameter path and rely
    on its prefix stripping; this placement cannot touch the keys at all, so
    existing checkpoints stay loadable either way.
    """
    off = _loopie_model(**_AC_OFF)
    on = _loopie_model(**_AC_ON)
    assert list(on.state_dict()) == list(off.state_dict())
    assert [n for n, _ in on.named_buffers()] == [n for n, _ in off.named_buffers()]
    on.load_state_dict(off.state_dict(), strict=True)


def test_attn_mode_checkpointing_is_untouched(count_checkpoints):
    """``attn`` mode still checkpoints branches inside ``_forward_once`` only."""
    r = 2
    model = _loopie_model(
        loop_count=r,
        activation_checkpointing=True,
        activation_checkpointing_mode="attn",
    )
    layer = model.model.layers[0]
    assert modeling._block_owns_layer_checkpoint(layer) is False

    input_ids, attention_mask, labels = _batch()
    model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loop_count=r,
    ).loss.backward()
    # One attention-branch checkpoint per application per layer, and nothing
    # around the layer or the loop.
    assert len(count_checkpoints) == _NUM_LAYERS * r, (
        f"expected {_NUM_LAYERS * r} attention-branch checkpoints; got "
        f"{len(count_checkpoints)}"
    )
