"""Compile-stability and FSDP2 invariants for Loopie stochastic depth.

Two constraints pull against each other here. The trip count must stay out of
every traced graph, or Dynamo specializes on it and each distinct R costs a
fresh trace (the failure ``ModernBertModel._drive_recurrence`` documents). And
the repeats must stay inside ``ModernBertEncoderLayer.forward``, or FSDP2 —
which shards per encoder layer — all-gathers once per repeat instead of once
per layer. Both live in this file because a fix for either can break the other.

``dynamic=False`` matches production: ``use_packing`` + ``use_static_inp_size``
resolve ``compile_dynamic`` to False in ``pure_pipeline``.
"""

import types

import pytest
import torch
import torch._dynamo
from torch._C._dynamo.eval_frame import _debug_get_cache_entry_list
from torch._dynamo.utils import counters

import nanoplm.pretraining.models.modern_bert.modeling as modeling
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.modeling import ModernBertEncoderLayer
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.pretraining.pure_pipeline import _count_layer_variants, _dynamo_recompile_limit

# 10 stored layers is the production loop-imp geometry and exceeds Dynamo's
# recompile_limit, so per-layer frame specialization would show up as an eager
# fallback rather than passing quietly.
_NUM_LAYERS = 10


def _cfg(**overrides):
    kwargs = dict(
        vocab_size=32,
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


def _loopie_model(**overrides):
    return PureProtModernBertMLM(
        _cfg(use_loopie=True, loopie_loop_count=2, **overrides)
    ).eval()


def _batch(cfg):
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    return input_ids, torch.ones_like(input_ids)


def test_compile_bounded_for_varying_loopie_loop_count(make_counting_backend):
    """Distinct per-micro-step R values must not each trigger a fresh trace."""
    torch.manual_seed(0)
    model = _loopie_model()
    input_ids, attention_mask = _batch(model.model_config)

    backend, n_compiles = make_counting_backend()
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend, dynamic=False)

    with torch.no_grad():
        compiled(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loop_count=torch.tensor(2, dtype=torch.int64),
        )
    warmup_compiles = n_compiles[0]
    assert warmup_compiles > 0, "expected at least one compile on first call"

    rs_after = (1, 3, 1, 3, 2, 4, 5, 6)
    for r in rs_after:
        with torch.no_grad():
            compiled(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loop_count=torch.tensor(r, dtype=torch.int64),
            )

    recompiles = n_compiles[0] - warmup_compiles
    assert recompiles == 0, (
        f"Expected 0 recompiles across {len(rs_after)} R values after warmup; got "
        f"{recompiles}. Either the _forward_repeats disable is bypassed, or R is "
        f"reaching a compiled segment as a Python int and installing a value guard."
    )


def test_no_per_layer_cache_thrash_at_production_depth(make_counting_backend):
    """The per-layer disable boundaries must not exhaust the Dynamo cache.

    Structurally identical layers share a cache entry, so the graph count tracks
    the number of distinct layer shapes, not the layer count. If it tracked the
    layer count instead, Dynamo would hit recompile_limit, fall back to eager,
    and make the zero-recompile assertion above pass vacuously.
    """
    torch.manual_seed(0)
    assert _NUM_LAYERS > torch._dynamo.config.recompile_limit, (
        f"test precondition: {_NUM_LAYERS} layers must exceed recompile_limit "
        f"({torch._dynamo.config.recompile_limit}) to exercise the failure"
    )
    model = _loopie_model()
    input_ids, attention_mask = _batch(model.model_config)

    backend, n_compiles = make_counting_backend()
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend, dynamic=False)

    # torch 2.12 logs the limit hit instead of warning; this makes it raise.
    with torch._dynamo.config.patch(fail_on_recompile_limit_hit=True):
        for r in (2, 3, 4):
            with torch.no_grad():
                compiled(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    loop_count=torch.tensor(r, dtype=torch.int64),
                )

    assert n_compiles[0] <= 4 * _NUM_LAYERS, (
        f"Expected the graph count to track distinct layer shapes; got "
        f"{n_compiles[0]} for {_NUM_LAYERS} layers."
    )


def test_fixed_path_compiles_exactly_as_an_unlooped_model(make_counting_backend):
    """Without the kwarg, loopie must trace to the graph it traced before.

    ``loop_count is None`` is a constant test, so Dynamo folds the branch and
    never reaches ``_forward_repeats``. Two consequences are asserted: no graph
    break names that method, and the graph count equals an unlooped model's.
    The count is not 1 — this model breaks on data-dependent branching and on
    ``run_flush`` for reasons that predate looping — so the unlooped model is
    the reference rather than a hard-coded number.
    """
    torch.manual_seed(0)
    model = _loopie_model()
    input_ids, attention_mask = _batch(model.model_config)

    backend, n_compiles = make_counting_backend()
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(model, backend=backend, dynamic=False)
    with torch.no_grad():
        compiled(input_ids=input_ids, attention_mask=attention_mask)
    looped_compiles = n_compiles[0]

    breaks = [reason for reason in counters["graph_break"] if "_forward_repeats" in reason]
    assert not breaks, (
        "The fixed path traced the Dynamo-disabled driver, so `loop_count is None` "
        f"was not constant-folded:\n{breaks}"
    )

    with torch.no_grad():
        compiled(input_ids=input_ids, attention_mask=attention_mask)
    assert n_compiles[0] == looped_compiles, "fixed path recompiled on a repeat call"

    plain = PureProtModernBertMLM(_cfg()).eval()
    plain_backend, n_plain = make_counting_backend()
    torch._dynamo.reset()
    compiled_plain = torch.compile(plain, backend=plain_backend, dynamic=False)
    with torch.no_grad():
        compiled_plain(input_ids=input_ids, attention_mask=attention_mask)

    assert looped_compiles == n_plain[0], (
        f"Fixed-depth loopie compiled into {looped_compiles} graphs vs "
        f"{n_plain[0]} for the same model without looping; the new branch added "
        f"a graph break."
    )


def test_truncated_bptt_traces_like_full_backprop(make_counting_backend):
    assert torch.is_grad_enabled(), "the mid-forward no_grad toggle is a no-op without this"
    torch.manual_seed(0)
    full = _loopie_model()
    input_ids, attention_mask = _batch(full.model_config)

    full_backend, n_full = make_counting_backend()
    torch._dynamo.reset()
    compiled_full = torch.compile(full, backend=full_backend, dynamic=False)
    compiled_full(input_ids=input_ids, attention_mask=attention_mask)

    torch.manual_seed(0)
    truncated = _loopie_model(loopie_bptt=1)
    trunc_backend, n_trunc = make_counting_backend()
    torch._dynamo.reset()
    counters.clear()
    compiled_trunc = torch.compile(truncated, backend=trunc_backend, dynamic=False)
    compiled_trunc(input_ids=input_ids, attention_mask=attention_mask)
    after_warmup = n_trunc[0]
    compiled_trunc(input_ids=input_ids, attention_mask=attention_mask)

    breaks = [reason for reason in counters["graph_break"] if "_forward_repeats" in reason]
    assert not breaks, (
        f"The truncated-BPTT split reached the disabled driver:\n{breaks}"
    )
    assert n_trunc[0] == after_warmup, "loopie_bptt=1 recompiled on a repeat call"
    assert n_trunc[0] == n_full[0], (
        f"loopie_bptt=1 compiled into {n_trunc[0]} graphs vs {n_full[0]} for the "
        f"same model with full backprop; the bptt split multiplied traces."
    )


def test_one_layer_call_per_step_regardless_of_depth():
    """FSDP2 all-gathers on __call__, so every repeat must stay inside forward.

    Counting forward pre-hooks against ``_forward_once`` executions is exactly
    the invariant fully_shard keys on, and it needs no GPUs to check.
    """
    torch.manual_seed(0)
    model = _loopie_model()
    input_ids, attention_mask = _batch(model.model_config)

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

    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loop_count=torch.tensor(4, dtype=torch.int64),
        )

    assert calls == {i: 1 for i in range(_NUM_LAYERS)}, (
        f"Each stored layer must be entered once per step (one all-gather, one "
        f"reshard); got {calls}."
    )
    assert bodies == {i: 4 for i in range(_NUM_LAYERS)}, (
        f"Each stored layer must run its body loop_count times; got {bodies}."
    )


def _stub_flash_varlen(q, k, v, *, window_size, **kwargs):
    """Shape-preserving CPU stand-in for the flash varlen kernel.

    Reads ``window_size`` like the real kernel's int argument does, so full
    and sliding layers keep their distinct guards.
    """
    return v * (1.0 + 0 * window_size[0])


# Flat token count per micro-batch; the three max_seqlen buckets below stand in
# for the packed collator's power-of-two buckets, the shapes that coexist in
# one compile epoch (batch-size warmup stages are separated by a reset).
_PACKED_TOKENS = 128
_MAX_SEQLENS = (32, 64, 128)
_JITTER_RS = (1, 2, 3)


@pytest.mark.parametrize(
    "pattern, num_layers, variants",
    # "F": layer 0 (Identity attn_norm) + the rest. "FS": that, plus the
    # sliding layers, whose window_size is an extra guard.
    [("F", 3, 2), ("FS", 4, 3)],
)
@pytest.mark.parametrize("bptt", [None, 1])
def test_jitter_training_fits_the_computed_recompile_limit(
    monkeypatch, make_counting_backend, pattern, num_layers, variants, bptt
):
    """Jitter + truncated BPTT in training must not overflow _forward_once's cache.

    The no_grad applications are a separate grad-mode guard, so the shared
    per-layer frame needs V x S x G entries. With the limit from
    ``_dynamo_recompile_limit`` and ``fail_on_recompile_limit_hit`` on, an
    undersized limit raises instead of silently running that shape eager.
    """
    monkeypatch.setattr(modeling, "_flash_varlen_fn", _stub_flash_varlen)
    monkeypatch.setattr(modeling, "_HAS_FLASH_VARLEN", True)
    torch.manual_seed(0)
    model = PureProtModernBertMLM(
        _cfg(
            num_hidden_layers=num_layers,
            attn_layer_pattern=pattern,
            use_loopie=True,
            loopie_loop_count=2,
            loopie_bptt=bptt,
        )
    ).train()
    assert _count_layer_variants(model.model.layers) == variants

    grad_modes = 2 if bptt is not None and max(_JITTER_RS) > bptt else 1
    limit = _dynamo_recompile_limit(
        model.model_config,
        layers=model.model.layers,
        loopie_jitter=1,
        global_batch_warmup_steps=0,
        n_input_shapes=len(_MAX_SEQLENS),
    )

    backend, _ = make_counting_backend()
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend, dynamic=False)
    with torch._dynamo.config.patch(recompile_limit=limit, fail_on_recompile_limit_hit=True):
        for max_seqlen in _MAX_SEQLENS:
            for r in _JITTER_RS:
                input_ids = torch.randint(0, model.model_config.vocab_size, (_PACKED_TOKENS,))
                cu_seqlens = torch.arange(0, _PACKED_TOKENS + 1, max_seqlen, dtype=torch.int32)
                torch._dynamo.mark_dynamic(cu_seqlens, 0)
                out = compiled(
                    input_ids=input_ids,
                    labels=input_ids.clone(),
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    position_ids=torch.arange(_PACKED_TOKENS) % max_seqlen,
                    loop_count=torch.tensor(r, dtype=torch.int64),
                )
                out.loss.backward()

    entries = len(_debug_get_cache_entry_list(ModernBertEncoderLayer._forward_once.__code__))
    assert entries == variants * len(_MAX_SEQLENS) * grad_modes, (
        f"_forward_once has {entries} cache entries; expected V={variants} x "
        f"S={len(_MAX_SEQLENS)} x G={grad_modes}."
    )
    torch._dynamo.reset()


def test_recompile_limit_arithmetic():
    loopie = types.SimpleNamespace(use_loopie=True, loopie_loop_count=2, loopie_bptt=1)
    full_bptt = types.SimpleNamespace(use_loopie=True, loopie_loop_count=2, loopie_bptt=None)
    plain = types.SimpleNamespace(use_loopie=False)

    def limit(cfg, jitter=1, warmup=0, shapes=6, variants=3):
        return _dynamo_recompile_limit(
            cfg,
            layers=[torch.nn.Linear(k + 1, 1) for k in range(variants)],
            loopie_jitter=jitter,
            global_batch_warmup_steps=warmup,
            n_input_shapes=shapes,
        )

    assert _count_layer_variants([torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)]) == 1
    assert limit(plain) == 8
    assert limit(plain, warmup=10) == 16
    assert limit(loopie, jitter=0) == 8
    assert limit(loopie) == 3 * 6 * 2  # max R = 3 > bptt = 1
    assert limit(full_bptt) == 3 * 6
    assert limit(loopie, shapes=1, variants=2, warmup=10) == 16  # warmup floor wins
    bptt_covers_r = types.SimpleNamespace(use_loopie=True, loopie_loop_count=2, loopie_bptt=3)
    assert limit(bptt_covers_r) == 3 * 6  # every R <= bptt: no no_grad applications
