"""Compile-stability and FSDP2 invariants for Loopie stochastic depth.

Two constraints pull against each other here. The trip count must not thrash
the Dynamo cache: a tensor R stays out of every traced graph (the
``_forward_repeats`` driver), while a Python int R is allowed to specialize,
once per distinct value, because the training support is three values and the
unrolled graph is the one a fixed-depth model traces. And the repeats must stay
inside ``ModernBertEncoderLayer.forward``, or FSDP2 — which shards per encoder
layer — all-gathers once per repeat instead of once per layer. Both live in
this file because a fix for either can break the other.

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
from nanoplm.pretraining.pure_pipeline import (
    _count_layer_variants,
    _dynamo_recompile_limit,
    _maybe_raise_dynamo_recompile_limit,
)

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


def test_int_loop_count_specializes_once_per_value(make_counting_backend):
    """A Python int R compiles once per distinct value and never again.

    This is the loopie training path (``_loop_count_fwd_kwargs`` passes an
    int). Each of R=1,2,3 pays one trace on first sight; repeats of any of them
    must hit the cache, and none may reach the Dynamo-disabled driver.
    """
    torch.manual_seed(0)
    model = _loopie_model()
    input_ids, attention_mask = _batch(model.model_config)

    backend, n_compiles = make_counting_backend()
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(model, backend=backend, dynamic=False)

    seen = []
    for r in (2, 1, 3):
        before = n_compiles[0]
        with torch.no_grad():
            compiled(input_ids=input_ids, attention_mask=attention_mask, loop_count=r)
        seen.append(n_compiles[0] - before)
    assert all(n > 0 for n in seen), f"each new R must trace at least once; got {seen}"
    after_warmup = n_compiles[0]

    for r in (1, 2, 3, 3, 1, 2):
        with torch.no_grad():
            compiled(input_ids=input_ids, attention_mask=attention_mask, loop_count=r)
    assert n_compiles[0] == after_warmup, (
        f"int R recompiled on values already seen: {n_compiles[0] - after_warmup} "
        "extra graphs after warmup"
    )

    breaks = [reason for reason in counters["graph_break"] if "_forward_repeats" in reason]
    assert not breaks, f"int R reached the eager driver:\n{breaks}"


def test_int_loop_count_traces_the_fixed_depth_graph(make_counting_backend):
    """``loop_count=2`` as an int must compile into the same graphs as fixed R=2.

    That equality is what makes a stochastic-depth 20xR arm's R=2 steps run
    the same kernels as a fixed 20x2 arm.
    """
    torch.manual_seed(0)
    model = _loopie_model()
    input_ids, attention_mask = _batch(model.model_config)

    fixed_backend, n_fixed = make_counting_backend()
    torch._dynamo.reset()
    compiled_fixed = torch.compile(model, backend=fixed_backend, dynamic=False)
    with torch.no_grad():
        compiled_fixed(input_ids=input_ids, attention_mask=attention_mask)

    int_backend, n_int = make_counting_backend()
    torch._dynamo.reset()
    compiled_int = torch.compile(model, backend=int_backend, dynamic=False)
    with torch.no_grad():
        compiled_int(input_ids=input_ids, attention_mask=attention_mask, loop_count=2)

    assert n_int[0] == n_fixed[0], (
        f"int R=2 compiled into {n_int[0]} graphs vs {n_fixed[0]} for the fixed-depth "
        "path; the int branch added a graph break or a split trace."
    )


def test_int_and_tensor_loop_count_agree_numerically():
    torch.manual_seed(0)
    model = _loopie_model()
    input_ids, attention_mask = _batch(model.model_config)
    with torch.no_grad():
        for r in (1, 3):
            via_int = model(input_ids=input_ids, attention_mask=attention_mask, loop_count=r)
            via_tensor = model(
                input_ids=input_ids, attention_mask=attention_mask,
                loop_count=torch.tensor(r, dtype=torch.int64),
            )
            torch.testing.assert_close(via_int.logits, via_tensor.logits)


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


@pytest.mark.parametrize("as_tensor", [False, True], ids=["int", "tensor"])
def test_one_layer_call_per_step_regardless_of_depth(as_tensor):
    """FSDP2 all-gathers on __call__, so every repeat must stay inside forward.

    Counting forward pre-hooks against ``_forward_once`` executions is exactly
    the invariant fully_shard keys on, and it needs no GPUs to check. Both the
    unrolled int path and the eager tensor driver must hold it.
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
            loop_count=torch.tensor(4, dtype=torch.int64) if as_tensor else 4,
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


# Flat token count per micro-batch; the max_seqlen buckets below stand in for
# the packed collator's power-of-two buckets, the shapes that coexist in one
# compile epoch (batch-size warmup stages are separated by a reset).
_PACKED_TOKENS = 64
_MAX_SEQLENS = (32, 64)
_JITTER_RS = (1, 2, 3)  # R0 = 2, jitter 1


@pytest.fixture
def single_rank_fsdp():
    """A one-rank gloo group, so ``fully_shard`` runs on CPU."""
    import socket

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    owns_group = not dist.is_initialized()
    if owns_group:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
        )
    mesh = init_device_mesh("cpu", (1,))

    def _shard(model, sublayer=False):
        # Per-layer units (attn and mlp first with sublayer granularity), then
        # the root, as pure_pipeline shards them.
        for layer in model.model.layers:
            if sublayer:
                fully_shard(layer.attn, mesh=mesh, reshard_after_forward=False)
                if layer.mlp is not None:
                    fully_shard(layer.mlp, mesh=mesh, reshard_after_forward=False)
            fully_shard(layer, mesh=mesh, reshard_after_forward=False)
        fully_shard(model, mesh=mesh, reshard_after_forward=False)
        return model

    yield _shard
    if owns_group:
        dist.destroy_process_group()


_MOE = dict(
    use_moe=True, moe_num_experts=4, moe_top_k=2, moe_leading_dense_layers=2,
    mlp_activation="swiglu",
)


_HUGINN = dict(
    use_huginn_looping=True, recycling_loop_count=3,
    recycling_num_prelude_layers=1, recycling_num_coda_layers=1,
)
_HUGINN_RS = (1, 2, 3, 4)


@pytest.mark.parametrize(
    "looping, pattern, num_layers, variants, bptt, jitter, moe, sublayer, seqlens, frame, expected",
    # "F": layer 0 (Identity attn_norm) + the rest. "FS": that, plus the
    # sliding layers, whose window_size is an extra guard. MoE with two
    # leading dense layers: layer 0, dense, MoE. Per-layer units: the layer
    # frame needs V x S x N, and bptt=1 adds nothing on the unrolled path.
    # Sublayer units: the frame is _forward_once, which splits on grad mode
    # under bptt (x 2) but not on R, and no longer sees the attention type or
    # MLP kind (those moved into their own frames), so only layer 0 differs:
    # 2 x S x 2, with a third max_seqlen to clear the default limit. The MoE
    # mlp frame stays at <= 4 entries (None/tensor last_aux_loss x grad mode).
    # The tensor-R fallback (_forward_repeats) is off the training path and
    # not sized by the limit. MoE without looping: the dispatch workspace
    # reuse graph-breaks at the resize_ in MoELayer._ensure_dispatch_buffers
    # (on by default without looping, CUDA cutlass path too), so forward's MoE
    # entry stops at the _forward_once call with no layer or max_seqlen guards
    # and, being newest, takes every layer from the second shape on: today
    # _forward_once holds 1 + V x (S - 1) = 13 of the 15 the limit allows (old
    # limit 8), and forward keeps the first shape's V. That count hangs on the
    # break, so the row (None) is bounded, not exact: 8 < entries <= limit. Huginn with recycling_bptt K = 1 (the
    # bptt column) runs R - 1 core passes under no_grad, so the layer frame
    # sees both grad modes, except that the prelude layer never runs under
    # no_grad: S x (V + 1) = 9 entries for R in 1..4, against 8 before and
    # V x S x 2 = 12 now. The limit counts x 2 for every Huginn run, since
    # recycling_bptt lives in the pretrain config, not the model config.
    [
        ("loopie", "F", 3, 2, None, 1, False, False, _MAX_SEQLENS, "forward", 2 * 2 * 3),
        ("loopie", "FS", 4, 3, 1, 1, False, False, _MAX_SEQLENS, "forward", 3 * 2 * 3),
        ("loopie", "F", 4, 3, None, 1, True, False, _MAX_SEQLENS, "forward", 3 * 2 * 3),
        ("loopie", "FS", 4, 3, 1, 1, False, True, (16, 32, 64), "_forward_once", 2 * 3 * 2),
        ("loopie", "FS", 4, 3, 1, 0, False, True, (16, 32, 64), "_forward_once", 2 * 3 * 2),
        ("loopie", "F", 4, 3, 1, 1, True, True, (16, 32, 64), "_forward_once", 2 * 3 * 2),
        ("none", "F", 4, 3, None, 0, True, False, (4, 8, 16, 32, 64), "_forward_once", None),
        ("huginn", "F", 4, 2, 1, 0, False, False, (16, 32, 64), "forward", 3 * (2 + 1)),
    ],
    ids=["F", "FS-bptt", "moe-jitter", "FS-sublayer-bptt", "FS-sublayer-bptt-j0",
         "moe-sublayer-bptt", "moe-no-loopie", "huginn-bptt"],
)
def test_jitter_training_fits_the_computed_recompile_limit(
    monkeypatch, make_counting_backend, single_rank_fsdp,
    looping, pattern, num_layers, variants, bptt, jitter, moe, sublayer, seqlens, frame, expected,
):
    """Loopie training under FSDP2 must fit the limit the pipeline sets.

    FSDP2's dynamo-disabled hooks break the graph at every FSDP unit, so
    ``ModernBertEncoderLayer.forward`` (or ``_forward_once`` with sublayer
    units) is one frame shared by all layers, and the int R and the input
    shapes specialize it. With the limit from ``_dynamo_recompile_limit`` and
    ``fail_on_recompile_limit_hit`` on, an undersized limit raises instead of
    silently running that shape eager.
    """
    monkeypatch.setattr(modeling, "_flash_varlen_fn", _stub_flash_varlen)
    monkeypatch.setattr(modeling, "_HAS_FLASH_VARLEN", True)
    torch.manual_seed(0)
    looping_cfg = {
        "loopie": dict(use_loopie=True, loopie_loop_count=2, loopie_bptt=bptt),
        "none": {},
        "huginn": _HUGINN,
    }[looping]
    model = PureProtModernBertMLM(
        _cfg(
            num_hidden_layers=num_layers,
            attn_layer_pattern=pattern,
            **looping_cfg,
            **(_MOE if moe else {}),
        )
    ).train()
    assert _count_layer_variants(model.model.layers) == variants
    limit = _dynamo_recompile_limit(
        model.model_config,
        layers=model.model.layers,
        loopie_jitter=jitter,
        global_batch_warmup_steps=0,
        n_input_shapes=len(seqlens),
    )
    if expected is not None:
        assert expected > 8, "test precondition: the default limit must be too small"
    single_rank_fsdp(model, sublayer=sublayer)

    backend, _ = make_counting_backend()
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(model, backend=backend, dynamic=False)
    if looping == "loopie":
        loop_kwargs = [dict(loop_count=r) for r in range(2 - jitter, 2 + jitter + 1)]
    elif looping == "huginn":
        loop_kwargs = [
            dict(loop_count=torch.tensor(r), recycling_bptt=torch.tensor(bptt))
            for r in _HUGINN_RS
        ]
    else:
        loop_kwargs = [{}]
    with torch._dynamo.config.patch(recompile_limit=limit, fail_on_recompile_limit_hit=True):
        for max_seqlen in seqlens:
            for extra in loop_kwargs:
                input_ids = torch.randint(0, model.model_config.vocab_size, (_PACKED_TOKENS,))
                cu_seqlens = torch.arange(0, _PACKED_TOKENS + 1, max_seqlen, dtype=torch.int32)
                torch._dynamo.mark_dynamic(cu_seqlens, 0)
                out = compiled(
                    input_ids=input_ids,
                    labels=input_ids.clone(),
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    position_ids=torch.arange(_PACKED_TOKENS) % max_seqlen,
                    **extra,
                )
                out.loss.backward()

    breaks = [reason for reason in counters["graph_break"] if "_forward_repeats" in reason]
    assert not breaks, f"int R reached the eager driver:\n{breaks}"
    code = getattr(ModernBertEncoderLayer, frame).__code__
    entries = len(_debug_get_cache_entry_list(code))
    if expected is None:
        assert 8 < entries <= limit, (
            f"ModernBertEncoderLayer.{frame} has {entries} cache entries; "
            f"expected 8 < n <= {limit}."
        )
    else:
        assert entries == expected, (
            f"ModernBertEncoderLayer.{frame} has {entries} cache entries; expected {expected}."
        )
    if looping == "none":
        assert len(_debug_get_cache_entry_list(ModernBertEncoderLayer.forward.__code__)) == variants
    if moe:
        from nanoplm.pretraining.models.modern_bert.moe import MoELayer

        assert len(_debug_get_cache_entry_list(MoELayer.forward.__code__)) <= 4
    torch._dynamo.reset()


def test_recompile_limit_arithmetic():
    truncated = types.SimpleNamespace(use_loopie=True, loopie_loop_count=2, loopie_bptt=1)
    full_bptt = types.SimpleNamespace(use_loopie=True, loopie_loop_count=2, loopie_bptt=None)
    plain = types.SimpleNamespace(use_loopie=False)
    huginn = types.SimpleNamespace(use_loopie=False, use_huginn_looping=True)

    def limit(cfg, jitter=1, warmup=0, shapes=6, variants=3):
        return _dynamo_recompile_limit(
            cfg,
            layers=[torch.nn.Linear(k + 1, 1) for k in range(variants)],
            loopie_jitter=jitter,
            global_batch_warmup_steps=warmup,
            n_input_shapes=shapes,
        )

    assert _count_layer_variants([torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)]) == 1
    assert limit(plain) == 3 * 6  # V x S
    assert limit(plain, shapes=2, variants=2) == 8  # floor wins
    assert limit(plain, shapes=2, variants=2, warmup=10) == 16
    assert limit(full_bptt, jitter=0) == 3 * 6
    assert limit(truncated, jitter=0) == 3 * 6 * 2  # sublayer _forward_once splits on grad mode
    assert limit(full_bptt) == 3 * 6 * 3  # V x S x (2j + 1)
    assert limit(full_bptt, jitter=2) == 3 * 6 * 5
    assert limit(truncated) == limit(full_bptt)  # grad modes do not multiply R
    assert limit(full_bptt, shapes=1, variants=1, warmup=10) == 16  # warmup floor wins
    assert limit(huginn) == 3 * 6 * 2  # no_grad recurrence splits on grad mode


def test_maybe_raise_recompile_limit_lifts_the_accumulated_cap():
    cfg = torch._dynamo.config
    with cfg.patch(recompile_limit=8, accumulated_recompile_limit=256):
        _maybe_raise_dynamo_recompile_limit(300)
        assert (cfg.recompile_limit, cfg.accumulated_recompile_limit) == (300, 300)
        with cfg.patch(recompile_limit=8, accumulated_recompile_limit=256):
            _maybe_raise_dynamo_recompile_limit(10)
            assert (cfg.recompile_limit, cfg.accumulated_recompile_limit) == (10, 256)
        with cfg.patch(recompile_limit=50):
            _maybe_raise_dynamo_recompile_limit(4)  # never lowers
            assert cfg.recompile_limit == 50
