"""The debug stats must be free of graph breaks and match the old hook math.

``LayerwiseDebugTracker`` used to install ``@torch.compiler.disable``d forward
hooks on every block and on the lm_head. Dynamo then skipped the whole compiled
frame, so ``debug_layerwise_metrics: true`` ran the model largely eager. The
stats now live in the model itself (``modern_bert.taps.RepeatTap`` /
``StatsTap``) and mutate tracker-owned tensors in place inside the graph. Two
things have to hold for that to be worth anything:

* the compiled graph count is identical with the tracker on and off, and no
  graph break mentions the taps or the tracker;
* the numbers are the ones the old hooks produced — the reference here
  re-registers those exact hook formulas on a deepcopy of the same model.

``dynamic=False`` matches production (``use_packing`` + ``use_static_inp_size``
resolve ``compile_dynamic`` to False in ``pure_pipeline``).
"""

import copy

import pytest
import torch
import torch._dynamo
import torch.nn.functional as F
from torch._dynamo.utils import counters

from nanoplm.pretraining.debug_metrics import _BLOCK_CLASS_NAMES, LayerwiseDebugTracker
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM

_NUM_LAYERS = 2
_VOCAB = 32
_MAX_REP = 2

_STAT_PREFIXES = ("debug/residual_rms/", "debug/final_norm/", "debug/logits/")

# Canon layers are in `debug_metrics._BLOCK_CLASS_NAMES`, so they are enumerated
# as blocks too (4 per layer under canon "abcd", one sample each in slot 0).
# Production enables them, so the residual key inventory is checked both ways.
_CANON = {
    False: dict(use_canon_layers=False),
    True: dict(use_canon_layers=True, canon_layers_mode="abcd"),
}
# Residual keys: looped layers give max_rep each, canon layers one each.
_N_RES_KEYS = {
    False: _NUM_LAYERS * _MAX_REP,
    True: _NUM_LAYERS * _MAX_REP + _NUM_LAYERS * 4,
}
# debug/logits/{norm,abs_max,entropy} + debug/final_norm/{rms,mean,abs_max}
_N_SCALAR_KEYS = 6


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


def _loopie_model(**overrides):
    """A small loopie model, identical every call (seeded), with wo != 0."""
    torch.manual_seed(0)
    model = PureProtModernBertMLM(
        _cfg(use_loopie=True, loopie_loop_count=_MAX_REP, **overrides)
    )
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model


def _tracker(model, tmp_path, max_rep=_MAX_REP):
    """Constructed BEFORE torch.compile: the accumulators must be the tensors
    the first trace lifts as graph inputs."""
    return LayerwiseDebugTracker(
        model=model,
        log_every=1,
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        is_main=True,
        distributed=False,
        max_rep=max_rep,
    )


def _batches(n_steps, batch=2, seq=8):
    out = []
    for _ in range(n_steps):
        input_ids = torch.randint(0, _VOCAB, (batch, seq))
        out.append((input_ids, torch.ones_like(input_ids), input_ids.clone()))
    return out


class _OldHookStats:
    """The pre-A1 forward hooks, formula for formula, on an eager model."""

    def __init__(self, model: torch.nn.Module, max_rep: int) -> None:
        self.max_rep = max_rep
        blocks = [
            mod
            for _, mod in model.named_modules()
            if any(c.__name__ in _BLOCK_CLASS_NAMES for c in type(mod).__mro__)
        ]
        self.res_sq = torch.zeros(len(blocks), max_rep)
        self.res_cnt = torch.zeros(len(blocks), max_rep)
        self.logit = torch.zeros(3)  # sq_sum, abs_max, entropy_sum
        self.logit_elem = 0.0
        self.logit_tok = 0.0
        self.final = torch.zeros(3)  # sq_sum, sum, abs_max
        self.final_elem = 0.0
        self._handles = [
            block.repeat_tap.register_forward_hook(self._res_hook(idx))
            for idx, block in enumerate(blocks)
        ]
        self._handles.append(
            model.model.final_norm.register_forward_hook(self._final_hook)
        )
        self._handles.append(model.decoder.register_forward_hook(self._logit_hook))

    def _res_hook(self, idx: int):
        def hook(module, inputs, output):
            if not module.training:
                return
            rep = int(inputs[1])
            if rep >= self.max_rep:
                return
            with torch.no_grad():
                xd = output.detach()
                sq = (xd * xd).mean()
                self.res_sq[idx, rep].add_(sq.float())
                self.res_cnt[idx, rep].add_(1.0)

        return hook

    def _final_hook(self, module, inputs, output):
        if not module.training:
            return
        with torch.no_grad():
            xd = output.detach()
            self.final[0].add_((xd * xd).sum().float())
            self.final[1].add_(xd.sum().float())
            self.final[2].copy_(torch.maximum(self.final[2], xd.abs().amax().float()))
            self.final_elem += float(xd.numel())

    def _logit_hook(self, module, inputs, output):
        if not module.training:
            return
        with torch.no_grad():
            logits = output.detach()
            lse = torch.logsumexp(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            entropy_per_token = lse - (probs * logits).sum(dim=-1)
            self.logit[0].add_((logits * logits).sum().float())
            self.logit[1].copy_(
                torch.maximum(self.logit[1], logits.abs().amax().float())
            )
            self.logit[2].add_(entropy_per_token.sum().float())
            n_elem = logits.numel()
            self.logit_elem += float(n_elem)
            self.logit_tok += float(n_elem // logits.shape[-1])

    def payload(self) -> dict:
        """The same fan-out `LayerwiseDebugTracker.flush` does."""
        out: dict = {}
        for i in range(self.res_cnt.shape[0]):
            for r in range(self.max_rep):
                cnt = float(self.res_cnt[i, r])
                if cnt > 0:
                    mean_sq = float(self.res_sq[i, r]) / cnt
                    out[f"debug/residual_rms/block_{i:03d}_rep{r}"] = mean_sq**0.5
        if self.logit_elem > 0:
            out["debug/logits/norm"] = float(self.logit[0]) ** 0.5
            out["debug/logits/abs_max"] = float(self.logit[1])
        if self.logit_tok > 0:
            out["debug/logits/entropy"] = float(self.logit[2]) / self.logit_tok
        if self.final_elem > 0:
            out["debug/final_norm/rms"] = (float(self.final[0]) / self.final_elem) ** 0.5
            out["debug/final_norm/mean"] = float(self.final[1]) / self.final_elem
            out["debug/final_norm/abs_max"] = float(self.final[2])
        return out

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


@pytest.mark.parametrize("canon", [False, True], ids=["no_canon", "canon"])
def test_tracker_adds_no_graphs_and_no_graph_breaks(
    make_counting_backend, tmp_path, canon
):
    """The whole point of A1: enabling the tracker must not split a graph.

    A full micro-step (forward, loss, backward) each way: the logit tap only
    fires on a scoring path, and the input-mutation epilogue AOTAutograd adds
    for the accumulators shows up in the backward graph count.
    """
    input_ids, attention_mask, labels = _batches(1)[0]

    off = _loopie_model(**_CANON[canon])
    off.train()
    backend_off, n_off = make_counting_backend()
    torch._dynamo.reset()
    counters.clear()
    compiled_off = torch.compile(off, backend=backend_off, dynamic=False)
    compiled_off(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loop_count=_MAX_REP,
    ).loss.backward()
    assert n_off[0] > 0, "expected at least one compile without the tracker"
    breaks_off = dict(counters["graph_break"])

    on = _loopie_model(**_CANON[canon])
    on.train()
    tracker = _tracker(on, tmp_path)
    backend_on, n_on = make_counting_backend()
    torch._dynamo.reset()
    counters.clear()
    compiled_on = torch.compile(on, backend=backend_on, dynamic=False)
    try:
        compiled_on(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            loop_count=_MAX_REP,
        ).loss.backward()
        assert float(tracker.residual_count.sum()) > 0.0, (
            "the taps did not accumulate, so the comparisons below would pass "
            "vacuously"
        )
        assert float(tracker._logit_stats[3]) > 0.0, "the logit tap never fired"
    finally:
        tracker.close()

    # Reason strings, with counts: a new break anywhere (or one more of an
    # existing kind) fails, which is what a tap-induced break would look like.
    assert dict(counters["graph_break"]) == breaks_off, (
        "the taps changed the graph breaks:\n"
        f"off: {sorted(breaks_off.items())}\non: "
        f"{sorted(dict(counters['graph_break']).items())}"
    )
    assert n_on[0] == n_off[0], (
        f"tracker on compiled into {n_on[0]} graphs vs {n_off[0]} with it off; "
        "the in-graph stats added a graph break."
    )


@pytest.mark.parametrize("canon", [False, True], ids=["no_canon", "canon"])
def test_compiled_stats_match_the_old_hook_formulas(tmp_path, canon):
    """Two micro-steps at R=2, compiled, against the old eager hook math."""
    model = _loopie_model(**_CANON[canon])
    reference = copy.deepcopy(model)
    model.train()
    reference.train()
    batches = _batches(2)

    tracker = _tracker(model, tmp_path)
    torch._dynamo.reset()
    compiled = torch.compile(model, backend="aot_eager", dynamic=False)
    ref_hooks = _OldHookStats(reference, max_rep=_MAX_REP)
    try:
        for input_ids, attention_mask, labels in batches:
            out = compiled(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                loop_count=_MAX_REP,
            )
            out.loss.backward()
            ref_out = reference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                loop_count=_MAX_REP,
            )
            ref_out.loss.backward()
        payload = tracker.flush(global_step=1)
    finally:
        ref_hooks.close()
        tracker.close()

    expected = ref_hooks.payload()
    assert len(expected) == _N_RES_KEYS[canon] + _N_SCALAR_KEYS, (
        f"the eager reference itself is incomplete: {sorted(expected)}"
    )
    got = {k: v for k, v in payload.items() if k.startswith(_STAT_PREFIXES)}
    assert set(got) == set(expected)
    for key, want in expected.items():
        torch.testing.assert_close(
            got[key], want, rtol=1e-4, atol=1e-6, msg=lambda m, _k=key: f"{_k}: {m}"
        )


def test_eval_mode_accumulates_nothing_under_compile(tmp_path):
    """The `self.training` guard has to exclude the in-loop evaluation."""
    model = _loopie_model()
    model.eval()
    tracker = _tracker(model, tmp_path)
    torch._dynamo.reset()
    compiled = torch.compile(model, backend="aot_eager", dynamic=False)
    try:
        with torch.no_grad():
            for input_ids, attention_mask, _ in _batches(2):
                compiled(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    loop_count=_MAX_REP,
                )
        assert float(tracker.residual_sq_sum.abs().sum()) == 0.0
        assert float(tracker.residual_count.abs().sum()) == 0.0
        assert float(tracker._logit_stats.abs().sum()) == 0.0
        assert float(tracker._final_stats.abs().sum()) == 0.0
        payload = tracker.flush(global_step=1)
    finally:
        tracker.close()

    assert not [k for k in payload if k.startswith(_STAT_PREFIXES)], (
        "eval-mode forwards leaked into the debug payload"
    )


def test_layer_checkpointing_counts_each_application_once(tmp_path):
    """Recompute must not double-count: one sample per (block, application).

    Eager `torch.utils.checkpoint` runs the layer body twice and would inflate
    both the sum and the count (means survive, counts do not). Compiled, the
    checkpoint HOP keeps the mutation once.
    """
    model = _loopie_model(
        activation_checkpointing=True, activation_checkpointing_mode="layer"
    )
    model.train()
    tracker = _tracker(model, tmp_path)
    torch._dynamo.reset()
    compiled = torch.compile(model, backend="aot_eager", dynamic=False)
    try:
        input_ids, attention_mask, labels = _batches(1)[0]
        out = compiled(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            loop_count=_MAX_REP,
        )
        out.loss.backward()
        counts = tracker.residual_count
        assert torch.equal(counts, torch.ones_like(counts)), (
            f"expected exactly one sample per (block, application) under layer "
            f"activation checkpointing; got\n{counts}"
        )
    finally:
        tracker.close()
