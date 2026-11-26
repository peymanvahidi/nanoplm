"""The debug activation stats cover real tokens only, never padding.

The packed collator fills every micro-batch to its token budget with a trailing
dummy pad sequence, and a padded batch carries pad positions. Neither is
trained, and in a real run the pad segment's residual grew hundreds of times
larger than the real tokens', so ``debug/residual_rms`` reported the pad
segment for most blocks. These tests pin the fix:

* padded path: the stats equal the same formulas over ``attention_mask == 1``,
  with and without Canon and mHC (their taps see other shapes);
* packed path: ``num_valid_tokens`` reaches every tap, and on GPU the
  stats of a batch with a large pad segment equal the real-prefix formulas,
  eager and compiled.
"""

import copy

import pytest
import torch
import torch._dynamo
import torch.nn.functional as F

from nanoplm.pretraining.debug_metrics import _BLOCK_CLASS_NAMES, LayerwiseDebugTracker
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.pretraining.models.modern_bert.taps import RepeatTap, StatsTap

_VOCAB = 32
_STAT_PREFIXES = ("debug/residual_rms/", "debug/final_norm/", "debug/logits/")


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


def _model(**overrides):
    torch.manual_seed(0)
    model = PureProtModernBertMLM(_cfg(**overrides))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    # Make the pad rows stand out, as they do in a trained model, so an
    # unmasked mean would visibly disagree with the real-token one. (Scaling
    # the pad embedding row would not do it: the embedding norm undoes that.)
    pad = model.config.pad_token_id

    def _inflate_pad(module, inputs, out):
        is_pad = (inputs[0] == pad).unsqueeze(-1)
        return torch.where(is_pad, out * 50.0, out)

    model.model.embeddings.register_forward_hook(_inflate_pad)
    return model


def _tracker(model, tmp_path, device="cpu"):
    return LayerwiseDebugTracker(
        model=model,
        log_every=1,
        device=torch.device(device),
        output_dir=str(tmp_path),
        is_main=True,
        distributed=False,
        max_rep=1,
    )


class _MaskedReference:
    """The tap formulas with forward hooks, over the rows ``row_mask`` keeps.

    Set ``row_mask`` (flat bool over the token rows: ``B*S`` padded, ``T``
    packed) before each forward. ``row_mask=None`` gives the old unmasked
    numbers, which the tests use to show the mask matters.
    """

    def __init__(self, model):
        blocks = [
            mod
            for _, mod in model.named_modules()
            if any(c.__name__ in _BLOCK_CLASS_NAMES for c in type(mod).__mro__)
        ]
        self.row_mask = None
        self.res_sq = torch.zeros(len(blocks), dtype=torch.float64)
        self.res_cnt = torch.zeros(len(blocks), dtype=torch.float64)
        self.final = [0.0, 0.0, 0.0, 0.0]  # sq_sum, sum, abs_max, numel
        self.logit = [0.0, 0.0, 0.0, 0.0]  # sq_sum, abs_max, entropy_sum, tokens
        self._handles = [
            b.repeat_tap.register_forward_hook(self._res_hook(i))
            for i, b in enumerate(blocks)
        ]
        self._handles.append(model.model.final_norm.register_forward_hook(self._final_hook))
        self._handles.append(model.decoder.register_forward_hook(self._logit_hook))

    def _rows(self, x):
        xf = x.detach().double()
        if self.row_mask is None:
            return xf.reshape(-1, x.shape[-1])
        return xf.reshape(self.row_mask.numel(), -1)[self.row_mask.reshape(-1).cpu()]

    def _res_hook(self, idx):
        def hook(module, inputs, output):
            if module.training:
                r = self._rows(output.cpu())
                self.res_sq[idx] += (r * r).mean()
                self.res_cnt[idx] += 1

        return hook

    def _final_hook(self, module, inputs, output):
        if module.training:
            r = self._rows(output.cpu())
            self.final[0] += float((r * r).sum())
            self.final[1] += float(r.sum())
            self.final[2] = max(self.final[2], float(r.abs().max()))
            self.final[3] += r.numel()

    def _logit_hook(self, module, inputs, output):
        if module.training:
            r = self._rows(output.cpu())
            ent = torch.logsumexp(r, -1) - (F.softmax(r, -1) * r).sum(-1)
            self.logit[0] += float((r * r).sum())
            self.logit[1] = max(self.logit[1], float(r.abs().max()))
            self.logit[2] += float(ent.sum())
            self.logit[3] += r.shape[0]

    def payload(self):
        out = {}
        for i in range(self.res_cnt.numel()):
            if self.res_cnt[i] > 0:
                out[f"debug/residual_rms/block_{i:03d}_rep0"] = float(
                    (self.res_sq[i] / self.res_cnt[i]).sqrt()
                )
        out["debug/logits/norm"] = self.logit[0] ** 0.5
        out["debug/logits/abs_max"] = self.logit[1]
        out["debug/logits/entropy"] = self.logit[2] / self.logit[3]
        out["debug/final_norm/rms"] = (self.final[0] / self.final[3]) ** 0.5
        out["debug/final_norm/mean"] = self.final[1] / self.final[3]
        out["debug/final_norm/abs_max"] = self.final[2]
        return out

    def close(self):
        for h in self._handles:
            h.remove()


def _assert_payload_matches(payload, expected):
    got = {k: v for k, v in payload.items() if k.startswith(_STAT_PREFIXES)}
    assert set(got) == set(expected)
    for key, want in expected.items():
        torch.testing.assert_close(
            got[key], want, rtol=2e-4, atol=1e-5, msg=lambda m, _k=key: f"{_k}: {m}"
        )


def _assert_mask_matters(masked, unmasked):
    """At least the residual of some block must move by >5%, or the test
    would pass with no mask at all."""
    gaps = [
        abs(masked[k] / unmasked[k] - 1)
        for k in masked
        if k.startswith("debug/residual_rms/")
    ]
    assert max(gaps) > 0.05, f"pad rows barely move the stats: {gaps}"


_PADDED_VARIANTS = {
    "plain": {},
    "canon": dict(use_canon_layers=True, canon_layers_mode="abcd"),
    "mhc": dict(use_mhc_lite=True, mhc_n_streams=4),
}


@pytest.mark.parametrize("variant", list(_PADDED_VARIANTS))
def test_padded_batch_stats_cover_real_tokens_only(tmp_path, variant):
    model = _model(**_PADDED_VARIANTS[variant])
    reference = copy.deepcopy(model)
    unmasked = copy.deepcopy(model)
    for m in (model, reference, unmasked):
        m.train()

    torch.manual_seed(1)
    input_ids = torch.randint(5, _VOCAB, (2, 12))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 7:] = 0  # second row: 5 pad positions
    input_ids[attention_mask == 0] = model.config.pad_token_id

    tracker = _tracker(model, tmp_path)
    ref = _MaskedReference(reference)
    ref.row_mask = attention_mask.bool()
    ref_all = _MaskedReference(unmasked)
    try:
        for m in (model, reference, unmasked):
            m(input_ids=input_ids, attention_mask=attention_mask)
        payload = tracker.flush(global_step=1)
    finally:
        tracker.close()
        ref.close()
        ref_all.close()

    expected = ref.payload()
    _assert_payload_matches(payload, expected)
    _assert_mask_matters(expected, ref_all.payload())


def test_forward_hands_num_valid_tokens_to_every_tap_in_training():
    model = _model(use_canon_layers=True, canon_layers_mode="abcd")
    taps = [m for m in model.modules() if isinstance(m, (RepeatTap, StatsTap))]
    assert len(taps) > 3
    input_ids = torch.randint(5, _VOCAB, (2, 8))
    n = torch.tensor(13)

    model.eval()
    model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), num_valid_tokens=n)
    assert all(t.valid_token_count is None for t in taps), "eval must not touch the taps"

    model.train()
    model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), num_valid_tokens=n)
    assert all(t.valid_token_count is n for t in taps)

    model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    assert all(t.valid_token_count is None for t in taps), "a later batch must reset it"


@pytest.mark.parametrize("shape", [(10, 8), (10, 4, 8)], ids=["packed", "packed_streams"])
def test_taps_drop_rows_past_valid_token_count(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape)
    x[7:] *= 1000.0  # the pad tail
    n = 7

    rep = RepeatTap().train()
    rep.enable_stats(1, torch.device("cpu"))
    rep.valid_token_count = torch.tensor(n)
    assert rep(x, 0) is x
    torch.testing.assert_close(rep.stats[0, 0], x[:n].pow(2).mean())
    assert float(rep.stats[1, 0]) == 1.0

    st = StatsTap(with_entropy=True).train()
    st.enable_stats(torch.device("cpu"))
    st.valid_token_count = torch.tensor(n)
    st(x)
    real = x[:n].reshape(n, -1)
    ent = torch.logsumexp(real, -1) - (F.softmax(real, -1) * real).sum(-1)
    want = torch.stack([
        real.pow(2).sum(), real.sum(), real.abs().max(),
        torch.tensor(float(real.numel())), ent.sum(), torch.tensor(float(n)),
    ])
    torch.testing.assert_close(st.stats, want)

    # A batch with no real token adds nothing and is not counted.
    rep.stats.zero_()
    rep.valid_token_count = torch.tensor(0)
    rep(x, 0)
    assert float(rep.stats[0, 0]) == 0.0 and float(rep.stats[1, 0]) == 0.0


def test_token_mask_must_match_the_token_dims():
    tap = RepeatTap().train()
    tap.enable_stats(1, torch.device("cpu"))
    with pytest.raises(ValueError, match="does not match"):
        tap(torch.randn(2, 6, 8), 0, torch.ones(12))


@pytest.mark.gpu
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_packed_batch_drops_the_trailing_pad_segment(tmp_path, compiled):
    """The training layout: proteins packed into one flat row, a trailing
    dummy pad sequence filling the budget, num_valid_tokens = the real prefix."""
    if not torch.cuda.is_available():
        pytest.skip("the packed path needs CUDA")
    from nanoplm.pretraining.models.modern_bert import modeling

    if not modeling._HAS_FLASH_VARLEN:
        pytest.skip("the packed path needs flash-attention varlen")

    model = _model().cuda()
    reference = copy.deepcopy(model)
    unmasked = copy.deepcopy(model)
    for m in (model, reference, unmasked):
        m.train()

    lengths, n_pad = [9, 14, 6], 35  # the pad segment is half the batch
    n_real = sum(lengths)
    torch.manual_seed(2)
    ids = torch.randint(5, _VOCAB, (n_real,))
    ids = torch.cat([ids, torch.full((n_pad,), model.config.pad_token_id)]).cuda()
    seg = lengths + [n_pad]
    cu = torch.tensor([0] + seg).cumsum(0).to(torch.int32).cuda()
    pos = torch.cat([torch.arange(n) for n in seg]).cuda()
    kwargs = dict(
        input_ids=ids, cu_seqlens=cu, max_seqlen=max(seg), position_ids=pos,
        num_valid_tokens=torch.tensor(n_real, device="cuda"),
    )

    tracker = _tracker(model, tmp_path, device="cuda")
    run = model
    if compiled:
        torch._dynamo.reset()
        run = torch.compile(model, backend="aot_eager", dynamic=False)
    ref = _MaskedReference(reference)
    ref.row_mask = torch.arange(ids.numel()) < n_real
    ref_all = _MaskedReference(unmasked)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(2):
                run(**kwargs)
                reference(**kwargs)
                unmasked(**kwargs)
        payload = tracker.flush(global_step=1)
    finally:
        tracker.close()
        ref.close()
        ref_all.close()

    expected = ref.payload()
    # bf16 activations: the tap reduces in fp32, the reference in fp64.
    got = {k: v for k, v in payload.items() if k.startswith(_STAT_PREFIXES)}
    assert set(got) == set(expected)
    for key, want in expected.items():
        torch.testing.assert_close(got[key], want, rtol=2e-3, atol=1e-4,
                                   msg=lambda m, _k=key: f"{_k}: {m}")
    _assert_mask_matters(expected, ref_all.payload())
