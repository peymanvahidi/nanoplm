"""The captured attention probabilities must be the ones SDPA actually used."""

import contextlib

import pytest
import torch
import torch.nn.functional as F

from nanoplm.pretraining.models.modern_bert.config import ModernBertConfig
from nanoplm.pretraining.models.modern_bert.modeling import (
    ModernBertModel,
    collect_attention_probs,
)


def _model(**overrides):
    config = ModernBertConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,          # global_attn_every_n_layers=3 -> both types
        num_attention_heads=4,
        max_position_embeddings=256,
        local_attention=8,            # sliding_window = 4, small enough to bite
        use_qk_norm=True,
        norm_type="rmsnorm",
        mlp_activation="swiglu",
        **overrides,
    )
    torch.manual_seed(0)
    model = ModernBertModel(config).eval()
    return model, config


def _padded_batch(config, lengths=(17, 11)):
    seq_len = max(lengths)
    input_ids = torch.randint(4, config.vocab_size, (len(lengths), seq_len))
    attention_mask = torch.zeros(len(lengths), seq_len, dtype=torch.long)
    for row, length in enumerate(lengths):
        attention_mask[row, :length] = 1
    return input_ids, attention_mask


def _spy_compare(model, config, input_ids, attention_mask, autocast, atol):
    """Run one forward, capturing both the probabilities and SDPA's own output."""
    captured = []
    real_sdpa = F.scaled_dot_product_attention

    def _spy(q, k, v, **kwargs):
        out = real_sdpa(q, k, v, **kwargs)
        captured.append((v.detach(), out.detach()))
        return out

    F.scaled_dot_product_attention = _spy
    try:
        with torch.no_grad(), autocast, collect_attention_probs(
            model, dtype=torch.float32
        ) as probs:
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        F.scaled_dot_product_attention = real_sdpa

    assert len(probs) == config.num_hidden_layers
    real = attention_mask.bool().cpu()
    for layer, (p, (v, out)) in enumerate(zip(probs, captured)):
        # the capture lands on the CPU; bring the kernel's own tensors along
        p, v, out = p.float(), v.cpu(), out.cpu()
        if v.shape[1] != p.shape[1]:  # GQA: SDPA broadcasts v across the group
            v = v.repeat_interleave(p.shape[1] // v.shape[1], dim=1)
        recomputed = torch.matmul(p, v.float())
        keep = real[:, None, :, None].expand_as(recomputed)
        assert torch.allclose(
            recomputed[keep], out.float()[keep.expand_as(out)], atol=atol
        ), (
            f"layer {layer} ({config.layer_types[layer]}) diverges: max abs "
            f"{(recomputed[keep] - out.float()[keep.expand_as(out)]).abs().max().item()}"
        )


@pytest.mark.parametrize("num_kv_heads", [None, 2])
def test_probs_reproduce_the_sdpa_output_gqa(num_kv_heads):
    """The GQA branch repeats k by hand; SDPA does it with enable_gqa."""
    model, config = _model(num_kv_heads=num_kv_heads)
    input_ids, attention_mask = _padded_batch(config)
    _spy_compare(
        model,
        config,
        input_ids,
        attention_mask,
        contextlib.nullcontext(),
        atol=1e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_probs_reproduce_the_sdpa_output_under_autocast():
    """The same equivalence, under the autocast the real eval actually runs in.

    The tolerance here is the kernel's own bf16 output rounding, so this case
    checks the mask, the scale and the dtype alignment under autocast -- not
    whether the softmax is accumulated in fp32, which it is far too coarse to
    see. `test_probs_are_autocast_invariant` is what pins that.
    """
    device = torch.device("cuda")
    model, config = _model()
    model = model.to(device)
    input_ids, attention_mask = _padded_batch(config)
    _spy_compare(
        model,
        config,
        input_ids.to(device),
        attention_mask.to(device),
        torch.amp.autocast("cuda", dtype=torch.bfloat16),
        # bounded by SDPA's own bf16 output rounding, not by the scores
        atol=2e-2,
    )


def test_probs_reproduce_the_sdpa_output():
    """probs @ v == SDPA(q, k, v) validates scale, mask and GQA in one shot.

    Comparing against the kernel's own output is the only check that fails if
    the recomputed mask or scale is wrong; probabilities that merely look
    plausible (rows summing to one, banded structure) would pass regardless.
    """
    model, config = _model()
    input_ids, attention_mask = _padded_batch(config)

    captured = []
    real_sdpa = F.scaled_dot_product_attention

    def _spy(q, k, v, **kwargs):
        out = real_sdpa(q, k, v, **kwargs)
        captured.append((v.detach(), out.detach()))
        return out

    F.scaled_dot_product_attention = _spy
    try:
        # fp32 capture: at the default fp16 the rounding of the probabilities
        # alone moves the product by ~2e-4, which would force a tolerance loose
        # enough to hide a genuinely wrong mask.
        with torch.no_grad(), collect_attention_probs(
            model, dtype=torch.float32
        ) as probs:
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        F.scaled_dot_product_attention = real_sdpa

    assert len(probs) == config.num_hidden_layers
    assert len(captured) == config.num_hidden_layers

    layer_types = config.layer_types
    assert "full_attention" in layer_types and "sliding_attention" in layer_types

    real = attention_mask.bool()
    for layer, (p, (v, out)) in enumerate(zip(probs, captured)):
        p = p.float()
        assert p.shape == (*out.shape[:3], out.shape[2]), layer
        recomputed = torch.matmul(p, v.float())
        # Real residues only: a padded query whose whole window is padding is
        # NaN out of SDPA and zero out of the capture, deliberately.
        recomputed = recomputed[real[:, None, :, None].expand_as(recomputed)]
        reference = out.float()[real[:, None, :, None].expand_as(out)]
        assert torch.allclose(recomputed, reference, atol=1e-5), (
            f"layer {layer} ({layer_types[layer]}) diverges: "
            f"max abs {(recomputed - reference).abs().max().item()}"
        )


def test_sliding_layers_stay_inside_the_window():
    """A window off by one would still reproduce the output if v were smooth."""
    model, config = _model()
    input_ids, attention_mask = _padded_batch(config, lengths=(17, 17))

    with torch.no_grad(), collect_attention_probs(model) as probs:
        model(input_ids=input_ids, attention_mask=attention_mask)

    window = config.sliding_window
    pos = torch.arange(input_ids.shape[1])
    outside = (pos[:, None] - pos[None, :]).abs() > window

    for layer, p in enumerate(probs):
        p = p.float()
        torch.testing.assert_close(
            p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-3, rtol=0
        )  # lengths are equal here, so every query has a non-empty window
        if config.layer_types[layer] == "sliding_attention":
            assert p[..., outside].abs().max() == 0.0, layer
        else:
            assert p[..., outside].abs().max() > 0.0, layer


def test_padding_is_never_attended():
    model, config = _model()
    lengths = (17, 11)
    input_ids, attention_mask = _padded_batch(config, lengths=lengths)

    with torch.no_grad(), collect_attention_probs(model) as probs:
        model(input_ids=input_ids, attention_mask=attention_mask)

    for layer, p in enumerate(probs):
        real = p[1, :, : lengths[1], :].float()
        assert real[..., lengths[1]:].abs().max() == 0.0, layer


def test_sink_is_off_by_default():
    """The training path must not pay for, or be perturbed by, the capture."""
    model, config = _model()
    input_ids, attention_mask = _padded_batch(config)

    with torch.no_grad(), collect_attention_probs(model):
        pass
    with torch.no_grad():
        baseline = model(input_ids=input_ids, attention_mask=attention_mask)
    with torch.no_grad(), collect_attention_probs(model) as probs:
        captured_run = model(input_ids=input_ids, attention_mask=attention_mask)

    assert len(probs) == config.num_hidden_layers
    torch.testing.assert_close(baseline, captured_run)

    attn = next(iter(model.layers)).attn
    assert attn._attn_probs_sink is None
    assert "_attn_probs_sink" not in model.state_dict()


def test_fp16_capture_stays_within_rounding():
    """The default fp16 storage must cost rounding only, not structure."""
    model, config = _model()
    input_ids, attention_mask = _padded_batch(config)

    with torch.no_grad(), collect_attention_probs(model) as half:
        model(input_ids=input_ids, attention_mask=attention_mask)
    with torch.no_grad(), collect_attention_probs(
        model, dtype=torch.float32
    ) as full:
        model(input_ids=input_ids, attention_mask=attention_mask)

    for layer, (h, f) in enumerate(zip(half, full)):
        assert h.dtype == torch.float16, layer
        assert f.dtype == torch.float32, layer
        torch.testing.assert_close(h.float(), f, atol=1e-3, rtol=0)


def test_empty_window_rows_are_zero_not_nan():
    """NaN in one padded row would spread over the whole map through APC."""
    model, config = _model()
    input_ids, attention_mask = _padded_batch(config, lengths=(17, 3))

    with torch.no_grad(), collect_attention_probs(model) as probs:
        model(input_ids=input_ids, attention_mask=attention_mask)

    window = config.sliding_window
    for layer, p in enumerate(probs):
        assert torch.isfinite(p.float()).all(), layer
    # the construction is only meaningful if some query really has no keys
    assert 3 + window < 17


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_probs_are_autocast_invariant():
    """Same q/k in, same probabilities out, inside an autocast region or not.

    This is the invariant that keeps the probe's features equal to what the
    model attended with: SDPA accumulates its softmax in fp32, but inside an
    enabled autocast region a matmul silently casts its inputs back to bf16
    however they were spelled, so the recompute has to switch autocast off.
    Measured, the omission moves probabilities by up to 3e-3 absolute -- an
    order of magnitude past the fp16 rounding they are then stored at, and too
    small for an output-level comparison to resolve, which is why it is checked
    here rather than through `probs @ v`.
    """
    from nanoplm.pretraining.models.modern_bert.modeling import ModernBertAttention

    model, config = _model()
    attention = next(
        m for m in model.modules() if isinstance(m, ModernBertAttention)
    ).cuda()

    torch.manual_seed(0)
    shape = (1, config.num_attention_heads, 24, config.head_dim)
    q = torch.randn(shape, device="cuda")
    k = torch.randn(shape, device="cuda")
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        outside = attention._eager_attention_probs(q, k, v, None)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            inside = attention._eager_attention_probs(q, k, v, None)

    assert outside.dtype == torch.float32
    torch.testing.assert_close(inside, outside, atol=0, rtol=0)


def test_unpair_attention_probs_indexing():
    """Head 2p + a, query token i, key token j <- pair p, row 2i + a, cols 2j and 2j + 1."""
    from nanoplm.pretraining.models.modern_bert.modeling import _unpair_attention_probs

    b, heads, s = 2, 4, 5
    torch.manual_seed(0)
    maps = torch.softmax(torch.randn(b, heads, s, s), dim=-1)
    w = torch.tensor([0.3, 0.7])
    paired = torch.zeros(b, heads // 2, 2 * s, 2 * s)
    for p in range(heads // 2):
        for a in range(2):
            for i in range(s):
                for j in range(s):
                    for slot in range(2):
                        paired[:, p, 2 * i + a, 2 * j + slot] = maps[:, 2 * p + a, i, j] * w[slot]
    assert torch.allclose(_unpair_attention_probs(paired), maps, atol=1e-6)


def test_paired_head_probs_reproduce_the_sdpa_output(monkeypatch):
    """The pair-space probabilities must reproduce SDPA, and the captured maps
    must be their per-head reduction (rows summing to 1 on real tokens)."""
    from nanoplm.pretraining.models.modern_bert import modeling

    model, config = _model(use_paired_head_attention=True)
    input_ids, attention_mask = _padded_batch(config)
    paired_probs, captured = [], []
    real_unpair, real_sdpa = modeling._unpair_attention_probs, F.scaled_dot_product_attention

    def _unpair(p):
        paired_probs.append(p.detach())
        return real_unpair(p)

    def _spy(q, k, v, **kwargs):
        out = real_sdpa(q, k, v, **kwargs)
        captured.append((v.detach(), out.detach()))
        return out

    monkeypatch.setattr(modeling, "_unpair_attention_probs", _unpair)
    monkeypatch.setattr(F, "scaled_dot_product_attention", _spy)
    with torch.no_grad(), collect_attention_probs(model, dtype=torch.float32) as probs:
        model(input_ids=input_ids, attention_mask=attention_mask)

    assert len(probs) == len(paired_probs) == len(captured) == config.num_hidden_layers
    real = attention_mask.bool()
    real2 = real.repeat_interleave(2, dim=-1)
    for layer, (p, pp, (v, out)) in enumerate(zip(probs, paired_probs, captured)):
        assert p.shape == (input_ids.shape[0], config.num_attention_heads) + input_ids.shape[1:] * 2
        recomputed = torch.matmul(pp.float(), v.float())
        keep = real2[:, None, :, None].expand_as(recomputed)
        assert torch.allclose(recomputed[keep], out.float()[keep], atol=1e-5), layer
        rows = p.sum(-1)[real[:, None, :].expand(-1, p.shape[1], -1)]
        assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5), layer
