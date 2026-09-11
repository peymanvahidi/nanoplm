"""The captured attention probabilities must be the ones SDPA actually used."""

import pytest
import torch
import torch.nn.functional as F

from nanoplm.pretraining.models.modern_bert.config import ModernBertConfig
from nanoplm.pretraining.models.modern_bert.modeling import (
    ModernBertModel,
    collect_attention_probs,
)


def _model():
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
