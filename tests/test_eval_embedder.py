"""Tests for the biotrainer ``CustomEmbedder`` wrapper in ``nanoplm.eval``.

The supervised frameworks (``pbc_supervised``) train task heads on a frozen
encoder's embeddings, so they go through ``CustomEmbedder`` rather than the
zero-shot ``BioEngineer``. What has to hold: the per-residue embedding is
residue-aligned (biotrainer rejects any other shape), the per-sequence one is
its mean, and a Loopie refinement-depth override reaches the forward pass.

They are skipped unless the ``eval`` extra (biotrainer 2.x) is installed.
"""

import click
import pytest
import torch

pytest.importorskip("biotrainer", reason="requires the nanoplm[eval] extra")
pytest.importorskip("biotrainer_core", reason="requires the nanoplm[eval] extra")

from nanoplm.cli.eval import _validate_frameworks  # noqa: E402
from nanoplm.eval import embedder as embedder_module  # noqa: E402
from nanoplm.eval.embedder import NanoPLMCustomEmbedder  # noqa: E402
from nanoplm.eval.pipeline import _load_biotrainer_symbols  # noqa: E402
from nanoplm.eval.zero_shot import special_token_offsets  # noqa: E402
from nanoplm.pretraining.models.modern_bert.model import (  # noqa: E402
    ProtModernBertMLMConfig,
)
from nanoplm.pretraining.models.modern_bert.pure_model import (  # noqa: E402
    PureProtModernBertMLM,
)
from nanoplm.pretraining.models.modern_bert.tokenizer import (  # noqa: E402
    ProtModernBertTokenizer,
)

SEQUENCES = [
    "MKALCLLLLPVLGLLTGSSGSAWGKVGAHA",
    "MSTNPKPQRK",
    "GGGSGGGSGGGSAAAWWW",
]
CPU = torch.device("cpu")


def _model(num_register_tokens: int = 0, **overrides):
    torch.manual_seed(0)
    tokenizer = ProtModernBertTokenizer(num_register_tokens=num_register_tokens)
    config = ProtModernBertMLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=tokenizer.vocab_size,
        num_register_tokens=num_register_tokens,
        **overrides,
    )
    return PureProtModernBertMLM(config).to(CPU).eval(), tokenizer


def _perturb(model):
    """Push every parameter off its init so no layer is accidentally identity."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)


def _build_embedder(num_register_tokens: int = 0, batch_size: int = 60, **overrides):
    model, tokenizer = _model(num_register_tokens, **overrides)
    embedder = NanoPLMCustomEmbedder.build(
        model, tokenizer, CPU, _load_biotrainer_symbols(), batch_size=batch_size
    )
    return embedder, model, tokenizer


# ---------------------------------------------------------------------------
# Shapes and special-token stripping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("registers", [0, 2])
@pytest.mark.parametrize("batch_size", [1, 2, 60])
def test_per_residue_is_residue_aligned(registers, batch_size):
    """biotrainer raises unless the embedding has exactly one row per residue."""
    embedder, model, _ = _build_embedder(
        num_register_tokens=registers, batch_size=batch_size
    )

    embedded = list(embedder.per_residue(SEQUENCES))

    assert [sequence for sequence, _ in embedded] == SEQUENCES
    for sequence, embedding in embedded:
        assert embedding.shape == (len(sequence), model.config.hidden_size)
        assert embedding.dtype == torch.float32
        assert embedding.device.type == "cpu"


def test_per_residue_is_unaffected_by_batch_padding():
    """Padding rows to the longest sequence must not change shorter rows."""
    embedder, _, _ = _build_embedder(batch_size=60)

    batched = dict(embedder.per_residue(SEQUENCES))
    one_at_a_time = {
        sequence: embedding
        for sequence in SEQUENCES
        for _, embedding in embedder.per_residue([sequence])
    }

    for sequence in SEQUENCES:
        torch.testing.assert_close(
            batched[sequence], one_at_a_time[sequence], rtol=1e-5, atol=1e-6
        )


def test_token_budget_splits_a_batch_the_row_count_would_allow(monkeypatch):
    """Supervised benchmarks embed 2000-residue sequences; rows alone are no bound."""
    embedder, model, _ = _build_embedder(batch_size=60)
    unsplit = dict(embedder.per_residue(SEQUENCES))

    forwards = []
    encoder_forward = model.model.forward

    def _counting_forward(**kwargs):
        forwards.append(kwargs["input_ids"].shape[0])
        return encoder_forward(**kwargs)

    monkeypatch.setattr(embedder_module, "TOKEN_BUDGET", 40)
    monkeypatch.setattr(model.model, "forward", _counting_forward)
    split = dict(embedder.per_residue(SEQUENCES))

    assert len(forwards) > 1, "the token budget never split the batch"
    for sequence in SEQUENCES:
        torch.testing.assert_close(split[sequence], unsplit[sequence])


def test_per_sequence_is_the_mean_over_residues():
    embedder, model, _ = _build_embedder()

    per_residue = dict(embedder.per_residue(SEQUENCES))
    per_sequence = list(embedder.per_sequence(SEQUENCES))

    assert [sequence for sequence, _ in per_sequence] == SEQUENCES
    for sequence, embedding in per_sequence:
        assert embedding.shape == (model.config.hidden_size,)
        torch.testing.assert_close(embedding, per_residue[sequence].mean(dim=0))


def test_embedding_is_the_final_normed_hidden_state():
    """The probe embeds the normalised encoder output, not the pre-norm state.

    Stubbing ``final_norm`` out exposes what the encoder feeds it, so the
    reference is the last layer's raw output rather than a restatement of the
    implementation.
    """
    embedder, model, tokenizer = _build_embedder()
    _perturb(model)
    sequence = SEQUENCES[1]

    _, embedding = next(iter(embedder.per_residue([sequence])))

    encoded = tokenizer([sequence], return_tensors="pt")
    encoder = model.model
    final_norm = encoder.final_norm
    encoder.final_norm = torch.nn.Identity()
    try:
        with torch.no_grad():
            pre_norm = encoder(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
    finally:
        encoder.final_norm = final_norm
    with torch.no_grad():
        normed = final_norm(pre_norm)

    n_leading, _ = special_token_offsets(tokenizer)
    residues = slice(n_leading, n_leading + len(sequence))
    torch.testing.assert_close(embedding, normed[0, residues], rtol=0, atol=0)
    assert (embedding - pre_norm[0, residues]).abs().max() > 1e-4


# ---------------------------------------------------------------------------
# Loopie refinement depth
# ---------------------------------------------------------------------------

def _embed_at(loop_count, model, tokenizer):
    embedder = NanoPLMCustomEmbedder.build(
        model, tokenizer, CPU, _load_biotrainer_symbols(), loop_count=loop_count
    )
    return dict(embedder.per_residue(SEQUENCES))


def test_loop_count_changes_the_embedding():
    model, tokenizer = _model(use_loopie=True, loopie_loop_count=2)
    _perturb(model)

    at_two = _embed_at(2, model, tokenizer)
    at_three = _embed_at(3, model, tokenizer)

    for sequence in SEQUENCES:
        assert (at_two[sequence] - at_three[sequence]).abs().max() > 1e-4, (
            f"depth 2 and depth 3 embedded {sequence!r} identically"
        )


@pytest.mark.parametrize("depth", [2, 3])
def test_loop_count_matches_a_model_built_at_that_depth(depth):
    model, tokenizer = _model(use_loopie=True, loopie_loop_count=2)
    _perturb(model)
    reference, _ = _model(use_loopie=True, loopie_loop_count=depth)
    reference.load_state_dict(model.state_dict())

    overridden = _embed_at(depth, model, tokenizer)
    built_in = _embed_at(None, reference, tokenizer)

    for sequence in SEQUENCES:
        torch.testing.assert_close(
            overridden[sequence], built_in[sequence], rtol=0, atol=0
        )


# ---------------------------------------------------------------------------
# Framework validation
# ---------------------------------------------------------------------------

def test_pbc_supervised_is_an_accepted_framework():
    _validate_frameworks(
        [
            "pgym",
            "pbc_zeroshot_contact",
            "pbc_supervised",
            "pbc_supervised_contact",
        ]
    )


@pytest.mark.parametrize("framework", ["flip", "pbc"])
def test_removed_frameworks_are_still_rejected(framework):
    with pytest.raises(click.ClickException, match="no longer supported"):
        _validate_frameworks([framework])


def test_attention_map_is_residue_aligned():
    """The probe indexes the map with an [L, L] mask, so L must be exact.

    Register tokens are the trap: they sit after the EOS, so trimming a single
    trailing token would leave them in the map as extra residues and silently
    shift every pair the probe trains on.
    """
    embedder, model, tokenizer = _build_embedder(num_register_tokens=2)
    _perturb(model)
    sequence = SEQUENCES[0]

    attention_map = embedder.compute_attention_map(sequence=sequence)

    layers = model.config.num_hidden_layers
    heads = model.config.num_attention_heads
    assert attention_map.shape == (len(sequence), len(sequence), layers * heads)
    assert torch.isfinite(attention_map.float()).all()


def test_attention_map_is_symmetric_across_residues():
    """Symmetrization is part of the contract; contacts are undirected."""
    embedder, model, _ = _build_embedder()
    _perturb(model)

    attention_map = embedder.compute_attention_map(sequence=SEQUENCES[1]).float()

    torch.testing.assert_close(
        attention_map, attention_map.transpose(0, 1), atol=1e-3, rtol=0
    )


def test_attention_map_matches_the_reference_recipe():
    """Same numbers as biotrainer's embedder would get from these probabilities.

    Guards against the post-processing drifting: order (strip, symmetrize, then
    APC) and the APC correction itself both change the features the probe sees.
    """
    from biotrainer.embedding.huggingface.huggingface_transformer_embedder import (
        HuggingfaceTransformerEmbedder as reference,
    )
    from nanoplm.pretraining.models.modern_bert.modeling import (
        collect_attention_probs,
    )

    embedder, model, tokenizer = _build_embedder(num_register_tokens=1)
    _perturb(model)
    sequence = SEQUENCES[2]
    n_leading, n_trailing = special_token_offsets(tokenizer)

    encoded = tokenizer([sequence], return_tensors="pt")
    with torch.no_grad(), collect_attention_probs(model.model) as per_layer:
        model.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
    end = encoded["input_ids"].shape[-1] - n_trailing
    expected = torch.cat([p[0] for p in per_layer], dim=0).float()
    expected = expected[:, n_leading:end, n_leading:end]
    expected = reference._apc(reference._symmetrize(expected)).permute(1, 2, 0)

    actual = embedder.compute_attention_map(sequence=sequence).float()
    torch.testing.assert_close(actual, expected.half().float(), atol=1e-3, rtol=0)


def test_attention_map_leaves_no_capture_state_behind():
    """A leaked sink would make every later training forward allocate an S x S."""
    embedder, model, _ = _build_embedder()
    embedder.compute_attention_map(sequence=SEQUENCES[1])

    from nanoplm.pretraining.models.modern_bert.modeling import ModernBertAttention

    for module in model.modules():
        if isinstance(module, ModernBertAttention):
            assert module._attn_probs_sink is None


# ---------------------------------------------------------------------------
# PBC_SUPERVISED dataset filter
# ---------------------------------------------------------------------------

def test_pbc_supervised_datasets_defaults_to_the_battery():
    """The battery wants two of the seven heads; the rest cost 13 min and 30 GB."""
    from nanoplm.eval.config import EvalConfig

    config = EvalConfig(checkpoint_dir="unused")
    assert config.pbc_supervised_datasets == ["secondary_structure", "scl"]


def test_pbc_supervised_filter_selects_only_the_named_datasets():
    """Mirrors the predicate pipeline.py builds, against biotrainer's own shape.

    `secondary_structure` is what carries newPISCES364, so a filter that drops
    it silently removes the metric rather than failing loudly.
    """
    wanted = {"secondary_structure", "scl"}
    task_filter = lambda task: task.dataset_name in wanted  # noqa: E731

    class _Task:
        def __init__(self, dataset_name):
            self.dataset_name = dataset_name

    every = [
        "conservation", "disorder_chezod", "frustration-classification",
        "frustration-regression", "phages", "scl", "secondary_structure",
    ]
    kept = [name for name in every if task_filter(_Task(name))]
    assert kept == ["scl", "secondary_structure"]
