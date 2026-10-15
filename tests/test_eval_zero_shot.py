"""Tests for the biotrainer zero-shot BioEngineer wrapper in ``nanoplm.eval``.

These cover what nanoPLM adds on top of biotrainer 2.1:

* batched masked-marginals, sized by the token budget rather than biotrainer's
  fixed batch,
* the categorical Jacobian through nanoPLM's tokenizer, which has no BOS token
  and may append register tokens, at the configured batch size.

They are skipped unless the ``eval`` extra (biotrainer 2.x) is installed.
"""

import pytest
import torch

pytest.importorskip("biotrainer", reason="requires the nanoplm[eval] extra")
pytest.importorskip("biotrainer_core", reason="requires the nanoplm[eval] extra")

from nanoplm.eval.pipeline import _load_biotrainer_symbols  # noqa: E402
from nanoplm.eval.zero_shot import (  # noqa: E402
    TOKEN_BUDGET,
    NanoPLMBioEngineer,
    special_token_offsets,
)
from nanoplm.pretraining.models.modern_bert.model import (  # noqa: E402
    ProtModernBertMLMConfig,
)
from nanoplm.pretraining.models.modern_bert.pure_model import (  # noqa: E402
    PureProtModernBertMLM,
)
from nanoplm.pretraining.models.modern_bert.tokenizer import (  # noqa: E402
    ProtModernBertTokenizer,
)

SEQUENCE = "MKALCLLLLPVLGLLTGSSGSAWGKVGAHA"
CPU = torch.device("cpu")


def _build_engineer(num_register_tokens: int = 0, batch_size: int = 60):
    """A tiny dense (non-MoE) model wired through the real BioEngineer stack."""
    torch.manual_seed(0)
    tokenizer = ProtModernBertTokenizer(num_register_tokens=num_register_tokens)
    config = ProtModernBertMLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=tokenizer.vocab_size,
        num_register_tokens=num_register_tokens,
    )
    model = PureProtModernBertMLM(config).to(CPU).eval()
    biotrainer = _load_biotrainer_symbols()
    engineer = NanoPLMBioEngineer.build(
        model, tokenizer, CPU, biotrainer, batch_size=batch_size
    )
    return engineer, model, tokenizer, biotrainer


# ---------------------------------------------------------------------------
# Special-token offsets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_bos", [False, True])
@pytest.mark.parametrize("registers", [0, 1, 2])
def test_special_token_offsets_match_the_tokenizer(use_bos, registers):
    tokenizer = ProtModernBertTokenizer(
        use_bos_token=use_bos, num_register_tokens=registers
    )
    leading, trailing = special_token_offsets(tokenizer)

    encoded = tokenizer([SEQUENCE], return_tensors="pt")["input_ids"]
    assert encoded.shape[1] == leading + len(SEQUENCE) + trailing


def test_strip_special_tokens_keeps_only_residue_positions():
    """The old implementation hardcoded ``tensor[0:-1]``, which kept register logits."""
    engineer, _, tokenizer, _ = _build_engineer(num_register_tokens=2)
    wrapper = engineer.model_wrapper

    encoded = tokenizer([SEQUENCE], return_tensors="pt")["input_ids"][0]
    stripped = wrapper._strip_special_tokens(encoded)

    assert stripped.shape[0] == len(SEQUENCE)
    # Every surviving token is a standard amino acid, no EOS / register tokens.
    assert set(stripped.tolist()) <= set(
        ProtModernBertTokenizer.STANDARD_AA_TOKEN_IDS
        | ProtModernBertTokenizer.NON_STANDARD_AA_TOKEN_IDS
    )


# ---------------------------------------------------------------------------
# Batched masked-marginals
# ---------------------------------------------------------------------------

def _sequential_masked_log_probs(wrapper, sequence: str) -> torch.Tensor:
    """biotrainer's unbatched reference implementation, one forward per position."""
    from biotrainer.bioengineer.models import (
        CustomBioEngineerModelWrapper,
    )

    # CustomBioEngineerModelWrapper does not define this method, so the attribute
    # resolves to BertLikeEngineer's implementation, the code path our override
    # replaces. A batch of 1 makes it one forward per masked position.
    return CustomBioEngineerModelWrapper._get_masked_log_probabilities(
        wrapper, sequence, 1
    )


@pytest.mark.parametrize("batch_size", [7, 60])
def test_batched_masked_marginals_match_the_sequential_path(batch_size):
    engineer, _, _, _ = _build_engineer(batch_size=batch_size)
    wrapper = engineer.model_wrapper

    batched = wrapper._get_masked_log_probabilities(SEQUENCE)
    sequential = _sequential_masked_log_probs(wrapper, SEQUENCE)

    assert batched.shape == sequential.shape == (len(SEQUENCE), wrapper._custom_bioengineer._tokenizer.vocab_size)
    max_diff = (batched - sequential).abs().max().item()
    assert torch.allclose(batched, sequential, atol=1e-5, rtol=1e-4), (
        f"batched masked-marginals diverged from the per-position path "
        f"(max |delta| = {max_diff})"
    )


def test_batched_masked_marginals_cover_register_token_models():
    engineer, _, _, _ = _build_engineer(num_register_tokens=2, batch_size=9)
    wrapper = engineer.model_wrapper

    batched = wrapper._get_masked_log_probabilities(SEQUENCE)
    sequential = _sequential_masked_log_probs(wrapper, SEQUENCE)

    assert batched.shape[0] == len(SEQUENCE)
    assert torch.allclose(batched, sequential, atol=1e-5, rtol=1e-4)


def test_chunk_size_shrinks_for_long_windows():
    engineer, _, _, _ = _build_engineer(batch_size=60)
    wrapper = engineer.model_wrapper

    assert wrapper._chunk_size(256) == 60          # token budget not binding
    assert wrapper._chunk_size(TOKEN_BUDGET) == 1      # one row already fills the budget
    assert wrapper._chunk_size(2 * TOKEN_BUDGET) == 1  # never drops below 1


def test_public_masked_marginals_match_the_sequential_path():
    """Through BioEngineer, as AutoEval's pgym calls it.

    biotrainer passes ``batch_size`` positionally to the override, so a
    signature that drops it fails here rather than on every pgym assay.
    """
    engineer, _, _, biotrainer = _build_engineer(batch_size=7)
    wrapper = engineer.model_wrapper
    mutations = ["M1A", "K2R", "A3W:L4P"]

    scores = engineer.zero_shot_masked_marginals(SEQUENCE, mutations)
    reference = wrapper._score_variants_from_marginal_probabilities(
        SEQUENCE,
        _sequential_masked_log_probs(wrapper, SEQUENCE),
        mutations,
        True,
        biotrainer["ZeroShotMethod"].MASKED_MARGINALS,
    )

    assert len(scores) == len(mutations)
    assert [s.total_score for s in scores] == pytest.approx(
        [r.total_score for r in reference], abs=1e-5, rel=1e-4
    )


# ---------------------------------------------------------------------------
# Categorical Jacobian / zero-shot contact
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("registers", [0, 2])
def test_categorical_jacobian_has_residue_aligned_shape(registers):
    """No BOS and trailing register tokens must still give an L x 20 x L x 20 map."""
    engineer, _, _, _ = _build_engineer(num_register_tokens=registers, batch_size=64)
    wrapper = engineer.model_wrapper

    short = SEQUENCE[:12]
    jac = wrapper._compute_categorical_jacobian(short)

    assert jac.shape == (len(short), 20, len(short), 20)


def test_categorical_jacobian_is_aligned_to_the_right_residue():
    """Mutating residue p must be the row the Jacobian attributes to p.

    biotrainer releases before PR #192 got this wrong for a tokenizer without
    BOS: they wrote mutations at column ``n+1`` and sliced logits ``[1:-1]``.
    """
    engineer, _, tokenizer, biotrainer = _build_engineer(batch_size=64)
    wrapper = engineer.model_wrapper
    STANDARD_AAS = biotrainer["STANDARD_AAS"]

    short = SEQUENCE[:10]
    jac = wrapper._compute_categorical_jacobian(short)

    n_leading, _ = special_token_offsets(tokenizer)
    aa_token_ids = torch.tensor(list(wrapper.aa_to_idx().values()))

    input_ids, attention_mask = wrapper._tokenize([short], preprocess=True)
    reference = wrapper._model_forward_fn(input_ids, attention_mask)
    reference = reference[n_leading : n_leading + len(short)][:, aa_token_ids]

    position, aa_index = 3, STANDARD_AAS.index("W")
    mutated = input_ids.clone()
    mutated[0, n_leading + position] = aa_token_ids[aa_index]
    mutated_logits = wrapper._model_forward_fn(mutated, attention_mask)
    mutated_logits = mutated_logits[n_leading : n_leading + len(short)][:, aa_token_ids]

    expected = mutated_logits - reference
    assert torch.allclose(jac[position, aa_index], expected, atol=1e-5, rtol=1e-4)


def test_contact_map_collapses_to_l_by_l():
    from biotrainer.bioengineer.bioengineer_utils import convert_cat_jac_to_contact_map

    engineer, _, _, _ = _build_engineer(batch_size=64)
    short = SEQUENCE[:12]

    jac = engineer.model_wrapper._compute_categorical_jacobian(short)
    contacts = convert_cat_jac_to_contact_map(jac.numpy())

    assert contacts.shape == (len(short), len(short))


def test_jacobian_rejects_a_tokenization_it_cannot_align():
    engineer, _, _, _ = _build_engineer()
    wrapper = engineer.model_wrapper

    # A sequence whose preprocessed length still matches, but hand a mismatched
    # one through the private path to prove the guard fires rather than
    # producing a silently misaligned map.
    original_tokenize = wrapper._tokenize
    wrapper._tokenize = lambda batch, preprocess=False: (
        original_tokenize(batch, preprocess)[0][:, :-1],
        original_tokenize(batch, preprocess)[1][:, :-1],
    )
    try:
        with pytest.raises(ValueError, match="disagree about which tokens are special"):
            wrapper._compute_categorical_jacobian(SEQUENCE[:8])
    finally:
        wrapper._tokenize = original_tokenize


def test_jacobian_contact_is_a_supported_method():
    """pbc_zeroshot_contact is rejected outright without this."""
    engineer, _, _, biotrainer = _build_engineer()
    ZeroShotMethod = biotrainer["ZeroShotMethod"]

    assert ZeroShotMethod.JACOBIAN_CONTACT in engineer.model_wrapper.supported_methods()


@pytest.mark.parametrize("registers, expected", [(0, 1023), (1, 1024), (2, 1024)])
def test_context_caps_residues_at_1022_and_tokens_at_1024(registers, expected):
    engineer, _, _, _ = _build_engineer(num_register_tokens=registers)

    assert engineer.model_wrapper.max_context_length() == expected


def test_jacobian_refuses_a_protein_past_the_context():
    engineer, _, _, biotrainer = _build_engineer()

    with pytest.raises(biotrainer["SequenceTooLongError"]):
        engineer.zero_shot_contact_map(
            biotrainer["ZeroShotMethod"].JACOBIAN_CONTACT, "A" * 1100
        )
