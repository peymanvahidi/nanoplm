"""The HF classes must be a drop-in shell around the native module tree."""

import pytest
import torch

from nanoplm.hf import NanoPLMConfig, NanoPLMForMaskedLM, NanoPLMModel
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.moe import MoELayer
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM

TINY = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_kv_heads=2,
    vocab_size=32,
)


def _pair(**overrides):
    kwargs = {**TINY, **overrides}
    native = PureProtModernBertMLM(ProtModernBertMLMConfig(**kwargs)).eval()
    hf = NanoPLMForMaskedLM(NanoPLMConfig(**kwargs)).eval()
    return native, hf


def test_state_dict_keys_match_exactly():
    native, hf = _pair()
    assert set(native.state_dict()) == set(hf.state_dict())


def test_native_weights_load_strictly():
    native, hf = _pair()
    missing, unexpected = hf.load_state_dict(native.state_dict(), strict=True)
    assert missing == [] and unexpected == []


def test_outputs_match_native_after_loading():
    native, hf = _pair()
    hf.load_state_dict(native.state_dict(), strict=True)
    ids = torch.tensor([[4, 5, 6, 7, 1, 0], [8, 9, 10, 1, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]])
    with torch.no_grad():
        want = native(input_ids=ids, attention_mask=mask).logits
        got = hf(input_ids=ids, attention_mask=mask).logits
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_base_model_returns_hidden_states():
    cfg = NanoPLMConfig(**TINY)
    base = NanoPLMModel(cfg).eval()
    ids = torch.tensor([[4, 5, 6, 1]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        out = base(input_ids=ids, attention_mask=mask)
    assert out.last_hidden_state.shape == (1, 4, cfg.hidden_size)


def test_tied_embeddings_share_storage():
    _, hf = _pair(tie_word_embeddings=True)
    assert (
        hf.get_output_embeddings().weight.data_ptr()
        == hf.get_input_embeddings().weight.data_ptr()
    )


def test_untied_embeddings_are_separate():
    _, hf = _pair(tie_word_embeddings=False)
    assert (
        hf.get_output_embeddings().weight.data_ptr()
        != hf.get_input_embeddings().weight.data_ptr()
    )


# --- save_pretrained / from_pretrained round trip ---------------------------
#
# `_prores_alphas` and `repo_inv_freq` are non-persistent buffers, so they
# never appear in a state_dict at all (native's or the HF shell's): loading a
# checkpoint can't restore them. transformers 5.x's `from_pretrained` replaces
# every non-persistent buffer with `torch.empty_like` (garbage) after loading
# and relies on `_init_weights` to fill them back in -- these cases parametrize
# the round trip over the configs that actually exercise those buffers,
# plus a bare MoE config to prove its (persistent-looking but likewise
# non-persistent) bookkeeping isn't disturbed by the same meta-device dance.

_PRORES_KWARGS = {**TINY, "use_prores": True, "prores_T": 1000}
_REPO_KWARGS = {**TINY, "num_kv_heads": 4, "use_repo": True, "repo_after_n_layers": 0}
_MOE_KWARGS = {**TINY, "use_moe": True, "moe_num_experts": 4, "moe_top_k": 2}

_ROUND_TRIP_CASES = {
    "default": (TINY, {}),
    "prores_mid_warmup": (_PRORES_KWARGS, {"prores_step": 300}),
    "repo_active": (_REPO_KWARGS, {"repo_active": True}),
    "moe_cutlass_cpu": (_MOE_KWARGS, {}),
}


def _round_trip_pair(model_kwargs: dict, hf_extra: dict):
    native = PureProtModernBertMLM(ProtModernBertMLMConfig(**model_kwargs)).eval()
    hf = NanoPLMForMaskedLM(NanoPLMConfig(**model_kwargs, **hf_extra)).eval()
    return native, hf


@pytest.mark.parametrize(
    "model_kwargs, hf_extra",
    _ROUND_TRIP_CASES.values(),
    ids=_ROUND_TRIP_CASES.keys(),
)
def test_save_and_reload_preserves_outputs(tmp_path, model_kwargs, hf_extra):
    native, hf = _round_trip_pair(model_kwargs, hf_extra)
    hf.load_state_dict(native.state_dict(), strict=True)
    ids = torch.tensor([[4, 5, 6, 7, 1, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    with torch.no_grad():
        want = hf(input_ids=ids, attention_mask=mask).logits
    hf.save_pretrained(tmp_path, safe_serialization=True)
    reloaded = NanoPLMForMaskedLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        got = reloaded(input_ids=ids, attention_mask=mask).logits
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert (tmp_path / "model.safetensors").exists()


def test_prores_alphas_survive_round_trip(tmp_path):
    """A mid-warmup ProRes model's frozen alphas (strictly between 0 and 1)
    must come back identical after a save/reload -- not the buffer's
    just-constructed default, and not garbage from the meta-device reload."""
    hf = NanoPLMForMaskedLM(NanoPLMConfig(**_PRORES_KWARGS, prores_step=300)).eval()
    alphas_before = hf.model._prores_alphas.clone()
    assert torch.all(alphas_before > 0) and torch.all(alphas_before < 1)

    hf.save_pretrained(tmp_path, safe_serialization=True)
    reloaded = NanoPLMForMaskedLM.from_pretrained(tmp_path).eval()

    torch.testing.assert_close(
        reloaded.model._prores_alphas, alphas_before, rtol=0, atol=0
    )


def test_repo_active_and_inv_freq_survive_round_trip(tmp_path):
    """`repo_active` and each layer's `repo_inv_freq` must come back exactly
    as constructed -- both are lost by a naive load: `repo_active` because it
    is not stored anywhere the training loop would freeze it, `repo_inv_freq`
    because it is a non-persistent buffer."""
    hf = NanoPLMForMaskedLM(NanoPLMConfig(**_REPO_KWARGS, repo_active=True)).eval()
    inv_freqs_before = [layer.attn.repo_inv_freq.clone() for layer in hf.model.layers]
    assert inv_freqs_before, "expected at least one RePO-enabled layer"

    hf.save_pretrained(tmp_path, safe_serialization=True)
    reloaded = NanoPLMForMaskedLM.from_pretrained(tmp_path).eval()

    assert reloaded.model.repo_active is True
    for before, layer in zip(inv_freqs_before, reloaded.model.layers):
        torch.testing.assert_close(layer.attn.repo_inv_freq, before, rtol=0, atol=0)


# --- generic non-persistent-buffer guard (R6-2b) ---------------------------
#
# `_prores_alphas`, `repo_inv_freq`, and the RoPE tables (`inv_freq_full` /
# `inv_freq_sliding`) are all non-persistent buffers `_init_weights` must
# materialize after a `from_pretrained` reload -- R6-2 found the first two,
# R6-2b found the RoPE tables were *also* affected (every model, not gated by
# any flag) after the round trip above intermittently produced NaNs. Rather
# than add one more one-off test per buffer, this test checks *every*
# non-persistent buffer generically, so a future one added to the native tree
# without a matching `_init_weights` fix fails here by name instead of only
# occasionally (allocator-luck permitting) downstream.
#
# "Non-persistent" is derived the same way `persistent=False` is defined: a
# buffer that `named_buffers()` sees but `state_dict()` does not (the latter
# only ever contains parameters + persistent buffers).
_MOE_SCRATCH_BUFFER_NAMES = {
    # MoELayer's dispatch-workspace buffers (moe.py): constructed via plain
    # `torch.empty` even in a directly constructed native model, and fully
    # overwritten (via `_ensure_dispatch_buffers`) before ever being read.
    # Garbage after a from_pretrained reload is exactly as harmless as garbage
    # after a fresh `torch.empty` call -- checked for shape/dtype only.
    "_dispatch_sort_values",
    "_dispatch_sorted_idx",
    "_dispatch_count_ones",
    "_dispatch_counts",
}


def _non_persistent_buffer_names(model: NanoPLMForMaskedLM) -> set[str]:
    return {name for name, _ in model.named_buffers()} - set(model.state_dict().keys())


@pytest.mark.parametrize(
    "model_kwargs, hf_extra",
    _ROUND_TRIP_CASES.values(),
    ids=_ROUND_TRIP_CASES.keys(),
)
def test_every_non_persistent_buffer_survives_round_trip(
    tmp_path, model_kwargs, hf_extra
):
    hf = NanoPLMForMaskedLM(NanoPLMConfig(**model_kwargs, **hf_extra)).eval()
    buffers_before = dict(hf.named_buffers())
    non_persistent = _non_persistent_buffer_names(hf)
    assert non_persistent, "expected at least one non-persistent buffer to check"

    hf.save_pretrained(tmp_path, safe_serialization=True)
    reloaded = NanoPLMForMaskedLM.from_pretrained(tmp_path).eval()
    buffers_after = dict(reloaded.named_buffers())

    for name in sorted(non_persistent):
        before, after = buffers_before[name], buffers_after[name]
        if name.rsplit(".", 1)[-1] in _MOE_SCRATCH_BUFFER_NAMES:
            assert after.shape == before.shape, name
            assert after.dtype == before.dtype, name
        else:
            assert torch.equal(after, before), (
                f"non-persistent buffer {name!r} did not survive the "
                "save_pretrained/from_pretrained round trip"
            )


def test_moe_last_aux_count_resets_each_forward():
    """Two consecutive HF forwards must each leave every MoELayer's
    `last_aux_count` at exactly 1 -- proof the reset at the top of the HF
    forward runs every call, rather than only once ever (which would let the
    count grow unbounded across calls, as it does across Huginn/Loopie
    iterations *within* one call by design)."""
    hf = NanoPLMForMaskedLM(NanoPLMConfig(**_MOE_KWARGS)).eval()
    ids = torch.tensor([[4, 5, 6, 7, 1, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]])

    moe_layers = [m for m in hf.modules() if isinstance(m, MoELayer)]
    assert moe_layers, "expected at least one MoELayer"

    with torch.no_grad():
        hf(input_ids=ids, attention_mask=mask)
        for m in moe_layers:
            assert m.last_aux_count == 1
        hf(input_ids=ids, attention_mask=mask)
        for m in moe_layers:
            assert m.last_aux_count == 1
