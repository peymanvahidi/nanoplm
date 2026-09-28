"""Export verification. Runs on CPU with a dense tiny model; GPU-gated tests
(marked below) additionally cover the CUDA and FP8 paths on real hardware.
"""

import copy

import pytest
import torch

from nanoplm.hf import NanoPLMConfig, NanoPLMForMaskedLM
from nanoplm.hf.vendor import (
    AUTO_MAP,
    assert_no_unguarded_optional_imports,
    vendor_modeling_code,
)
from nanoplm.hf.verify import (
    VERIFY_SEQUENCES,
    ExportVerificationError,
    verify_exported,
)
from nanoplm.pretraining.fp8 import convert_fp8_targets, restore_dense_from_te
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer

WITH_CUDA = torch.cuda.is_available()
try:
    import transformer_engine.pytorch as te

    WITH_TE = True
except Exception:  # pragma: no cover - depends on the runtime image
    te = None
    WITH_TE = False

TINY = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_kv_heads=2,
    vocab_size=32,
)
MOE = dict(use_moe=True, moe_num_experts=4, moe_top_k=2, moe_kernel_backend="cutlass")
RECYCLING = dict(
    num_hidden_layers=3,
    use_huginn_looping=True,
    recycling_loop_count=2,
    recycling_num_prelude_layers=1,
    recycling_num_coda_layers=1,
    # Default (nonzero) recycling_s0_noise_sigma left as-is on purpose: this
    # is exactly the case that fails without verify_exported's RNG-forking.
)


def test_verify_sequences_cover_the_awkward_cases():
    assert len(VERIFY_SEQUENCES) >= 3
    assert any(len(s) > 60 for s in VERIFY_SEQUENCES), "need a long sequence"
    assert any(set("XBOUZ") & set(s) for s in VERIFY_SEQUENCES), "need non-standard AAs"
    assert any(len(s) < 15 for s in VERIFY_SEQUENCES), "need a short sequence"


def _export(config_kwargs, out_dir, *, seed=0):
    """Build a native model and its vendored, gated export -- the shape of a
    real export (R12-1's helper additions: a saved tokenizer, vendored
    modeling code, and auto_map set before save_pretrained)."""
    torch.manual_seed(seed)
    native = PureProtModernBertMLM(
        ProtModernBertMLMConfig(**{**TINY, **config_kwargs})
    ).eval()
    hf = NanoPLMForMaskedLM(NanoPLMConfig(**{**TINY, **config_kwargs})).eval()
    hf.load_state_dict(native.state_dict(), strict=True)
    hf.config.auto_map = AUTO_MAP
    hf.save_pretrained(out_dir, safe_serialization=True)
    ProtModernBertTokenizer(num_register_tokens=0).save_pretrained(out_dir)
    vendor_modeling_code(out_dir)
    assert_no_unguarded_optional_imports(out_dir)
    return native


def _export_tiny(tmp_path):
    return _export({}, tmp_path), tmp_path


def _native_tokenizer():
    return ProtModernBertTokenizer(num_register_tokens=0)


def test_matching_export_passes(tmp_path):
    native, out = _export_tiny(tmp_path)
    result = verify_exported(
        native,
        out,
        tokenizer=_native_tokenizer(),
        device=torch.device("cpu"),
        fp8_source=False,
    )
    assert result.passed
    assert result.max_logit_diff == pytest.approx(0.0, abs=1e-6)


def test_perturbed_export_is_rejected(tmp_path):
    native, out = _export_tiny(tmp_path)
    # Corrupt one weight the way a bad gate/up swap would. Scaling rather than
    # shifting: the encoder's final norm output is close to zero-mean per
    # feature, so adding a constant to every weight entry nearly cancels out
    # (measured ~1e-6, under tolerance) -- scaling does not have that problem.
    reloaded = NanoPLMForMaskedLM.from_pretrained(out)
    with torch.no_grad():
        reloaded.head.dense.weight.mul_(2.0)
    reloaded.save_pretrained(out, safe_serialization=True)
    with pytest.raises(ExportVerificationError, match="logits"):
        verify_exported(
            native,
            out,
            tokenizer=_native_tokenizer(),
            device=torch.device("cpu"),
            fp8_source=False,
        )


def test_fp8_source_uses_a_looser_tolerance(tmp_path):
    native, out = _export_tiny(tmp_path)
    strict = verify_exported(
        native,
        out,
        tokenizer=_native_tokenizer(),
        device=torch.device("cpu"),
        fp8_source=False,
    )
    loose = verify_exported(
        native,
        out,
        tokenizer=_native_tokenizer(),
        device=torch.device("cpu"),
        fp8_source=True,
    )
    assert loose.tolerance > strict.tolerance


# --- (a) CUDA: proves R12-2 (native.model(...) avoids the fp32 varlen crash) ---


@pytest.mark.skipif(not WITH_CUDA, reason="needs CUDA")
def test_matching_export_passes_on_cuda(tmp_path):
    native, out = _export_tiny(tmp_path)
    result = verify_exported(
        native,
        out,
        tokenizer=_native_tokenizer(),
        device=torch.device("cuda"),
        fp8_source=False,
    )
    assert result.passed


# --- (b) recycling: proves R12-3's seeding --------------------------------------


def test_recycling_export_passes(tmp_path):
    native = _export(RECYCLING, tmp_path)
    result = verify_exported(
        native,
        tmp_path,
        tokenizer=_native_tokenizer(),
        device=torch.device("cpu"),
        fp8_source=False,
    )
    assert result.passed


# --- (c) MoE ---------------------------------------------------------------------


def test_moe_export_passes(tmp_path):
    native = _export(MOE, tmp_path)
    result = verify_exported(
        native,
        tmp_path,
        tokenizer=_native_tokenizer(),
        device=torch.device("cpu"),
        fp8_source=False,
    )
    assert result.passed


# --- (d) FP8 source on CUDA: native stays in TE layout after verify ------------


@pytest.mark.skipif(
    not (WITH_CUDA and WITH_TE), reason="needs CUDA + Transformer Engine"
)
def test_fp8_source_on_cuda_passes_and_leaves_native_in_te_layout(tmp_path):
    device = torch.device("cuda")
    torch.manual_seed(0)
    native = PureProtModernBertMLM(ProtModernBertMLMConfig(**TINY)).to(device).eval()
    n = convert_fp8_targets(native, target="mlp,attn_wqkv,attn_wo")
    assert n > 0

    # Export is built from a restored DEEP COPY -- native itself must stay
    # in TE layout, which the assertion below checks.
    restored = copy.deepcopy(native)
    restore_dense_from_te(restored, restored.config)

    hf = NanoPLMForMaskedLM(NanoPLMConfig(**TINY)).eval()
    hf.load_state_dict(restored.state_dict(), strict=True)
    hf.config.auto_map = AUTO_MAP
    hf.save_pretrained(tmp_path, safe_serialization=True)
    ProtModernBertTokenizer(num_register_tokens=0).save_pretrained(tmp_path)
    vendor_modeling_code(tmp_path)
    assert_no_unguarded_optional_imports(tmp_path)

    result = verify_exported(
        native,
        tmp_path,
        tokenizer=_native_tokenizer(),
        device=device,
        fp8_source=True,
    )
    assert result.passed
    # layers[0] has no MLP (no_mlp_on_first_layer); layers[-1] does.
    assert isinstance(native.model.layers[-1].mlp.layernorm_mlp, te.LayerNormMLP)


# --- (e) tokenizer mismatch --------------------------------------------------------


def test_tokenizer_mismatch_is_rejected(tmp_path):
    native, out = _export_tiny(tmp_path)
    with pytest.raises(ExportVerificationError, match="tokenizer"):
        verify_exported(
            native,
            out,
            # Export was saved with num_register_tokens=0; this tokenizer
            # appends a register token, so its ids/mask disagree.
            tokenizer=ProtModernBertTokenizer(num_register_tokens=1),
            device=torch.device("cpu"),
            fp8_source=False,
        )


# --- (f) the error names which tensor diverged --------------------------------


def test_perturbing_the_head_names_logits_only(tmp_path):
    native, out = _export_tiny(tmp_path)
    reloaded = NanoPLMForMaskedLM.from_pretrained(out)
    with torch.no_grad():
        reloaded.head.dense.weight.mul_(2.0)
    reloaded.save_pretrained(out, safe_serialization=True)
    with pytest.raises(ExportVerificationError) as excinfo:
        verify_exported(
            native,
            out,
            tokenizer=_native_tokenizer(),
            device=torch.device("cpu"),
            fp8_source=False,
        )
    assert "logits" in str(excinfo.value)
    assert "last_hidden_state" not in str(excinfo.value)


def test_perturbing_an_encoder_layer_names_last_hidden_state(tmp_path):
    native, out = _export_tiny(tmp_path)
    reloaded = NanoPLMForMaskedLM.from_pretrained(out)
    with torch.no_grad():
        # attn.Wo is zero-initialized (wo_init="zero", a zero-init-residual
        # scheme), so scaling it is a no-op and adding a constant to every
        # entry nearly cancels for the same reason as head.dense above
        # (measured ~1.5e-6, under tolerance). A single large entry sidesteps
        # both: it cannot be zeroed by scaling and cannot cancel in a sum.
        reloaded.model.layers[0].attn.Wo.weight[0, 0] += 10.0
    reloaded.save_pretrained(out, safe_serialization=True)
    with pytest.raises(ExportVerificationError, match="last_hidden_state"):
        verify_exported(
            native,
            out,
            tokenizer=_native_tokenizer(),
            device=torch.device("cpu"),
            fp8_source=False,
        )


# --- (g) the caller's RNG state and TF32 flags are untouched -------------------


def _rng_snapshot():
    snapshot = {
        "cpu": torch.random.get_rng_state(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
    if WITH_CUDA:
        snapshot["cuda"] = torch.cuda.get_rng_state_all()
    return snapshot


def _assert_rng_snapshots_equal(before, after):
    assert torch.equal(before["cpu"], after["cpu"])
    assert before["matmul_tf32"] == after["matmul_tf32"]
    assert before["cudnn_tf32"] == after["cudnn_tf32"]
    if WITH_CUDA:
        assert all(torch.equal(a, b) for a, b in zip(before["cuda"], after["cuda"]))


def test_rng_state_and_tf32_flags_are_restored_after_success(tmp_path):
    native, out = _export_tiny(tmp_path)
    torch.manual_seed(1)  # give the caller's RNG something to disturb-and-check
    before = _rng_snapshot()
    verify_exported(
        native,
        out,
        tokenizer=_native_tokenizer(),
        device=torch.device("cpu"),
        fp8_source=False,
    )
    _assert_rng_snapshots_equal(before, _rng_snapshot())


def test_rng_state_and_tf32_flags_are_restored_after_failure(tmp_path):
    native, out = _export_tiny(tmp_path)
    reloaded = NanoPLMForMaskedLM.from_pretrained(out)
    with torch.no_grad():
        reloaded.head.dense.weight.mul_(2.0)
    reloaded.save_pretrained(out, safe_serialization=True)

    torch.manual_seed(2)
    before = _rng_snapshot()
    with pytest.raises(ExportVerificationError):
        verify_exported(
            native,
            out,
            tokenizer=_native_tokenizer(),
            device=torch.device("cpu"),
            fp8_source=False,
        )
    _assert_rng_snapshots_equal(before, _rng_snapshot())


# --- (h) a bundle without auto_map is rejected ---------------------------------


def test_export_without_auto_map_is_rejected(tmp_path):
    native = PureProtModernBertMLM(ProtModernBertMLMConfig(**TINY)).eval()
    hf = NanoPLMForMaskedLM(NanoPLMConfig(**TINY)).eval()
    hf.load_state_dict(native.state_dict(), strict=True)
    hf.save_pretrained(tmp_path, safe_serialization=True)  # no auto_map set
    ProtModernBertTokenizer(num_register_tokens=0).save_pretrained(tmp_path)
    with pytest.raises(ExportVerificationError, match="auto_map"):
        verify_exported(
            native,
            tmp_path,
            tokenizer=_native_tokenizer(),
            device=torch.device("cpu"),
            fp8_source=False,
        )
