"""The HF shells must initialize parameters with nanoPLM's native scheme.

Before this fix, `NanoPLMPreTrainedModel._init_weights` only rematerialized
non-persistent buffers and never wrote a Parameter: a directly constructed
`NanoPLMModel` / `NanoPLMForMaskedLM` got plain PyTorch defaults for
`nn.Linear` / `nn.Embedding`, and uninitialized `torch.empty` memory for
`MoELayer.Wi` / `MoELayer.Wo` (raw Parameters, not `nn.Linear`s). The same
held for any key a checkpoint lacked on `from_pretrained`.
"""

from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file

from nanoplm.hf import NanoPLMConfig, NanoPLMForMaskedLM, NanoPLMModel
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM

TINY = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_kv_heads=2,
    vocab_size=32,
)

MOE = dict(
    TINY,
    use_moe=True,
    moe_num_experts=4,
    moe_top_k=2,
    moe_kernel_backend="cutlass",
)

MHC_LITE = dict(TINY, use_mhc_lite=True)

# recycling_use_takase_init defaults to True, so this also exercises the
# Takase spike-init path (`_apply_takase_init`) at the tail of init.
RECYCLING = dict(
    TINY,
    num_hidden_layers=4,
    use_huginn_looping=True,
    recycling_loop_count=2,
    recycling_num_prelude_layers=1,
    recycling_num_coda_layers=1,
)

# (b) is intentionally narrower than (a): recycling's Takase init runs AFTER
# the decoder's (skipped, when tied) `normal_` call, so whether that call
# happens shifts the RNG stream reaching Takase -- an RNG-order difference
# outside this task's control, not a bug. default/MoE/mHC-lite have no such
# tail draw, so they must match exactly.
ALL_CONFIGS = {
    "default": TINY,
    "moe": MOE,
    "mhc_lite": MHC_LITE,
    "recycling": RECYCLING,
}
PARITY_CONFIGS = {"default": TINY, "moe": MOE, "mhc_lite": MHC_LITE}

_real_empty = torch.empty


def _nan_empty(*args, **kwargs):
    """`torch.empty`, but every freshly allocated floating tensor starts as
    NaN instead of whatever the allocator happened to hand back. Makes test
    (a) deterministic: the bug it targets depended on allocator luck.
    """
    t = _real_empty(*args, **kwargs)
    if t.is_floating_point():
        t.fill_(float("nan"))
    return t


def _hf_config(kwargs: dict) -> NanoPLMConfig:
    return NanoPLMConfig.from_model_config(ProtModernBertMLMConfig(**kwargs))


# --- (a) every parameter is written by init ---------------------------------


@pytest.mark.parametrize("cls", [NanoPLMModel, NanoPLMForMaskedLM])
@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_every_parameter_is_finite_after_construction(name, cls):
    cfg = _hf_config(ALL_CONFIGS[name])
    with patch.object(torch, "empty", side_effect=_nan_empty):
        model = cls(cfg)
    non_finite = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    assert non_finite == []


# --- (b) seeded parity with the native model --------------------------------


@pytest.mark.parametrize("name", PARITY_CONFIGS)
def test_seeded_parity_with_native(name):
    kwargs = PARITY_CONFIGS[name]
    model_cfg = ProtModernBertMLMConfig(**kwargs)
    hf_cfg = NanoPLMConfig.from_model_config(model_cfg)

    torch.manual_seed(0)
    native = PureProtModernBertMLM(model_cfg)

    torch.manual_seed(0)
    hf = NanoPLMForMaskedLM(hf_cfg)

    native_sd, hf_sd = native.state_dict(), hf.state_dict()
    assert set(native_sd) == set(hf_sd)
    for key in native_sd:
        assert torch.equal(native_sd[key], hf_sd[key]), key


# --- (c) loaded weights survive from_pretrained -----------------------------


# use_x0_lambdas rides along with use_resid_lambdas (structurally identical
# guard, same test). use_mhc_lite is a separate config -- the two are
# mutually exclusive (config.py's ModernBertConfig.__post_init__ rejects
# use_mhc_lite with use_resid_lambdas) -- so this also exercises the mHC
# W_all bias slices and alpha_pre/post/res guards, not just the resid/x0
# lambdas the brief names explicitly.
_ROUND_TRIP_GUARD_CONFIGS = {
    "resid_and_x0_lambdas": dict(TINY, use_resid_lambdas=True, use_x0_lambdas=True),
    "mhc_lite": dict(TINY, use_mhc_lite=True),
}


@pytest.mark.parametrize("name", _ROUND_TRIP_GUARD_CONFIGS)
def test_loaded_weights_survive_from_pretrained(tmp_path, name):
    cfg = _hf_config(_ROUND_TRIP_GUARD_CONFIGS[name])
    hf = NanoPLMForMaskedLM(cfg).eval()
    with torch.no_grad():
        for p in hf.parameters():
            p.add_(1.0)  # perturb every trainable parameter away from init
        if hf.model.resid_lambdas is not None:
            hf.model.resid_lambdas.fill_(0.37)  # recognisable non-init value
    saved = {k: v.clone() for k, v in hf.state_dict().items()}

    hf.save_pretrained(tmp_path, safe_serialization=True)
    reloaded = NanoPLMForMaskedLM.from_pretrained(tmp_path)

    reloaded_sd = reloaded.state_dict()
    assert set(reloaded_sd) == set(saved)
    for key, value in saved.items():
        assert torch.equal(reloaded_sd[key], value), key


# --- (d) missing keys get initialized, loaded keys stay ---------------------


def test_missing_key_initialized_loaded_keys_stay(tmp_path):
    cfg = _hf_config(MOE)
    hf = NanoPLMForMaskedLM(cfg).eval()
    hf.save_pretrained(tmp_path, safe_serialization=True)

    weights_path = tmp_path / "model.safetensors"
    state_dict = load_file(str(weights_path))
    missing_key = next(k for k in state_dict if k.endswith("mlp.Wi"))
    expected = {k: v.clone() for k, v in state_dict.items() if k != missing_key}
    del state_dict[missing_key]
    save_file(state_dict, str(weights_path))

    with patch.object(torch, "empty", side_effect=_nan_empty):
        reloaded = NanoPLMForMaskedLM.from_pretrained(tmp_path)

    reloaded_sd = reloaded.state_dict()
    assert torch.isfinite(reloaded_sd[missing_key]).all()
    for key, value in expected.items():
        assert torch.equal(reloaded_sd[key], value), key
