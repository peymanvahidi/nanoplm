"""The FP8 -> dense restore. Requires Transformer Engine, so GPU-gated.

Forward passes run the model in bf16: the pure-torch attention path always
takes the flash-attention varlen branch when ``attention_mask`` is given on a
CUDA tensor (regardless of dtype), and flash-attention only accepts
fp16/bf16/fp8 -- fp32 raises. Weight comparisons stay exact (``torch.equal``):
copy_-in/copy_-out through the transforms in fp8.py never rounds, whatever
the dtype.
"""

import copy
import re

import pytest
import torch

te = pytest.importorskip("transformer_engine.pytorch", reason="needs TE")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="TE module layout is built on GPU"
)

from nanoplm.pretraining.fp8 import convert_fp8_targets, restore_dense_from_te
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


def _build(**kwargs):
    cfg = ProtModernBertMLMConfig(**kwargs)
    return PureProtModernBertMLM(cfg).eval().to("cuda", torch.bfloat16)


def _ids_and_mask():
    # Built lazily (not at module scope): a CUDA tensor at import time would
    # raise "No CUDA GPUs are available" before the `pytestmark` skipif above
    # ever gets a chance to run, aborting collection instead of skipping.
    ids = torch.tensor([[4, 5, 6, 7, 1, 0]], device="cuda")
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]], device="cuda")
    return ids, mask


def _assert_round_trips(dense, target: str) -> int:
    """convert -> restore and check: shape parity, exact weights, close logits."""
    ids, mask = _ids_and_mask()
    reference = {k: v.clone() for k, v in dense.state_dict().items()}
    with torch.no_grad():
        want = dense(input_ids=ids, attention_mask=mask).logits.float()

    n = convert_fp8_targets(dense, target=target)
    assert n > 0
    restored = restore_dense_from_te(dense, dense.config)
    assert restored == n

    got_state = dense.state_dict()
    assert set(got_state) == set(reference), "restore changed the module tree shape"
    for key, expected in reference.items():
        assert torch.equal(got_state[key], expected), f"weight mismatch at {key!r}"
    assert not any(
        m.training for m in dense.modules()
    ), "restore left a rebuilt module in training mode"

    with torch.no_grad():
        got = dense(input_ids=ids, attention_mask=mask).logits.float()
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)
    return n


@pytest.mark.parametrize(
    "target", ["mlp", "attn_wo", "attn_wqkv", "mlp,attn_wqkv,attn_wo"]
)
@pytest.mark.parametrize("activation", ["swiglu", "geglu", "srelu"])
def test_restore_recovers_the_original_weights(target, activation):
    dense = _build(mlp_activation=activation, **TINY)
    _assert_round_trips(dense, target)


def test_restore_handles_unfused_qkv():
    """fused_qkv=False splits attention into Wq/Wk/Wv (3 linears). The generic
    ``restored == n`` check in ``_assert_round_trips`` is what actually proves
    per-layer counting here: if the restore counted per *linear* instead of
    per *layer*, ``restored`` would come out higher than ``n`` and fail."""
    dense = _build(mlp_activation="swiglu", fused_qkv=False, **TINY)
    _assert_round_trips(dense, "mlp,attn_wqkv,attn_wo")


def test_restore_handles_biases():
    """mlp_bias/attention_bias=True exercise the fc1/fc2/Wo/Q/K/V bias-copy
    branches that the default bias=False configs above never touch."""
    dense = _build(
        mlp_activation="swiglu",
        mlp_bias=True,
        attention_bias=True,
        **TINY,
    )
    _assert_round_trips(dense, "mlp,attn_wqkv,attn_wo")


def test_restore_handles_rmsnorm():
    dense = _build(mlp_activation="swiglu", norm_type="rmsnorm", **TINY)
    _assert_round_trips(dense, "mlp,attn_wqkv,attn_wo")


def test_restore_rebuilds_moe_dense_layer_at_moe_dense_intermediate_size():
    """A leading dense layer inside an MoE stack must come back at
    ``moe_dense_intermediate_size``, not the plain ``intermediate_size``.

    Layer 0 is dense (moe_leading_dense_layers=2) but has no MLP at all
    (no_mlp_on_first_layer defaults True); layer 1 is the dense layer that
    actually carries one; layer 2 is a real MoE layer and is never converted.
    """
    dense = _build(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_kv_heads=2,
        mlp_activation="swiglu",
        use_moe=True,
        moe_num_experts=4,
        moe_top_k=2,
        moe_leading_dense_layers=2,
    )
    dense_layer = dense.model.layers[1]
    assert (
        dense_layer.mlp.Wi.weight.shape[0]
        == 2 * dense.config.moe_dense_intermediate_size
    )
    assert dense.model.layers[0].mlp is None
    from nanoplm.pretraining.models.modern_bert.moe import MoELayer

    assert isinstance(dense.model.layers[2].mlp, MoELayer)

    reference_wi = dense_layer.mlp.Wi.weight.detach().clone()
    reference_wo = dense_layer.mlp.Wo.weight.detach().clone()

    n = convert_fp8_targets(dense, target="mlp")
    assert n == 1
    restored = restore_dense_from_te(dense, dense.config)
    assert restored == 1

    dense_layer = dense.model.layers[1]
    assert (
        dense_layer.mlp.Wi.weight.shape[0]
        == 2 * dense.config.moe_dense_intermediate_size
    )
    assert torch.equal(dense_layer.mlp.Wi.weight, reference_wi)
    assert torch.equal(dense_layer.mlp.Wo.weight, reference_wo)
    assert isinstance(dense.model.layers[2].mlp, MoELayer)
    assert not any(
        m.training for m in dense.modules()
    ), "restore left a rebuilt module in training mode"


def test_restore_rejects_a_config_whose_activation_disagrees_with_the_checkpoint():
    dense = _build(mlp_activation="swiglu", **TINY)
    convert_fp8_targets(dense, target="mlp")

    mismatched_cfg = ProtModernBertMLMConfig(mlp_activation="geglu", **TINY)
    from nanoplm.pretraining.models.modern_bert.pure_model import modernbert_config_from

    with pytest.raises(ValueError, match="swiglu"):
        restore_dense_from_te(dense, modernbert_config_from(mismatched_cfg))


def test_restore_is_a_noop_on_a_dense_model():
    dense = _build(**TINY)
    assert restore_dense_from_te(dense, dense.config) == 0


# --- a trained TE LayerNorm bias has no dense slot: refuse ---------------------
#
# With norm_type="layernorm" the native norm is bias-free (norm_bias=False, not
# user-settable), but TE's LayerNormMLP / LayerNormLinear still own a trainable
# `layer_norm_bias`, and FP8 training moves it off zero. The restore used to copy
# it only when the native norm had a bias, silently dropping the trained values.
# It must refuse instead -- and still accept an exactly-zero bias (converted but
# never trained). Three layers, so layer 1 is converted and clean while the
# offender sits in layer 2: the refusal must come before layer 1 is restored.

LAYERNORM = dict(TINY, num_hidden_layers=3, norm_type="layernorm")
ALL_TARGETS = "mlp,attn_wqkv,attn_wo"


def _te_layernorm_module(wrapper):
    if hasattr(wrapper, "layernorm_mlp"):
        return wrapper.layernorm_mlp
    return wrapper.layernorm_linear


def _te_norm_biases(model) -> dict:
    return {
        name: param
        for name, param in model.named_parameters()
        if name.endswith("layer_norm_bias")
    }


@pytest.mark.parametrize(
    ("fused_qkv", "module_path"),
    [
        (True, "model.layers.2.mlp"),
        (True, "model.layers.2.attn.Wqkv"),
        (False, "model.layers.2.mlp"),
        (False, "model.layers.2.attn.Wq"),
        (False, "model.layers.2.attn.Wk"),
        (False, "model.layers.2.attn.Wv"),
    ],
)
def test_restore_refuses_a_nonzero_te_layernorm_bias(fused_qkv, module_path):
    dense = _build(mlp_activation="swiglu", fused_qkv=fused_qkv, **LAYERNORM)
    convert_fp8_targets(dense, target=ALL_TARGETS)
    te_module = _te_layernorm_module(dense.get_submodule(module_path))
    with torch.no_grad():
        te_module.layer_norm_bias[5] = -0.25
    before = {k: v.clone() for k, v in dense.state_dict().items()}

    with pytest.raises(ValueError) as excinfo:
        restore_dense_from_te(dense, dense.config)

    message = str(excinfo.value)
    assert module_path in message
    assert "0.25" in message  # the max |bias|
    assert "FP8" in message and "LayerNorm" in message
    # Nothing was restored, not even the clean layer 1: the refusal comes first.
    after = dense.state_dict()
    assert set(after) == set(before)
    for key, expected in before.items():
        assert torch.equal(after[key], expected), key


def test_restore_refusal_lists_every_offending_block():
    dense = _build(mlp_activation="swiglu", fused_qkv=False, **LAYERNORM)
    convert_fp8_targets(dense, target=ALL_TARGETS)
    with torch.no_grad():
        for path in ("model.layers.1.attn.Wk", "model.layers.2.mlp"):
            _te_layernorm_module(dense.get_submodule(path)).layer_norm_bias[0] = 0.5
    with pytest.raises(ValueError) as excinfo:
        restore_dense_from_te(dense, dense.config)
    named = set(re.findall(r"model\.layers\.\d+(?:\.\w+)+", str(excinfo.value)))
    # Exactly the two offenders; the clean Wq/Wv beside Wk are not named.
    assert named == {"model.layers.1.attn.Wk", "model.layers.2.mlp"}


@pytest.mark.parametrize("fused_qkv", [True, False], ids=["fused", "unfused"])
def test_restore_accepts_an_exactly_zero_te_layernorm_bias(fused_qkv):
    dense = _build(mlp_activation="swiglu", fused_qkv=fused_qkv, **LAYERNORM)

    # A converted-but-never-trained model: TE creates layer_norm_bias (zeros)
    # for every LayerNorm module, even though the native norm has no bias.
    converted = copy.deepcopy(dense)
    convert_fp8_targets(converted, target=ALL_TARGETS)
    biases = _te_norm_biases(converted)
    # 2 converted layers x (MLP + fused Wqkv, or MLP + Wq/Wk/Wv).
    assert len(biases) == (4 if fused_qkv else 8), sorted(biases)
    assert all(torch.count_nonzero(b) == 0 for b in biases.values())

    _assert_round_trips(dense, ALL_TARGETS)
