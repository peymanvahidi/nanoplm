import torch
import pytest
from torch import nn
import transformer_engine.pytorch as te

from nanoplm.pretraining.fp8 import (
    convert_attention_qkv_to_te_layernorm_linear,
    convert_attention_wo_to_te_linear,
    convert_fp8_targets,
    convert_mlp_blocks_to_te_layernorm_mlp,
    te_fp8_recipe_from_config,
)
from nanoplm.pretraining.config import PretrainingConfig


class _Cfg:
    fp8_recipe = "delayed"
    fp8_format = "hybrid"
    fp8_amax_history_len = 16
    fp8_reduce_amax = False


class _EncoderLayerWrapper(nn.Module):
    def __init__(self, activation: str = "swiglu"):
        super().__init__()
        self.layer = nn.Module()
        self.layer.canon_c = None
        self.layer.mlp_norm = nn.RMSNorm(16, eps=1e-5)
        self.layer.mlp = nn.Module()
        self.layer.mlp.activation = activation
        self.layer.mlp.fused_up_gate = True
        self.layer.mlp.Wi = nn.Linear(16, 64, bias=False)
        self.layer.mlp.Wo = nn.Linear(32, 16, bias=False)
        self.layer.mlp.drop = nn.Dropout(0.0)
        self.layer.mlp.canon_d = None


class _SReluEncoderLayerWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Module()
        self.layer.canon_c = None
        self.layer.mlp_norm = nn.RMSNorm(16, eps=1e-5)
        self.layer.mlp = nn.Module()
        self.layer.mlp.Wi = nn.Linear(16, 32, bias=False)
        self.layer.mlp.Wo_weight = nn.Parameter(torch.empty(32, 16))
        self.layer.mlp.Wo_bias = None
        self.layer.mlp.drop = nn.Dropout(0.0)


class _FusedAttentionLayerWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Module()
        self.layer.canon_a = None
        self.layer.attn_norm = nn.RMSNorm(16, eps=1e-5)
        self.layer.attn = nn.Module()
        self.layer.attn.fused_qkv = True
        self.layer.attn.Wqkv = nn.Linear(16, 48, bias=False)
        self.layer.attn.Wq = None
        self.layer.attn.Wq2 = None
        self.layer.attn.Wk = None
        self.layer.attn.Wv = None


class _UnfusedAttentionLayerWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Module()
        self.layer.canon_a = None
        self.layer.attn_norm = nn.RMSNorm(16, eps=1e-5)
        self.layer.attn = nn.Module()
        self.layer.attn.fused_qkv = False
        self.layer.attn.Wqkv = None
        self.layer.attn.Wq = nn.Linear(16, 16, bias=False)
        self.layer.attn.Wq2 = None
        self.layer.attn.Wk = nn.Linear(16, 16, bias=False)
        self.layer.attn.Wv = nn.Linear(16, 16, bias=False)


class _AttentionMlpLayerWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Module()
        self.layer.canon_a = None
        self.layer.attn_norm = nn.RMSNorm(16, eps=1e-5)
        self.layer.attn = nn.Module()
        self.layer.attn.fused_qkv = True
        self.layer.attn.Wqkv = nn.Linear(16, 48, bias=False)
        self.layer.attn.Wq = None
        self.layer.attn.Wq2 = None
        self.layer.attn.Wk = None
        self.layer.attn.Wv = None
        self.layer.attn.Wo = nn.Linear(16, 16, bias=False)
        self.layer.canon_c = None
        self.layer.mlp_norm = nn.RMSNorm(16, eps=1e-5)
        self.layer.mlp = nn.Module()
        self.layer.mlp.Wi = nn.Linear(16, 32, bias=False)
        self.layer.mlp.Wo_weight = nn.Parameter(torch.empty(32, 16))
        self.layer.mlp.Wo_bias = None
        self.layer.mlp.drop = nn.Dropout(0.0)
        self.layer.mlp.canon_d = None


class _LayerStack(nn.Module):
    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_AttentionMlpLayerWrapper().layer for _ in range(n_layers)]
        )


def test_convert_mlp_blocks_to_te_layernorm_mlp_copies_norm_and_mlp_weights():
    model = _EncoderLayerWrapper()
    old_norm_weight = model.layer.mlp_norm.weight.detach().clone()
    old_wi_weight = model.layer.mlp.Wi.weight.detach().clone()
    old_wo_weight = model.layer.mlp.Wo.weight.detach().clone()

    converted = convert_mlp_blocks_to_te_layernorm_mlp(model)

    assert converted == 1
    assert isinstance(model.layer.mlp_norm, nn.Identity)
    assert isinstance(model.layer.mlp.layernorm_mlp, te.LayerNormMLP)
    fused = model.layer.mlp.layernorm_mlp
    old_up_weight, old_gate_weight = old_wi_weight.chunk(2, dim=0)
    assert torch.allclose(fused.layer_norm_weight.detach().cpu(), old_norm_weight)
    assert torch.allclose(
        fused.fc1_weight.detach().cpu(),
        torch.cat([old_gate_weight, old_up_weight], dim=0),
    )
    assert torch.allclose(fused.fc2_weight.detach().cpu(), old_wo_weight)


def test_convert_mlp_blocks_to_te_layernorm_mlp_accepts_srelu():
    model = _SReluEncoderLayerWrapper()
    nn.init.normal_(model.layer.mlp.Wo_weight)
    old_norm_weight = model.layer.mlp_norm.weight.detach().clone()
    old_wi_weight = model.layer.mlp.Wi.weight.detach().clone()
    old_wo_weight = model.layer.mlp.Wo_weight.detach().clone()

    converted = convert_mlp_blocks_to_te_layernorm_mlp(model)

    assert converted == 1
    assert isinstance(model.layer.mlp_norm, nn.Identity)
    assert isinstance(model.layer.mlp.layernorm_mlp, te.LayerNormMLP)
    fused = model.layer.mlp.layernorm_mlp
    assert fused.activation == "srelu"
    assert torch.allclose(fused.layer_norm_weight.detach().cpu(), old_norm_weight)
    assert torch.allclose(fused.fc1_weight.detach().cpu(), old_wi_weight)
    assert torch.allclose(fused.fc2_weight.detach().cpu(), old_wo_weight.T)


def test_convert_attention_wo_to_te_linear_swaps_only_wo():
    model = _AttentionMlpLayerWrapper()
    old_norm = model.layer.attn_norm
    old_wo_weight = model.layer.attn.Wo.weight.detach().clone()
    old_wqkv = model.layer.attn.Wqkv

    converted = convert_attention_wo_to_te_linear(model)

    assert converted == 1
    assert isinstance(model.layer.attn.Wo, te.Linear)
    assert torch.allclose(model.layer.attn.Wo.weight.detach().cpu(), old_wo_weight)
    # no norm involved and qkv untouched
    assert model.layer.attn_norm is old_norm
    assert model.layer.attn.Wqkv is old_wqkv


def test_convert_attention_qkv_to_te_layernorm_linear_handles_fused_qkv():
    model = _FusedAttentionLayerWrapper()
    old_norm = model.layer.attn_norm
    old_qkv_weight = model.layer.attn.Wqkv.weight.detach().clone()

    converted = convert_attention_qkv_to_te_layernorm_linear(model)

    assert converted == 1
    assert isinstance(model.layer.attn_norm, nn.Identity)
    assert isinstance(model.layer.attn.Wqkv.layernorm_linear, te.LayerNormLinear)
    fused_qkv = model.layer.attn.Wqkv.layernorm_linear
    assert fused_qkv.layer_norm_weight is old_norm.weight
    assert torch.allclose(fused_qkv.weight.detach().cpu(), old_qkv_weight)


def test_convert_attention_qkv_to_te_layernorm_linear_handles_unfused_qkv():
    model = _UnfusedAttentionLayerWrapper()
    old_norm = model.layer.attn_norm
    old_wq = model.layer.attn.Wq.weight.detach().clone()
    old_wk = model.layer.attn.Wk.weight.detach().clone()
    old_wv = model.layer.attn.Wv.weight.detach().clone()

    converted = convert_fp8_targets(model, target="attn_wqkv")

    assert converted == 1
    assert isinstance(model.layer.attn_norm, nn.Identity)
    for name, old_weight in (("Wq", old_wq), ("Wk", old_wk), ("Wv", old_wv)):
        wrapped = getattr(model.layer.attn, name)
        assert isinstance(wrapped.layernorm_linear, te.LayerNormLinear)
        assert wrapped.layernorm_linear.layer_norm_weight is old_norm.weight
        assert torch.allclose(wrapped.layernorm_linear.weight.detach().cpu(), old_weight)
    assert model.layer.attn.Wqkv is None
    assert model.layer.attn.fused_qkv is False


def test_convert_fp8_targets_accepts_all_three_targets_combined():
    model = _AttentionMlpLayerWrapper()
    nn.init.normal_(model.layer.mlp.Wo_weight)

    converted = convert_fp8_targets(model, target="attn_wqkv,attn_wo,mlp")

    assert converted == 3
    assert isinstance(model.layer.attn_norm, nn.Identity)
    assert isinstance(model.layer.mlp_norm, nn.Identity)
    assert isinstance(model.layer.attn.Wqkv.layernorm_linear, te.LayerNormLinear)
    assert isinstance(model.layer.attn.Wo, te.Linear)
    assert isinstance(model.layer.mlp.layernorm_mlp, te.LayerNormMLP)


def test_convert_fp8_targets_rejects_removed_mlp_wi_target():
    model = _AttentionMlpLayerWrapper()

    with pytest.raises(ValueError, match="FP8 target"):
        convert_fp8_targets(model, target=["attn_wqkv", "mlp_wi"])


def test_convert_fp8_targets_can_skip_last_transformer_layers():
    model = _LayerStack(n_layers=4)
    for layer in model.model.layers:
        nn.init.normal_(layer.mlp.Wo_weight)

    converted = convert_fp8_targets(
        model,
        target="attn_wqkv,attn_wo",
        skip_last_n_layers=2,
    )

    assert converted == 4
    for layer in model.model.layers[:2]:
        assert isinstance(layer.attn_norm, nn.Identity)
        assert isinstance(layer.attn.Wqkv.layernorm_linear, te.LayerNormLinear)
        assert isinstance(layer.attn.Wo, te.Linear)
    for layer in model.model.layers[2:]:
        assert isinstance(layer.attn_norm, nn.RMSNorm)
        assert isinstance(layer.attn.Wqkv, nn.Linear)
        assert not isinstance(layer.attn.Wo, te.Linear)


def test_pretraining_config_accepts_fp8_skip_last_n_layers():
    cfg = PretrainingConfig(
        dataset_dir="/tmp/nanoplm-dummy",
        fp8_target=["attn_wqkv", "attn_wo"],
        fp8_skip_last_n_layers="3",
    )

    assert cfg.fp8_target == ["attn_wqkv", "attn_wo"]
    assert cfg.fp8_skip_last_n_layers == 3


def test_te_fp8_recipe_from_config_accepts_current_config_shape():
    recipe = te_fp8_recipe_from_config(_Cfg())

    assert recipe is not None


# ---------------------------------------------------------------------------
# geglu routing
#
# ModernBertMLP (geglu) and ModernBertSwiGLUMLP are structurally identical, so
# the conversion used to classify both as "swiglu" purely from shapes and
# silently retrained gelu-gated layers as silu-gated ones under fp8.
# ---------------------------------------------------------------------------

def _geglu_layer(hidden=16, inter=32, bias=False):
    from nanoplm.pretraining.models.modern_bert.config import ModernBertConfig
    from nanoplm.pretraining.models.modern_bert.modeling import ModernBertMLP

    cfg = ModernBertConfig(
        hidden_size=hidden, intermediate_size=inter, num_hidden_layers=2,
        num_attention_heads=4, num_kv_heads=2, vocab_size=32,
        mlp_activation="geglu", mlp_bias=bias, norm_type="rmsnorm",
        fused_up_gate=True,
    )
    layer = nn.Module()
    layer.canon_c = None
    layer.mlp_norm = nn.RMSNorm(hidden, eps=1e-5)
    layer.mlp = ModernBertMLP(cfg)
    model = nn.Module()
    model.layer = layer
    return model


def test_geglu_is_not_classified_as_swiglu():
    from nanoplm.pretraining.fp8 import _te_layernorm_mlp_activation

    assert _te_layernorm_mlp_activation(_geglu_layer().layer) == "geglu"
    assert _te_layernorm_mlp_activation(_EncoderLayerWrapper().layer) == "swiglu"


def test_convert_mlp_blocks_routes_geglu_to_te_geglu_without_swapping_halves():
    model = _geglu_layer()
    old_wi_weight = model.layer.mlp.Wi.weight.detach().clone()
    old_wo_weight = model.layer.mlp.Wo.weight.detach().clone()

    converted = convert_mlp_blocks_to_te_layernorm_mlp(model)

    assert converted == 1
    fused = model.layer.mlp.layernorm_mlp
    assert fused.activation == "geglu"
    # TE geglu gates on the FIRST half, exactly like ModernBertMLP, so the
    # halves must carry over in order -- unlike swiglu, which swaps them.
    assert torch.allclose(fused.fc1_weight.detach().cpu(), old_wi_weight)
    assert torch.allclose(fused.fc2_weight.detach().cpu(), old_wo_weight)


def test_unrecognised_gate_activation_is_left_in_eager():
    from nanoplm.pretraining.fp8 import _te_layernorm_mlp_activation

    model = _geglu_layer()
    model.layer.mlp.act = torch.tanh  # not a gate TE can be trusted to match
    assert _te_layernorm_mlp_activation(model.layer) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE LayerNormMLP needs CUDA")
def test_te_geglu_conversion_matches_eager_math():
    torch.manual_seed(0)
    model = _geglu_layer(hidden=64, inter=128).cuda()
    x = torch.randn(2, 8, 64, device="cuda")

    with torch.no_grad():
        expected = model.layer.mlp(model.layer.mlp_norm(x))

    convert_mlp_blocks_to_te_layernorm_mlp(model)
    with torch.no_grad():
        got = model.layer.mlp(x)

    assert torch.allclose(got, expected, atol=2e-2, rtol=2e-2), (
        f"max abs diff {(got - expected).abs().max().item():.3e}"
    )
