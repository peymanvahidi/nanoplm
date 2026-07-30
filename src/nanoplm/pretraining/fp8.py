"""Transformer Engine FP8 helpers for the pure-torch pipeline."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from nanoplm.pretraining.cpu_offload import run_target

_TE_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover - import availability depends on the runtime image
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import (
        DelayedScaling,
        Float8BlockScaling,
        Float8CurrentScaling,
        Format,
        MXFP8BlockScaling,
        NVFP4BlockScaling,
    )
except Exception as exc:  # pragma: no cover
    te = None  # type: ignore[assignment]
    DelayedScaling = None  # type: ignore[assignment]
    Float8BlockScaling = None  # type: ignore[assignment]
    Float8CurrentScaling = None  # type: ignore[assignment]
    Format = None  # type: ignore[assignment]
    MXFP8BlockScaling = None  # type: ignore[assignment]
    NVFP4BlockScaling = None  # type: ignore[assignment]
    _TE_IMPORT_ERROR = exc


def _require_te():
    if te is None:
        raise ImportError(
            "fp8=True in the pure-torch pipeline now requires Transformer Engine. "
            "Install transformer-engine or run without fp8."
        ) from _TE_IMPORT_ERROR
    return te


class _TELayerNormMLPWrapper(nn.Module):
    """Adapter with the ModernBERT MLP forward signature."""

    def __init__(self, layernorm_mlp: nn.Module):
        super().__init__()
        self.layernorm_mlp = layernorm_mlp
        # Set by cpu_offload.install() when activation offload is enabled.
        self._cpu_offload = None

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del position_ids, cu_seqlens, attention_mask
        return run_target(self._cpu_offload, self.layernorm_mlp, x)


class _TELayerNormLinearWrapper(nn.Module):
    """Adapter that can stand in for ``nn.Linear`` after a separate pre-norm."""

    def __init__(self, layernorm_linear: nn.Module):
        super().__init__()
        self.layernorm_linear = layernorm_linear
        # Set by cpu_offload.install() when activation offload is enabled.
        self._cpu_offload = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return run_target(self._cpu_offload, self.layernorm_linear, x)


def iter_te_cpu_offload_target_wrappers(module: nn.Module):
    """Yield TE wrapper modules eligible for CPU activation offload.

    Yielded in module-registration order, which for ModernBERT matches
    forward-execution order (attention QKV before MLP, layers in sequence) --
    the order ``cpu_offload.install`` relies on for its 1-step-ahead schedule.
    """
    for child in module.modules():
        if isinstance(child, (_TELayerNormLinearWrapper, _TELayerNormMLPWrapper)):
            yield child


def _normalization_name(norm: nn.Module) -> str:
    return "RMSNorm" if isinstance(norm, nn.RMSNorm) else "LayerNorm"


def _tie_norm_to_te_layernorm_module(te_module: nn.Module, norm: nn.Module) -> None:
    te_module.layer_norm_weight = norm.weight
    if isinstance(norm, nn.LayerNorm) and getattr(norm, "bias", None) is not None:
        te_module.layer_norm_bias = norm.bias


def _copy_linear_to_te_layernorm_linear(
    *,
    linear: nn.Linear,
    norm: nn.Module,
    name: str | None,
) -> nn.Module:
    te_mod = _require_te()
    normalization = _normalization_name(norm)
    new_linear = te_mod.LayerNormLinear(
        linear.in_features,
        linear.out_features,
        eps=float(norm.eps),
        bias=linear.bias is not None,
        normalization=normalization,
        params_dtype=linear.weight.dtype,
        device=linear.weight.device,
        name=name,
    )
    _tie_norm_to_te_layernorm_module(new_linear, norm)
    with torch.no_grad():
        new_linear.weight.copy_(linear.weight)
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias)
    new_linear.train(linear.training)
    return _TELayerNormLinearWrapper(new_linear)


def _skip_last_transformer_layers(module: nn.Module, skip_last_n_layers: int) -> set[nn.Module]:
    if skip_last_n_layers <= 0:
        return set()

    layer_lists: list[list[nn.Module]] = []
    for child in module.modules():
        layers = getattr(child, "layers", None)
        if not isinstance(layers, nn.ModuleList):
            continue
        entries: list[nn.Module] = []
        for layer in layers:
            enc = getattr(layer, "layer", layer)
            if hasattr(enc, "attn") or hasattr(enc, "mlp"):
                entries.append(layer)
        if entries:
            layer_lists.append(entries)

    if not layer_lists:
        return set()

    layers = max(layer_lists, key=len)
    skipped: set[nn.Module] = set()
    for layer in layers[-skip_last_n_layers:]:
        skipped.add(layer)
        skipped.add(getattr(layer, "layer", layer))
    return skipped


def _te_layernorm_mlp_activation(layer: nn.Module) -> str | None:
    mlp_norm = getattr(layer, "mlp_norm", None)
    mlp = getattr(layer, "mlp", None)
    if mlp is None or mlp_norm is None:
        return None
    if getattr(layer, "canon_c", None) is not None:
        return None
    if getattr(mlp, "canon_d", None) is not None:
        return None
    if not isinstance(mlp_norm, (nn.LayerNorm, nn.RMSNorm)):
        return None

    wi = getattr(mlp, "Wi", None)
    if not isinstance(wi, nn.Linear):
        return None

    drop = getattr(mlp, "drop", None)
    if isinstance(drop, nn.Dropout) and float(drop.p) != 0.0:
        return None

    if bool(getattr(mlp, "fused_up_gate", False)):
        wo = getattr(mlp, "Wo", None)
        if not isinstance(wo, nn.Linear):
            return None
        if wi.bias is not None and wo.bias is None:
            return None
        if wi.bias is None and wo.bias is not None:
            return None
        if wi.out_features != 2 * wo.in_features:
            return None
        # ModernBertMLP (geglu) and ModernBertSwiGLUMLP are shape-identical, so
        # the shapes above cannot tell them apart -- only ModernBertMLP carries
        # an explicit `.act`, while the SwiGLU module hardcodes silu. Getting
        # this wrong silently trains a different activation under fp8, so an
        # unrecognised gate falls back to eager rather than guessing.
        act = getattr(mlp, "act", None)
        if act is None:
            return "swiglu"
        if act is F.gelu:
            return "geglu"
        return None

    wo_weight = getattr(mlp, "Wo_weight", None)
    wo_bias = getattr(mlp, "Wo_bias", None)
    if isinstance(wo_weight, nn.Parameter) and wo_weight.ndim == 2:
        if wi.bias is not None and wo_bias is None:
            return None
        if wi.bias is None and wo_bias is not None:
            return None
        if wi.out_features == wo_weight.shape[0] and wi.in_features == wo_weight.shape[1]:
            return "srelu"
    return None


def _is_te_layernorm_mlp_candidate(layer: nn.Module) -> bool:
    return _te_layernorm_mlp_activation(layer) is not None


def _copy_mlp_branch_to_te_layernorm_mlp(layer: nn.Module, *, name: str | None = None) -> nn.Module:
    te_mod = _require_te()
    mlp_norm = layer.mlp_norm
    mlp = layer.mlp
    wi = mlp.Wi
    activation = _te_layernorm_mlp_activation(layer)
    if activation is None:
        raise ValueError("Layer is not eligible for Transformer Engine LayerNormMLP conversion.")
    gated_activation = activation in {"swiglu", "geglu"}
    swap_gate_halves = activation == "swiglu"
    ffn_hidden_size = mlp.Wo.in_features if gated_activation else mlp.Wo_weight.shape[0]
    normalization = "RMSNorm" if isinstance(mlp_norm, nn.RMSNorm) else "LayerNorm"
    new_mlp = te_mod.LayerNormMLP(
        wi.in_features,
        ffn_hidden_size,
        eps=float(mlp_norm.eps),
        bias=wi.bias is not None,
        normalization=normalization,
        activation=activation,
        params_dtype=wi.weight.dtype,
        device=wi.weight.device,
        name=name,
    )
    with torch.no_grad():
        new_mlp.layer_norm_weight.copy_(mlp_norm.weight)
        if normalization == "LayerNorm" and getattr(mlp_norm, "bias", None) is not None:
            new_mlp.layer_norm_bias.copy_(mlp_norm.bias)
        if gated_activation:
            # TE applies the nonlinearity to the FIRST half of fc1 for both
            # gated forms. ModernBertMLP (geglu) already does that, so its
            # halves carry over as-is; ModernBertSwiGLUMLP applies silu to the
            # SECOND half, so its halves have to swap on the way in.
            if swap_gate_halves:
                up_weight, gate_weight = wi.weight.chunk(2, dim=0)
                new_mlp.fc1_weight.copy_(torch.cat([gate_weight, up_weight], dim=0))
            else:
                new_mlp.fc1_weight.copy_(wi.weight)
            new_mlp.fc2_weight.copy_(mlp.Wo.weight)
        else:
            new_mlp.fc1_weight.copy_(wi.weight)
            new_mlp.fc2_weight.copy_(mlp.Wo_weight.T)
        if wi.bias is not None:
            if gated_activation:
                if swap_gate_halves:
                    up_bias, gate_bias = wi.bias.chunk(2, dim=0)
                    new_mlp.fc1_bias.copy_(torch.cat([gate_bias, up_bias], dim=0))
                else:
                    new_mlp.fc1_bias.copy_(wi.bias)
                new_mlp.fc2_bias.copy_(mlp.Wo.bias)
            else:
                new_mlp.fc1_bias.copy_(wi.bias)
                new_mlp.fc2_bias.copy_(mlp.Wo_bias)
    new_mlp.train(mlp.training)
    return _TELayerNormMLPWrapper(new_mlp)


def convert_mlp_blocks_to_te_layernorm_mlp(
    module: nn.Module,
    *,
    srelu_only: bool = False,
    skip_last_n_layers: int = 0,
) -> int:
    """Replace eligible ``mlp_norm + MLP`` branches with TE ``LayerNormMLP``."""

    _require_te()
    skipped_layers = _skip_last_transformer_layers(module, skip_last_n_layers)
    replacements: list[tuple[str, nn.Module]] = []
    for fqn, child in module.named_modules():
        if child in skipped_layers:
            continue
        activation = _te_layernorm_mlp_activation(child)
        if activation is not None and (not srelu_only or activation == "srelu"):
            replacements.append((fqn, child))

    for fqn, layer in replacements:
        te_mlp = _copy_mlp_branch_to_te_layernorm_mlp(
            layer,
            name=f"{fqn}.layernorm_mlp" if fqn else "layernorm_mlp",
        )
        layer.mlp_norm = nn.Identity()
        layer.mlp = te_mlp

    return len(replacements)


def _is_attn_wo_candidate(layer: nn.Module) -> bool:
    attn = getattr(layer, "attn", None)
    if attn is None:
        return False
    return isinstance(getattr(attn, "Wo", None), nn.Linear)


def convert_attention_wo_to_te_linear(
    module: nn.Module,
    *,
    skip_last_n_layers: int = 0,
) -> int:
    """Replace eligible attention output projections (``attn.Wo``) with TE ``Linear``.

    Unlike the qkv/mlp targets there is no preceding norm to absorb, so this is
    a plain fp8 Linear swap.
    """

    te_mod = _require_te()
    skipped_layers = _skip_last_transformer_layers(module, skip_last_n_layers)
    replacements: list[tuple[str, nn.Module]] = []
    for fqn, child in module.named_modules():
        if child in skipped_layers:
            continue
        if _is_attn_wo_candidate(child):
            replacements.append((fqn, child))

    for fqn, layer in replacements:
        linear = layer.attn.Wo
        new_linear = te_mod.Linear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            params_dtype=linear.weight.dtype,
            device=linear.weight.device,
            name=f"{fqn}.attn_wo" if fqn else "attn_wo",
        )
        with torch.no_grad():
            new_linear.weight.copy_(linear.weight)
            if linear.bias is not None:
                new_linear.bias.copy_(linear.bias)
        new_linear.train(linear.training)
        layer.attn.Wo = new_linear

    return len(replacements)


def _attention_qkv_linears(attn: nn.Module) -> list[tuple[str, nn.Linear]] | None:
    wqkv = getattr(attn, "Wqkv", None)
    if isinstance(wqkv, nn.Linear):
        return [("Wqkv", wqkv)]

    parts: list[tuple[str, nn.Linear]] = []
    for name in ("Wq", "Wq2", "Wk", "Wv"):
        linear = getattr(attn, name, None)
        if linear is None:
            continue
        if not isinstance(linear, nn.Linear):
            return None
        parts.append((name, linear))
    if {"Wq", "Wk", "Wv"}.issubset({name for name, _ in parts}):
        return parts
    return None


def _is_attn_wqkv_candidate(layer: nn.Module) -> bool:
    attn_norm = getattr(layer, "attn_norm", None)
    attn = getattr(layer, "attn", None)
    if attn is None or attn_norm is None:
        return False
    if getattr(layer, "canon_a", None) is not None:
        return False
    if not isinstance(attn_norm, (nn.LayerNorm, nn.RMSNorm)):
        return False
    return _attention_qkv_linears(attn) is not None


def convert_attention_qkv_to_te_layernorm_linear(
    module: nn.Module,
    *,
    skip_last_n_layers: int = 0,
) -> int:
    """Replace eligible ``attn_norm + Q/K/V`` projections with TE ``LayerNormLinear``."""

    _require_te()
    skipped_layers = _skip_last_transformer_layers(module, skip_last_n_layers)
    replacements: list[tuple[str, nn.Module]] = []
    for fqn, child in module.named_modules():
        if child in skipped_layers:
            continue
        if _is_attn_wqkv_candidate(child):
            replacements.append((fqn, child))

    for fqn, layer in replacements:
        parts = _attention_qkv_linears(layer.attn)
        if parts is None:
            continue
        for name, linear in parts:
            setattr(
                layer.attn,
                name,
                _copy_linear_to_te_layernorm_linear(
                    linear=linear,
                    norm=layer.attn_norm,
                    name=f"{fqn}.attn_{name}" if fqn else f"attn_{name}",
                ),
            )
        layer.attn_norm = nn.Identity()

    return len(replacements)


def _normalize_fp8_targets(raw_targets) -> list[str]:
    aliases = {
        "mlp": "mlp",
        "attn_wqkv": "attn_wqkv",
        "attn-wqkv": "attn_wqkv",
        "attention_wqkv": "attn_wqkv",
        "attention-wqkv": "attn_wqkv",
        "qkv": "attn_wqkv",
        "attn_wo": "attn_wo",
        "attn-wo": "attn_wo",
        "attention_wo": "attn_wo",
        "attention-wo": "attn_wo",
        "wo": "attn_wo",
    }
    if isinstance(raw_targets, str):
        pieces = raw_targets.replace("+", ",").split(",")
    elif isinstance(raw_targets, (list, tuple)):
        pieces = []
        for item in raw_targets:
            pieces.extend(str(item).replace("+", ",").split(","))
    else:
        pieces = [str(raw_targets)]

    targets: list[str] = []
    for piece in pieces:
        key = str(piece).strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"Unsupported pure-torch FP8 target: {raw_targets!r}")
        target = aliases[key]
        if target not in targets:
            targets.append(target)
    if not targets:
        raise ValueError("Unsupported empty pure-torch FP8 target.")
    return targets


def convert_fp8_targets(
    module: nn.Module,
    *,
    target: str | list[str] = "mlp",
    skip_last_n_layers: int = 0,
) -> int:
    converted = 0
    for item in _normalize_fp8_targets(target):
        if item == "mlp":
            converted += convert_mlp_blocks_to_te_layernorm_mlp(
                module,
                skip_last_n_layers=skip_last_n_layers,
            )
        elif item == "attn_wqkv":
            converted += convert_attention_qkv_to_te_layernorm_linear(
                module,
                skip_last_n_layers=skip_last_n_layers,
            )
        elif item == "attn_wo":
            converted += convert_attention_wo_to_te_linear(
                module,
                skip_last_n_layers=skip_last_n_layers,
            )
        else:  # pragma: no cover - guarded by _normalize_fp8_targets
            raise ValueError(f"Unsupported pure-torch FP8 target: {item!r}")
    return converted


def te_fp8_format_from_config(cfg) -> Format:
    _require_te()
    fp8_format = str(getattr(cfg, "fp8_format", "hybrid")).strip().lower()
    if fp8_format == "hybrid":
        return Format.HYBRID
    if fp8_format == "e4m3":
        return Format.E4M3
    raise ValueError(
        "fp8_format must be one of: hybrid, e4m3 "
        f"(got {getattr(cfg, 'fp8_format', None)!r})"
    )


def te_fp8_recipe_from_config(cfg):
    _require_te()
    recipe_name = str(getattr(cfg, "fp8_recipe", "delayed")).strip().lower()
    fp8_format = te_fp8_format_from_config(cfg)
    if recipe_name == "delayed":
        return DelayedScaling(
            fp8_format=fp8_format,
            amax_history_len=int(getattr(cfg, "fp8_amax_history_len", 64)),
            reduce_amax=bool(getattr(cfg, "fp8_reduce_amax", True)),
        )
    if recipe_name == "current":
        return Float8CurrentScaling(fp8_format=fp8_format)
    if recipe_name == "block":
        return Float8BlockScaling(fp8_format=fp8_format)
    if recipe_name == "mxfp8":
        return MXFP8BlockScaling(fp8_format=fp8_format)
    if recipe_name == "nvfp4":
        return NVFP4BlockScaling()
    raise ValueError(
        "fp8_recipe must be one of: delayed, current, block, mxfp8, nvfp4 "
        f"(got {getattr(cfg, 'fp8_recipe', None)!r})"
    )


def te_fp8_autocast_context(*, enabled: bool, cfg, distributed: bool):
    if not enabled:
        return nullcontext()
    te_mod = _require_te()
    recipe = te_fp8_recipe_from_config(cfg)
    fp8_group = dist.group.WORLD if (distributed and dist.is_initialized()) else None
    return te_mod.autocast(
        enabled=True,
        recipe=recipe,
        amax_reduction_group=fp8_group,
    )


__all__ = [
    "convert_attention_qkv_to_te_layernorm_linear",
    "convert_attention_wo_to_te_linear",
    "convert_fp8_targets",
    "convert_mlp_blocks_to_te_layernorm_mlp",
    "iter_te_cpu_offload_target_wrappers",
    "te_fp8_autocast_context",
    "te_fp8_format_from_config",
    "te_fp8_recipe_from_config",
]
