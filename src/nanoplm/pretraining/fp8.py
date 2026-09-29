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
            new_mlp.fc1_weight.copy_(native_wi_to_te_fc1(wi.weight, activation))
            new_mlp.fc2_weight.copy_(native_wo_to_te_fc2(mlp.Wo.weight, activation))
        else:
            new_mlp.fc1_weight.copy_(native_wi_to_te_fc1(wi.weight, activation))
            new_mlp.fc2_weight.copy_(native_wo_to_te_fc2(mlp.Wo_weight, activation))
        if wi.bias is not None:
            new_mlp.fc1_bias.copy_(native_wi_to_te_fc1(wi.bias, activation))
            if gated_activation:
                new_mlp.fc2_bias.copy_(mlp.Wo.bias)
            else:
                new_mlp.fc2_bias.copy_(mlp.Wo_bias)
    new_mlp.train(mlp.training)
    return _TELayerNormMLPWrapper(new_mlp)


# ---------------------------------------------------------------------------
# Dense restore (FP8/TE layout -> native modules). Used by `nanoplm export`.
#
# The forward conversion above has three fc1/fc2 layouts, keyed by the same
# activation strings `_te_layernorm_mlp_activation` returns:
#   - "swiglu" (ModernBertSwiGLUMLP): fc1 = [gate, up] from native Wi =
#     [up, gate] -- halves swap, for weight and bias alike. TE applies the
#     nonlinearity to the FIRST half of fc1; ModernBertSwiGLUMLP applies silu
#     to the SECOND half of Wi, so the halves have to swap on the way in.
#   - "geglu" (ModernBertMLP, act=F.gelu): fc1 = Wi unchanged -- ModernBertMLP
#     already gates on the first half, exactly like TE.
#   - "srelu" (ModernBertSReluMLP): fc1 = Wi unchanged; fc2 = Wo_weight.T
#     (sRelu keeps Wo as (intermediate, hidden) to match its Triton kernel).
# fc2 is unchanged for both gated forms; fc2 biases carry over unchanged in
# all three layouts.
#
# These are pure tensor functions so they can be unit-tested on CPU without
# Transformer Engine installed. The module-level walk that uses them for the
# restore direction is a separate function that needs TE.
# ---------------------------------------------------------------------------


def native_wi_to_te_fc1(t: torch.Tensor, activation: str) -> torch.Tensor:
    """Native fused ``Wi`` (weight or bias) -> TE ``fc1``. Mirrors the forward path.

    Only "swiglu" swaps the dim-0 halves; "geglu" and "srelu" pass through.
    Swapping a non-swiglu tensor would silently compute silu(up) * gate
    instead of the model's actual activation.
    """
    if activation == "swiglu":
        if t.shape[0] % 2 != 0:
            raise ValueError(
                f"swiglu Wi must have an even first dimension, got {tuple(t.shape)}"
            )
        up, gate = t.chunk(2, dim=0)
        return torch.cat([gate, up], dim=0)
    if activation in {"geglu", "srelu"}:
        return t
    raise ValueError(f"Unsupported activation for fc1 conversion: {activation!r}")


def te_fc1_to_native_wi(t: torch.Tensor, activation: str) -> torch.Tensor:
    """TE ``fc1`` (weight or bias) -> native fused ``Wi``. Inverse of ``native_wi_to_te_fc1``.

    TE stores [gate, up] for swiglu; native stores [up, gate]. Swapping these
    backwards silently computes silu(up) * gate instead of silu(gate) * x_proj
    -- a model that loads, embeds plausibly, and is wrong.
    """
    if activation == "swiglu":
        if t.shape[0] % 2 != 0:
            raise ValueError(
                f"swiglu fc1 must have an even first dimension, got {tuple(t.shape)}"
            )
        gate, up = t.chunk(2, dim=0)
        return torch.cat([up, gate], dim=0)
    if activation in {"geglu", "srelu"}:
        return t
    raise ValueError(f"Unsupported activation for fc1 conversion: {activation!r}")


def native_wo_to_te_fc2(w: torch.Tensor, activation: str) -> torch.Tensor:
    """Native ``Wo`` weight -> TE ``fc2`` weight. Mirrors the forward path.

    Only "srelu" transposes; "swiglu"/"geglu" pass through. Weights only --
    fc2 biases carry over unchanged in every layout.
    """
    if activation == "srelu":
        if w.ndim != 2:
            raise ValueError(f"srelu Wo must be 2-D, got {tuple(w.shape)}")
        return w.T.contiguous()
    if activation in {"swiglu", "geglu"}:
        return w
    raise ValueError(f"Unsupported activation for fc2 conversion: {activation!r}")


def te_fc2_to_native_wo(w: torch.Tensor, activation: str) -> torch.Tensor:
    """TE ``fc2`` weight -> native ``Wo`` weight. Inverse of ``native_wo_to_te_fc2``.

    sReLU keeps Wo as (intermediate, hidden) to match its Triton kernel layout,
    while TE's fc2 is (hidden, intermediate); the gated forms need no change.
    """
    if activation == "srelu":
        if w.ndim != 2:
            raise ValueError(f"srelu fc2 must be 2-D, got {tuple(w.shape)}")
        return w.T.contiguous()
    if activation in {"swiglu", "geglu"}:
        return w
    raise ValueError(f"Unsupported activation for fc2 conversion: {activation!r}")


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


def _has_te_bias(bias: torch.Tensor) -> bool:
    """TE linear/MLP bias tensors are never ``None`` -- a disabled bias is an
    empty tensor, not a missing attribute. ``is not None`` would always be
    true and silently give every restored linear a spurious zero bias.
    """
    return bias.numel() > 0


_QKV_NAMES = ("Wqkv", "Wq", "Wq2", "Wk", "Wv")


def _te_norm_bias_max_abs(te_module: nn.Module) -> float:
    """Max |``layer_norm_bias``| of a TE LayerNorm* module; 0.0 when it has none.

    TE's RMSNorm modules set ``layer_norm_bias = None``; its LayerNorm modules
    always own one (zero-initialised, trainable), whether or not the native
    norm they replaced had a bias.
    """
    bias = getattr(te_module, "layer_norm_bias", None)
    if bias is None or bias.numel() == 0:
        return 0.0
    return float(bias.detach().abs().max())


def _refuse_unrepresentable_te_norm_biases(model: nn.Module, config) -> None:
    """Raise before any module is touched if a TE norm bias would be dropped.

    ``norm_type: layernorm`` builds a bias-free native norm
    (``norm_bias=False``, not user-settable), but the TE modules that replace
    it under FP8 own a trainable ``layer_norm_bias``, and FP8 training moves it
    away from zero. The dense architecture has nowhere to put those values, so
    restoring would silently change the model. An exactly-zero bias (converted,
    never trained) carries no information and is fine. Unfused Q/K/V parts
    each own a separate, untied bias, so every wrapped linear is checked.
    """
    from nanoplm.pretraining.models.modern_bert.modeling import _create_norm

    native_norm = _create_norm(
        config.hidden_size, config.norm_eps, config.norm_bias, config.norm_type
    )
    if getattr(native_norm, "bias", None) is not None:
        return  # the native norm has a bias slot; the restore copies into it

    offenders: list[tuple[str, float]] = []
    for fqn, layer in model.named_modules():
        prefix = f"{fqn}." if fqn else ""
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, _TELayerNormMLPWrapper):
            max_abs = _te_norm_bias_max_abs(mlp.layernorm_mlp)
            if max_abs != 0.0:
                offenders.append((f"{prefix}mlp", max_abs))
        attn = getattr(layer, "attn", None)
        if attn is None:
            continue
        for name in _QKV_NAMES:
            wrapped = getattr(attn, name, None)
            if isinstance(wrapped, _TELayerNormLinearWrapper):
                max_abs = _te_norm_bias_max_abs(wrapped.layernorm_linear)
                if max_abs != 0.0:
                    offenders.append((f"{prefix}attn.{name}", max_abs))

    if not offenders:
        return
    where = ", ".join(f"{path} (max |bias| = {m:.4g})" for path, m in offenders)
    raise ValueError(
        "Cannot export this checkpoint without changing the model. It was "
        "trained with FP8 and LayerNorm, and Transformer Engine learned "
        f"LayerNorm biases in {len(offenders)} block(s): {where}. nanoPLM's "
        "dense architecture uses a bias-free LayerNorm, so it has nowhere to "
        "put these biases; exporting would silently drop them and change the "
        "model's outputs. Checkpoints trained with RMSNorm, or without FP8, "
        "are not affected."
    )


def _restore_mlp_branch(layer: nn.Module, config) -> None:
    """Rebuild native ``mlp_norm`` + ``mlp`` from a ``_TELayerNormMLPWrapper``."""
    from nanoplm.pretraining.models.modern_bert.modeling import (
        _create_dense_mlp,
        _create_norm,
    )

    te_mlp = layer.mlp.layernorm_mlp
    # Not `layer.mlp_norm`/`layer.mlp` -- convert leaves those as freshly
    # constructed `nn.Identity()`/wrapper instances (always training=True);
    # only the inner TE module carries the real flag, set by convert's own
    # `new_mlp.train(mlp.training)`.
    old_training = te_mlp.training
    device = te_mlp.fc1_weight.device
    dtype = te_mlp.fc1_weight.dtype

    norm = _create_norm(
        config.hidden_size, config.norm_eps, config.norm_bias, config.norm_type
    ).to(device=device, dtype=dtype)
    mlp = _create_dense_mlp(config).to(device=device, dtype=dtype)
    norm.train(old_training)
    mlp.train(old_training)

    layer.mlp_norm = norm
    layer.mlp = mlp

    activation = _te_layernorm_mlp_activation(layer)
    if activation != te_mlp.activation:
        raise ValueError(
            "MLP activation mismatch during restore: the config implies "
            f"{activation!r}, but the converted checkpoint was built with "
            f"{te_mlp.activation!r}."
        )
    gated = activation in {"swiglu", "geglu"}

    with torch.no_grad():
        norm.weight.copy_(te_mlp.layer_norm_weight)
        # A bias-free native norm skips this; restore_dense_from_te has already
        # refused if the TE bias it leaves behind is nonzero.
        if getattr(norm, "bias", None) is not None:
            norm.bias.copy_(te_mlp.layer_norm_bias)

        mlp.Wi.weight.copy_(te_fc1_to_native_wi(te_mlp.fc1_weight, activation))
        if mlp.Wi.bias is not None:
            mlp.Wi.bias.copy_(te_fc1_to_native_wi(te_mlp.fc1_bias, activation))

        if gated:
            mlp.Wo.weight.copy_(te_fc2_to_native_wo(te_mlp.fc2_weight, activation))
            if mlp.Wo.bias is not None:
                mlp.Wo.bias.copy_(te_mlp.fc2_bias)
        else:
            mlp.Wo_weight.copy_(te_fc2_to_native_wo(te_mlp.fc2_weight, activation))
            if mlp.Wo_bias is not None:
                mlp.Wo_bias.copy_(te_mlp.fc2_bias)


def _restore_attn_wo(layer: nn.Module) -> None:
    """Replace ``attn.Wo`` (a ``te.Linear``) with a copy in ``nn.Linear``."""
    wo = layer.attn.Wo
    dense_wo = nn.Linear(
        wo.in_features, wo.out_features, bias=_has_te_bias(wo.bias)
    ).to(device=wo.weight.device, dtype=wo.weight.dtype)
    with torch.no_grad():
        dense_wo.weight.copy_(wo.weight)
        if dense_wo.bias is not None:
            dense_wo.bias.copy_(wo.bias)
    dense_wo.train(wo.training)
    layer.attn.Wo = dense_wo


def _attn_has_te_qkv(layer: nn.Module) -> bool:
    attn = getattr(layer, "attn", None)
    if attn is None:
        return False
    return any(
        isinstance(getattr(attn, name, None), _TELayerNormLinearWrapper)
        for name in _QKV_NAMES
    )


def _restore_attn_qkv_branch(layer: nn.Module, config) -> None:
    """Rebuild native ``attn_norm`` + Q/K/V linears from TE ``LayerNormLinear``.

    Every converted part shares one tied norm (``_tie_norm_to_te_layernorm_module``
    shares the Parameter), so it is read off the first wrapper and restored once.
    """
    from nanoplm.pretraining.models.modern_bert.modeling import _create_norm

    attn = layer.attn
    wrapped_names = [
        name
        for name in _QKV_NAMES
        if isinstance(getattr(attn, name, None), _TELayerNormLinearWrapper)
    ]
    first = getattr(attn, wrapped_names[0]).layernorm_linear
    device = first.weight.device
    dtype = first.weight.dtype

    # Not `layer.attn_norm` -- convert leaves that as a freshly constructed
    # `nn.Identity()` (always training=True); `first` is the inner TE module,
    # which carries the real flag from convert's own `new_linear.train(...)`.
    norm = _create_norm(
        config.hidden_size, config.norm_eps, config.norm_bias, config.norm_type
    ).to(device=device, dtype=dtype)
    norm.train(first.training)
    with torch.no_grad():
        norm.weight.copy_(first.layer_norm_weight)
        # As in _restore_mlp_branch: a nonzero TE bias with no native slot was
        # already refused by restore_dense_from_te, for every wrapped part.
        if getattr(norm, "bias", None) is not None:
            norm.bias.copy_(first.layer_norm_bias)

    for name in wrapped_names:
        wrapped = getattr(attn, name)
        te_linear = wrapped.layernorm_linear
        dense = nn.Linear(
            te_linear.in_features,
            te_linear.out_features,
            bias=_has_te_bias(te_linear.bias),
        ).to(device=te_linear.weight.device, dtype=te_linear.weight.dtype)
        with torch.no_grad():
            dense.weight.copy_(te_linear.weight)
            if dense.bias is not None:
                dense.bias.copy_(te_linear.bias)
        # Not `wrapped.training` -- `wrapped` is the freshly constructed
        # `_TELayerNormLinearWrapper` (always training=True); `te_linear` is
        # the inner TE module that actually carries the flag.
        dense.train(te_linear.training)
        setattr(attn, name, dense)

    layer.attn_norm = norm


def restore_dense_from_te(model: nn.Module, config) -> int:
    """Invert ``convert_fp8_targets``, in place. Returns the block count restored.

    Export writes a portable, dense model; TE modules would demand
    Transformer Engine on every consumer machine. Walks the tree, finds every
    TE wrapper ``convert_fp8_targets`` installed, and rebuilds the native
    modules it replaced, using the transforms above. Counts one restored
    block per converted LAYER, matching ``convert_fp8_targets``'s own count --
    an unfused QKV layer touches three or four linears but is still one block.

    Raises ``ValueError``, before modifying anything, if a TE module carries a
    nonzero LayerNorm bias the native (bias-free) norm cannot hold -- see
    ``_refuse_unrepresentable_te_norm_biases``.
    """
    _require_te()
    _refuse_unrepresentable_te_norm_biases(model, config)

    mlp_layers = [
        layer
        for layer in model.modules()
        if isinstance(getattr(layer, "mlp", None), _TELayerNormMLPWrapper)
    ]
    wo_layers = [
        layer
        for layer in model.modules()
        if isinstance(getattr(getattr(layer, "attn", None), "Wo", None), te.Linear)
    ]
    qkv_layers = [layer for layer in model.modules() if _attn_has_te_qkv(layer)]

    for layer in mlp_layers:
        _restore_mlp_branch(layer, config)
    for layer in wo_layers:
        _restore_attn_wo(layer)
    for layer in qkv_layers:
        _restore_attn_qkv_branch(layer, config)

    return len(mlp_layers) + len(wo_layers) + len(qkv_layers)


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
    "restore_dense_from_te",
    "te_fp8_autocast_context",
    "te_fp8_format_from_config",
    "te_fp8_recipe_from_config",
]
