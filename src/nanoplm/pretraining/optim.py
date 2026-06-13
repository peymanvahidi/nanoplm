"""Optimizer utilities shared by pretraining pipelines."""

from __future__ import annotations

import torch
import torch.distributed as dist

from nanoplm.utils.logger import logger
from nanoplm.pretraining.config import PretrainingConfig


class MuonAdamWGroup:
    """Wraps a Dion Muon/NorMuon optimizer (muon params only) and a separate
    torch.optim.AdamW (fused=True) for the AdamW param groups.

    Exposes a unified interface so the training loop can treat it as a single
    optimizer: .step(), .zero_grad(), .state_dict(), .load_state_dict(),
    and .param_groups (muon groups first, then adamw groups).
    """

    def __init__(self, muon_optimizer, adamw_optimizer):
        self.muon = muon_optimizer
        self.adamw = adamw_optimizer
        # LambdaLR stores defaults from optimizer; provide a minimal one.
        self.defaults = {"lr": muon_optimizer.defaults.get("lr", 1e-3)}

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    def step(self, closure=None):
        self.muon.step(closure)
        self.adamw.step(closure)

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if "muon" in state_dict and "adamw" in state_dict:
            self.muon.load_state_dict(state_dict["muon"])
            self.adamw.load_state_dict(state_dict["adamw"])
        else:
            logger.warning(
                "Optimizer checkpoint uses legacy single-optimizer format. "
                "Skipping optimizer state restore — momentum/variance buffers "
                "will restart from scratch."
            )


class OptimizerGroup:
    """Small optimizer-like wrapper for stepping multiple inner optimizers."""

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)
        if not self.optimizers:
            raise ValueError("OptimizerGroup requires at least one optimizer.")
        self.defaults = getattr(self.optimizers[0], "defaults", {})

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def __iter__(self):
        return iter(self.optimizers)

    def __len__(self):
        return len(self.optimizers)

    def step(self, closure=None):
        loss = None
        for opt in self.optimizers:
            loss = opt.step(closure)
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        states = state_dict.get("optimizers")
        if not isinstance(states, list) or len(states) != len(self.optimizers):
            logger.warning(
                "Composite Muon optimizer checkpoint does not match current optimizer "
                "layout. Skipping Muon-side optimizer state restore."
            )
            return
        for opt, sd in zip(self.optimizers, states):
            opt.load_state_dict(sd)


def is_muon_optimizer(optimizer) -> bool:
    """Return True if *optimizer* is a Dion matrix optimizer or a MuonAdamWGroup."""
    if isinstance(optimizer, MuonAdamWGroup):
        return True
    classes: list = []
    try:
        from dion import Muon as DionMuon
        classes.append(DionMuon)
    except ImportError:
        pass
    try:
        from dion import NorMuon as DionNorMuon
        classes.append(DionNorMuon)
    except ImportError:
        pass
    try:
        from dion import MuonH as DionMuonH
        classes.append(DionMuonH)
    except ImportError:
        pass
    try:
        from dion import Aurora as DionAurora
        classes.append(DionAurora)
    except ImportError:
        pass
    return bool(classes) and isinstance(optimizer, tuple(classes))


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _is_embedding_or_unembedding_param(name: str) -> bool:
    lname = name.lower()
    if "embeddings.tok_embeddings" in lname:
        return True
    if lname.endswith("decoder.weight") or lname.endswith("decoder.bias"):
        return True
    return (
        "embedding" in lname
        or "lm_head" in lname
        or "unembedding" in lname
    )


def _is_zero_init_fragile_param(name: str) -> bool:
    """Detect 2-D parameters that are zero-initialized and have structurally
    dormant rows, making them unsafe for NorMuon's per-row variance tracking.

    Currently matches:
      - mHC-lite W_all.weight  (32×3072, 75% of rows intentionally suppressed)
      - RePO W_z.weight        (8×96, zero-init with per-head rows)
      - DiffAttnV2 lambda_proj.weight (num_heads×hidden, zero-init, very few rows)
    """
    parts = name.split(".")
    if len(parts) >= 2 and parts[-2] == "W_all" and parts[-1] == "weight":
        return True
    if len(parts) >= 2 and parts[-2] == "W_z" and parts[-1] == "weight":
        return True
    if len(parts) >= 2 and parts[-2] == "lambda_proj" and parts[-1] == "weight":
        return True
    return False

def _classify_noble_param(name: str) -> str | None:
    """Classify a NOBLE param for optimizer routing.

    Returns one of: 'wup', 'wdown', 'M', 'freq', 'phase', or None (not a NOBLE
    param). Per paper §3.3, W_down uses the base learning rate (AdamW at
    noble_base_lr), not Muon.
    """
    if ".cosnet.M" in name:
        return "M"
    if ".cosnet.omega" in name:
        return "freq"
    if ".cosnet.phi" in name:
        return "phase"
    if name.endswith(".noble_w_up.weight"):
        return "wup"
    if name.endswith(".W_down.weight"):
        return "wdown"
    return None


def _is_router_param(name: str) -> bool:
    parts = name.split(".")
    return len(parts) >= 3 and parts[-3:] == ["router", "gate", "weight"]


def _is_moe_routed_expert_param(name: str) -> bool:
    parts = name.split(".")
    return len(parts) >= 2 and parts[-2:] in (["mlp", "Wi"], ["mlp", "Wo"])


def _is_mlp_matrix_param(name: str) -> bool:
    lname = name.lower()
    return ".mlp." in lname or ".mhc_mlp." in lname or ".shared_expert." in lname


def _collect_qkv_split_sizes(raw_model: torch.nn.Module) -> dict[int, tuple[int, ...]]:
    """Map id(param) -> row-block sizes for fused QKV weights.

    With fused_qkv=true the q/k/v projections share one parameter, which Muon
    would otherwise orthogonalize as a single tall matrix, blending the three
    projections. Dion's per-group ``split_sizes`` option runs Newton-Schulz on
    the (q, k, v) row blocks independently — the same update separate Wq/Wk/Wv
    parameters would receive — while keeping the single wide GEMM. Block sizes
    are derived from the owning attention module (GQA-aware); DiffAttnV2
    doubles the Q heads into two groups, mirrored here as two q blocks to
    match the unfused Wq/Wq2 layout.
    """
    mapping: dict[int, tuple[int, ...]] = {}
    for module in raw_model.modules():
        wqkv = getattr(module, "Wqkv", None)
        weight = getattr(wqkv, "weight", None)
        if weight is None and isinstance(wqkv, torch.nn.Module):
            # TE conversion wraps the projection (e.g. _TELayerNormLinearWrapper
            # holding a LayerNormLinear), so the fused weight is nested. The
            # wrapper holds exactly one 2D parameter: the fused qkv weight.
            two_d = [p for p in wqkv.parameters() if p.ndim == 2]
            weight = two_d[0] if len(two_d) == 1 else None
        if weight is None or weight.ndim != 2:
            continue
        head_dim = getattr(module, "head_dim", None)
        num_q_heads = getattr(module, "num_q_heads", None)
        num_kv_heads = getattr(module, "num_kv_heads", None)
        if head_dim is None or num_q_heads is None or num_kv_heads is None:
            continue
        q_dim = num_q_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        if getattr(module, "lambda_proj", None) is not None:
            sizes = (q_dim // 2, q_dim // 2, kv_dim, kv_dim)
        else:
            sizes = (q_dim, kv_dim, kv_dim)
        if weight.shape[0] != sum(sizes):
            logger.warning(
                f"Skipping QKV split for a Wqkv weight of shape "
                f"{tuple(weight.shape)}: derived blocks {sizes} do not sum to "
                f"dim 0."
            )
            continue
        mapping[id(weight)] = sizes
    return mapping


def build_muon_optimizer(
    model: torch.nn.Module,
    pretrain_config: PretrainingConfig,
    distributed_mesh=None,
) -> MuonAdamWGroup:
    raw_model = unwrap_model(model)
    hyperball_mlp_only = (
        str(pretrain_config.optimizer).lower() in {"muonh", "aurora"}
        and str(getattr(pretrain_config, "muonh_scope", "all")).lower() == "mlp"
    )
    hyperball_all = (
        str(pretrain_config.optimizer).lower() in {"muonh", "aurora"}
        and not hyperball_mlp_only
    )

    muon_params: list[torch.nn.Parameter] = []
    hyperball_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    resid_params: list[torch.nn.Parameter] = []
    x0_params: list[torch.nn.Parameter] = []
    # NOBLE param groups (separate LR each)
    noble_wup_params: list[torch.nn.Parameter] = []
    noble_wdown_params: list[torch.nn.Parameter] = []
    noble_M_params: list[torch.nn.Parameter] = []
    noble_freq_params: list[torch.nn.Parameter] = []
    noble_phase_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()

    for name, param in raw_model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        lname = name.lower()

        if lname.endswith("resid_lambdas") or ".resid_lambdas" in lname:
            resid_params.append(param)
        elif lname.endswith("x0_lambdas") or ".x0_lambdas" in lname:
            x0_params.append(param)
        # NOBLE routing (must come before generic ndim checks).
        # Per paper §3.3: W_down uses the base learning rate (AdamW at
        # noble_base_lr), the cosnet/W_up params get elevated LRs.
        elif (noble_kind := _classify_noble_param(name)) is not None:
            {"wup": noble_wup_params, "wdown": noble_wdown_params,
             "M": noble_M_params, "freq": noble_freq_params,
             "phase": noble_phase_params}[noble_kind].append(param)
        elif param.ndim == 1 or _is_embedding_or_unembedding_param(name):
            adamw_params.append(param)
        elif param.ndim == 2:
            if _is_zero_init_fragile_param(name) or _is_router_param(name):
                adamw_params.append(param)
            elif hyperball_mlp_only and _is_mlp_matrix_param(name):
                hyperball_params.append(param)
            elif hyperball_all:
                hyperball_params.append(param)
            else:
                muon_params.append(param)
        elif param.ndim == 3 and _is_moe_routed_expert_param(name):
            # MoE routed experts are stored as stacked 3D tensors for grouped_mm.
            # NorMuon supports batched 2D updates, so keep these on the Muon side
            # instead of falling back to AdamW just because of the storage layout.
            if hyperball_mlp_only and _is_mlp_matrix_param(name):
                hyperball_params.append(param)
            elif hyperball_all:
                hyperball_params.append(param)
            else:
                muon_params.append(param)
        else:
            adamw_params.append(param)

    if not muon_params and not hyperball_params:
        raise ValueError(
            "No eligible matrix parameters found for Muon (expected 2D hidden-layer weights)."
        )

    # Route fused QKV weights on the Muon/NorMuon side into per-split_sizes
    # groups so q/k/v are orthogonalized independently. Hyperball params are
    # left untouched: MuonH/Aurora do not support split_sizes and would
    # silently ignore the group option.
    muon_split_groups: dict[tuple[int, ...], list[torch.nn.Parameter]] = {}
    qkv_split_map = _collect_qkv_split_sizes(raw_model)
    if qkv_split_map and muon_params:
        remaining: list[torch.nn.Parameter] = []
        for param in muon_params:
            sizes = qkv_split_map.get(id(param))
            if sizes is not None:
                muon_split_groups.setdefault(sizes, []).append(param)
            else:
                remaining.append(param)
        muon_params = remaining
        if muon_split_groups:
            logger.info(
                "Fused-QKV Muon split groups: "
                + ", ".join(
                    f"{len(ps)} tensors with split_sizes={sizes}"
                    for sizes, ps in muon_split_groups.items()
                )
            )

    muon_numel = sum(p.numel() for p in muon_params)
    hyperball_numel = sum(p.numel() for p in hyperball_params)
    adamw_numel = sum(p.numel() for p in adamw_params)
    resid_numel = sum(p.numel() for p in resid_params)
    x0_numel = sum(p.numel() for p in x0_params)
    noble_total = (
        len(noble_wup_params) + len(noble_wdown_params) + len(noble_M_params)
        + len(noble_freq_params) + len(noble_phase_params)
    )
    noble_numel = (
        sum(p.numel() for p in noble_wup_params)
        + sum(p.numel() for p in noble_wdown_params)
        + sum(p.numel() for p in noble_M_params)
        + sum(p.numel() for p in noble_freq_params)
        + sum(p.numel() for p in noble_phase_params)
    )
    logger.info(
        f"Muon grouping: muon_params={len(muon_params)} tensors, "
        f"hyperball_params={len(hyperball_params)} tensors, "
        f"adamw_params={len(adamw_params)} tensors, "
        f"resid_scalar_params={len(resid_params)} tensors, "
        f"x0_scalar_params={len(x0_params)} tensors; "
        f"muon_numel={muon_numel:,}, hyperball_numel={hyperball_numel:,}, "
        f"adamw_numel={adamw_numel:,}, "
        f"resid_numel={resid_numel:,}, x0_numel={x0_numel:,}"
        + (f", noble_params={noble_total} tensors (wup={len(noble_wup_params)}, "
           f"wdown={len(noble_wdown_params)}, "
           f"M={len(noble_M_params)}, freq={len(noble_freq_params)}, "
           f"phase={len(noble_phase_params)}, noble_numel={noble_numel:,})" if noble_total > 0 else "")
    )

    # Compute NOBLE LR values if any NOBLE params exist
    noble_lrs: dict | None = None
    if noble_total > 0:
        # Extract hidden_size from model for LR ratio computation
        hidden_size = getattr(getattr(raw_model, "config", None), "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(getattr(raw_model, "model_config", None), "hidden_size", None)
        noble_rank = getattr(getattr(raw_model, "config", None), "noble_rank", None)
        if noble_rank is None:
            noble_rank = getattr(getattr(raw_model, "model_config", None), "noble_rank", None)
        if hidden_size is None or noble_rank is None:
            raise ValueError(
                f"Cannot determine hidden_size (got {hidden_size}) / noble_rank "
                f"(got {noble_rank}) for NOBLE LR scaling. Ensure the model has "
                f".config.hidden_size and .config.noble_rank attributes."
            )
        base_lr = pretrain_config.noble_base_lr
        ratio = hidden_size / noble_rank
        noble_lrs = {
            "wup": base_lr * (ratio ** (2 * pretrain_config.noble_lr_gamma)),
            "wdown": base_lr,
            "M": base_lr * (ratio ** pretrain_config.noble_lr_gamma_M),
            "freq": base_lr * pretrain_config.noble_freq_lr_mult,
            "phase": base_lr * pretrain_config.noble_phase_lr_mult,
        }
        logger.info(
            f"NOBLE LR scaling: ratio={ratio:.1f}, "
            f"wup_lr={noble_lrs['wup']:.6f}, wdown_lr={noble_lrs['wdown']:.6f}, "
            f"M_lr={noble_lrs['M']:.6f}, "
            f"freq_lr={noble_lrs['freq']:.6f}, phase_lr={noble_lrs['phase']:.6f}"
        )

    return build_optimizer(
        muon_params=muon_params,
        muon_split_groups=muon_split_groups,
        muonh_params=hyperball_params,
        adamw_params=adamw_params,
        resid_params=resid_params,
        x0_params=x0_params,
        noble_wup_params=noble_wup_params,
        noble_wdown_params=noble_wdown_params,
        noble_M_params=noble_M_params,
        noble_freq_params=noble_freq_params,
        noble_phase_params=noble_phase_params,
        noble_lrs=noble_lrs,
        muon_learning_rate=pretrain_config.muon_learning_rate,
        muon_weight_decay=pretrain_config.muon_weight_decay,
        muon_cautious_weight_decay=pretrain_config.muon_cautious_weight_decay,
        muon_use_polar_express=pretrain_config.muon_use_polar_express,
        muon_momentum=pretrain_config.muon_momentum,
        muon_nesterov=pretrain_config.muon_nesterov,
        muon_eps=pretrain_config.muon_eps,
        muon_variant=str(pretrain_config.optimizer).lower(),
        muonh_normalization=pretrain_config.muonh_normalization,
        adamw_learning_rate=pretrain_config.adam_learning_rate,
        adamw_weight_decay=pretrain_config.adam_weight_decay,
        adamw_betas=(pretrain_config.adam_beta1, pretrain_config.adam_beta2),
        adamw_epsilon=pretrain_config.adam_epsilon,
        distributed_mesh=distributed_mesh,
    )


polar_express_coeffs = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]
polar_express_coeffs = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in polar_express_coeffs[:-1]
] + [polar_express_coeffs[-1]]


@torch.compile(dynamic=False, fullgraph=True)
def _polar_express_paper(G: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + epsilon)
    if G.size(-2) > G.size(-1):
        for a, b, c in polar_express_coeffs:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X


def build_optimizer(
    muon_params: list[torch.nn.Parameter],
    muonh_params: list[torch.nn.Parameter] | None,
    adamw_params: list[torch.nn.Parameter],
    muon_learning_rate: float,
    muon_weight_decay: float,
    muon_cautious_weight_decay: bool,
    muon_use_polar_express: bool,
    muon_momentum: float,
    muon_nesterov: bool,
    muon_eps: float,
    adamw_learning_rate: float,
    adamw_weight_decay: float,
    adamw_betas: tuple[float, float],
    adamw_epsilon: float,
    resid_params: list[torch.nn.Parameter] | None = None,
    x0_params: list[torch.nn.Parameter] | None = None,
    noble_wup_params: list[torch.nn.Parameter] | None = None,
    noble_wdown_params: list[torch.nn.Parameter] | None = None,
    noble_M_params: list[torch.nn.Parameter] | None = None,
    noble_freq_params: list[torch.nn.Parameter] | None = None,
    noble_phase_params: list[torch.nn.Parameter] | None = None,
    noble_lrs: dict | None = None,
    muon_variant: str = "muon",
    muonh_normalization: str = "neuron",
    muon_split_groups: dict[tuple[int, ...], list[torch.nn.Parameter]] | None = None,
    distributed_mesh=None,
) -> MuonAdamWGroup:
    """Build a MuonAdamWGroup: Dion matrix optimizer(s) + torch.optim.AdamW(fused=True)."""
    muonh_params = muonh_params or []
    muon_split_groups = muon_split_groups or {}
    if not muon_params and not muon_split_groups and not muonh_params:
        raise ValueError("Muon optimizer requires at least one matrix parameter.")
    resid_params = resid_params or []
    x0_params = x0_params or []
    noble_wup_params = noble_wup_params or []
    noble_wdown_params = noble_wdown_params or []
    noble_M_params = noble_M_params or []
    noble_freq_params = noble_freq_params or []
    noble_phase_params = noble_phase_params or []
    has_adamw = (
        adamw_params or resid_params or x0_params
        or noble_wup_params or noble_wdown_params
    )
    if not has_adamw:
        raise ValueError("Muon optimizer requires at least one AdamW/scalar parameter.")

    if distributed_mesh is None:
        if dist.is_available() and dist.is_initialized():
            distributed_mesh = dist.group.WORLD

    ns_func = _polar_express_paper if muon_use_polar_express else None

    # Param groups for the Dion Muon/NorMuon side: plain matrix params plus
    # one group per distinct split_sizes (fused QKV weights whose q/k/v row
    # blocks are orthogonalized independently).
    def _muon_param_groups() -> list[dict]:
        groups: list[dict] = []
        if muon_params:
            groups.append(dict(params=muon_params))
        for sizes, params in muon_split_groups.items():
            groups.append(dict(params=params, split_sizes=list(sizes)))
        return groups

    NANOCHAT_RESID_LR = 0.005
    NANOCHAT_X0_LR = 0.5
    NANOCHAT_RESID_BETAS = (0.8, 0.95)
    NANOCHAT_X0_BETAS = (0.96, 0.95)
    NANOCHAT_SCALAR_EPS = 1e-10

    # -- Dion Muon/NorMuon/MuonH/Aurora: matrix params -----------------------
    import inspect
    from dion import Muon as DionMuon
    from dion import NorMuon as DionNorMuon

    variant = str(muon_variant).lower()
    if variant not in {"muon", "normuon", "muonh", "aurora"}:
        raise ValueError(
            f"Unknown muon_variant: {muon_variant!r}. Expected one of: muon, normuon, muonh, aurora."
        )

    common_kwargs = dict(
        distributed_mesh=distributed_mesh,
        lr=float(muon_learning_rate),
        mu=float(muon_momentum),
        epsilon=float(muon_eps),
        nesterov=bool(muon_nesterov),
        use_triton=True,
        newton_schulz_func=ns_func,
        adjust_lr="rms_norm",
    )
    if variant in {"muonh", "aurora"}:
        if variant == "muonh":
            try:
                from dion import MuonH as HyperballCls
            except ImportError as e:
                raise ImportError(
                    "optimizer='muonh' requires a dion build that exposes MuonH."
                ) from e
        else:
            try:
                from dion import Aurora as HyperballCls
            except ImportError as e:
                raise ImportError(
                    "optimizer='aurora' requires a dion build that exposes Aurora."
                ) from e
        optimizers = []
        if muon_params or muon_split_groups:
            normuon_kwargs = {
                **common_kwargs,
                "weight_decay": float(muon_weight_decay),
                "cautious_wd": bool(muon_cautious_weight_decay),
                "muon_beta2": 0.95,
            }
            if "normalization" in inspect.signature(DionNorMuon).parameters:
                normuon_kwargs["normalization"] = "neuron"
            optimizers.append(DionNorMuon(_muon_param_groups(), **normuon_kwargs))
        if muonh_params:
            hyperball_kwargs = {
                **common_kwargs,
                "betas": (0.9, 0.95),
                "weight_decay": float(muon_weight_decay),
            }
            if variant == "muonh":
                normalization = str(muonh_normalization).strip().lower()
                hyperball_kwargs["normalization"] = None if normalization == "none" else normalization
                hyperball_kwargs["muon_beta2"] = 0.95
                if "use_polar_express" in inspect.signature(HyperballCls).parameters:
                    hyperball_kwargs["use_polar_express"] = bool(muon_use_polar_express)
            else:
                if "cautious_wd" in inspect.signature(HyperballCls).parameters:
                    hyperball_kwargs["cautious_wd"] = bool(muon_cautious_weight_decay)
            optimizers.append(HyperballCls([dict(params=muonh_params)], **hyperball_kwargs))
        muon_opt = OptimizerGroup(optimizers) if len(optimizers) > 1 else optimizers[0]
    else:
        Cls = DionNorMuon if variant == "normuon" else DionMuon
        kwargs = {
            **common_kwargs,
            "weight_decay": float(muon_weight_decay),
            "cautious_wd": bool(muon_cautious_weight_decay),
        }
        if variant == "normuon":
            kwargs["muon_beta2"] = 0.95
            # `normalization` is only on the NorMuon PR branch; skip if absent.
            if "normalization" in inspect.signature(DionNorMuon).parameters:
                kwargs["normalization"] = "neuron"
        muon_opt = Cls(_muon_param_groups(), **kwargs)

    # -- torch.optim.AdamW(fused=True): adamw / resid / x0 params -------------
    adamw_groups: list[dict] = []
    if adamw_params:
        adamw_groups.append(dict(
            params=adamw_params,
            lr=float(adamw_learning_rate),
            weight_decay=float(adamw_weight_decay),
            betas=adamw_betas,
            eps=float(adamw_epsilon),
        ))
    if resid_params:
        adamw_groups.append(dict(
            params=resid_params,
            lr=NANOCHAT_RESID_LR,
            weight_decay=0.0,
            betas=NANOCHAT_RESID_BETAS,
            eps=NANOCHAT_SCALAR_EPS,
        ))
    if x0_params:
        adamw_groups.append(dict(
            params=x0_params,
            lr=NANOCHAT_X0_LR,
            weight_decay=0.0,
            betas=NANOCHAT_X0_BETAS,
            eps=NANOCHAT_SCALAR_EPS,
        ))
    # NOBLE param groups with paper-prescribed LRs (paper §3.3).
    # W_down uses the base LR (no scaling); W_up, M, freq, phase get elevated.
    if noble_wup_params and noble_lrs:
        adamw_groups.append(dict(
            params=noble_wup_params,
            lr=noble_lrs["wup"],
            weight_decay=float(adamw_weight_decay),
            betas=adamw_betas,
            eps=float(adamw_epsilon),
        ))
    if noble_wdown_params and noble_lrs:
        adamw_groups.append(dict(
            params=noble_wdown_params,
            lr=noble_lrs["wdown"],
            weight_decay=float(adamw_weight_decay),
            betas=adamw_betas,
            eps=float(adamw_epsilon),
        ))
    if noble_M_params and noble_lrs:
        adamw_groups.append(dict(
            params=noble_M_params,
            lr=noble_lrs["M"],
            weight_decay=float(adamw_weight_decay),
            betas=adamw_betas,
            eps=float(adamw_epsilon),
        ))
    if noble_freq_params and noble_lrs:
        adamw_groups.append(dict(
            params=noble_freq_params,
            lr=noble_lrs["freq"],
            weight_decay=float(adamw_weight_decay),
            betas=adamw_betas,
            eps=float(adamw_epsilon),
        ))
    if noble_phase_params and noble_lrs:
        adamw_groups.append(dict(
            params=noble_phase_params,
            lr=noble_lrs["phase"],
            weight_decay=float(adamw_weight_decay),
            betas=adamw_betas,
            eps=float(adamw_epsilon),
        ))

    adamw_opt = torch.optim.AdamW(adamw_groups, fused=True)

    return MuonAdamWGroup(muon_opt, adamw_opt)
