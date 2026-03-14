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


def is_muon_optimizer(optimizer) -> bool:
    """Return True if *optimizer* is a Dion Muon/NorMuon instance or a MuonAdamWGroup."""
    if isinstance(optimizer, MuonAdamWGroup):
        return True
    try:
        from dion import Muon as DionMuon
        from dion import NorMuon as DionNorMuon
        return isinstance(optimizer, (DionMuon, DionNorMuon))
    except ImportError:
        return False


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


def build_muon_optimizer(
    model: torch.nn.Module,
    pretrain_config: PretrainingConfig,
    distributed_mesh=None,
) -> MuonAdamWGroup:
    raw_model = unwrap_model(model)

    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    resid_params: list[torch.nn.Parameter] = []
    x0_params: list[torch.nn.Parameter] = []
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
        elif param.ndim == 1 or _is_embedding_or_unembedding_param(name):
            adamw_params.append(param)
        elif param.ndim == 2:
            if _is_zero_init_fragile_param(name):
                adamw_params.append(param)
            else:
                muon_params.append(param)
        else:
            adamw_params.append(param)

    if not muon_params:
        raise ValueError(
            "No eligible matrix parameters found for Muon (expected 2D hidden-layer weights)."
        )

    logger.info(
        f"Muon grouping: muon_params={len(muon_params)} tensors, "
        f"adamw_params={len(adamw_params)} tensors, "
        f"resid_scalar_params={len(resid_params)} tensors, "
        f"x0_scalar_params={len(x0_params)} tensors"
    )

    return build_optimizer(
        muon_params=muon_params,
        adamw_params=adamw_params,
        resid_params=resid_params,
        x0_params=x0_params,
        muon_learning_rate=pretrain_config.muon_learning_rate,
        muon_weight_decay=pretrain_config.muon_weight_decay,
        muon_cautious_weight_decay=pretrain_config.muon_cautious_weight_decay,
        muon_use_polar_express=pretrain_config.muon_use_polar_express,
        muon_momentum=pretrain_config.muon_momentum,
        muon_nesterov=pretrain_config.muon_nesterov,
        muon_eps=pretrain_config.muon_eps,
        use_normuon=str(pretrain_config.optimizer).lower() == "normuon",
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
    use_normuon: bool = False,
    distributed_mesh=None,
) -> MuonAdamWGroup:
    """Build a MuonAdamWGroup: Dion Muon/NorMuon for muon params + torch.optim.AdamW(fused=True) for AdamW params."""
    if not muon_params:
        raise ValueError("Muon optimizer requires at least one matrix parameter.")
    resid_params = resid_params or []
    x0_params = x0_params or []
    if not (adamw_params or resid_params or x0_params):
        raise ValueError("Muon optimizer requires at least one AdamW/scalar parameter.")

    if distributed_mesh is None:
        if dist.is_available() and dist.is_initialized():
            distributed_mesh = dist.group.WORLD

    ns_func = _polar_express_paper if muon_use_polar_express else None

    NANOCHAT_RESID_LR = 0.005
    NANOCHAT_X0_LR = 0.5
    NANOCHAT_RESID_BETAS = (0.8, 0.95)
    NANOCHAT_X0_BETAS = (0.96, 0.95)
    NANOCHAT_SCALAR_EPS = 1e-10

    # -- Dion Muon/NorMuon: only muon params ----------------------------------
    from dion import Muon as DionMuon
    from dion import NorMuon as DionNorMuon

    Cls = DionNorMuon if use_normuon else DionMuon
    kwargs = dict(
        distributed_mesh=distributed_mesh,
        lr=float(muon_learning_rate),
        mu=float(muon_momentum),
        weight_decay=float(muon_weight_decay),
        epsilon=float(muon_eps),
        nesterov=bool(muon_nesterov),
        use_triton=True,
        cautious_wd=bool(muon_cautious_weight_decay),
        newton_schulz_func=ns_func,
    )
    if use_normuon:
        kwargs["muon_beta2"] = 0.95
        kwargs["adjust_lr"] = "rms_norm"
    else:
        kwargs["adjust_lr"] = None

    muon_opt = Cls([dict(params=muon_params)], **kwargs)

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

    adamw_opt = torch.optim.AdamW(adamw_groups, fused=True)

    return MuonAdamWGroup(muon_opt, adamw_opt)
