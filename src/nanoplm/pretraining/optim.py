"""Optimizer utilities shared by pretraining pipelines."""

from __future__ import annotations

import torch
import torch.distributed as dist

from nanoplm.utils.logger import logger

def is_muon_optimizer(optimizer) -> bool:
    """Return True if *optimizer* is a Dion Muon or NorMuon instance."""
    try:
        from dion import Muon as DionMuon
        from dion import NorMuon as DionNorMuon
        return isinstance(optimizer, (DionMuon, DionNorMuon))
    except ImportError:
        return False

def build_muon_optimizer(
    model: torch.nn.Module,
    pretrain_config,
):
    """Partition model params into Muon (2D) and AdamW (1D/embedding) groups
    and build a Dion Muon/NorMuon optimizer.

    ``pretrain_config`` is expected to carry the standard Muon/AdamW hyper-
    parameter attributes (duck-typed so both pipeline flavours can call this).
    """
    raw_model = model.module if hasattr(model, "module") else model

    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()

    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))

        if param.ndim == 1:
            adamw_params.append(param)
            continue
        if _is_embedding_or_unembedding_param(name):
            adamw_params.append(param)
            continue
        if param.ndim == 2:
            muon_params.append(param)
            continue

        # Muon is intended for hidden-layer matrices; route everything else to AdamW.
        adamw_params.append(param)

    if not muon_params:
        raise ValueError(
            "No eligible matrix parameters found for Muon (expected 2D hidden-layer weights)."
        )

    logger.info(
        "Muon grouping: "
        f"muon_params={len(muon_params)} tensors, "
        f"adamw_params={len(adamw_params)} tensors"
    )

    return build_optimizer(
        muon_params=muon_params,
        adamw_params=adamw_params,
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
    )


polar_express_coeffs = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]
# safety factor for numerical stability (but exclude last polynomial)
polar_express_coeffs = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in polar_express_coeffs[:-1]
] + [polar_express_coeffs[-1]]


@torch.compile(dynamic=False, fullgraph=True)
def _polar_express_paper(G: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()  # for speed
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + epsilon)
    if G.size(-2) > G.size(-1):  # Tall matrix
        for a, b, c in polar_express_coeffs:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:  # Wide matrix (original math)
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
    use_normuon: bool = False,
):
    """Build a single Dion Muon/NorMuon optimizer that handles both muon and adamw param groups."""
    if not muon_params:
        raise ValueError("Muon optimizer requires at least one matrix parameter.")
    if not adamw_params:
        raise ValueError("Muon optimizer requires at least one AdamW parameter.")

    distributed_mesh = None
    if dist.is_available() and dist.is_initialized():
        distributed_mesh = dist.group.WORLD

    ns_func = _polar_express_paper if muon_use_polar_express else None

    param_groups = [
        dict(params=muon_params),
        dict(
            params=adamw_params,
            algorithm="adamw",
            lr=float(adamw_learning_rate),
            weight_decay=float(adamw_weight_decay),
            beta1=adamw_betas[0],
            beta2=adamw_betas[1],
            epsilon=float(adamw_epsilon),
        ),
    ]

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

    return Cls(param_groups, **kwargs)


def _is_embedding_or_unembedding_param(name: str) -> bool:
    lname = name.lower()

    # HF ModernBERT naming:
    # - token embedding matrix: model.embeddings.tok_embeddings.weight
    # - MLM output head: decoder.weight / decoder.bias
    #   (decoder.weight is tied to token embeddings by default and may not appear
    #   as a distinct named parameter).
    if "embeddings.tok_embeddings" in lname:
        return True
    if lname.endswith("decoder.weight") or lname.endswith("decoder.bias"):
        return True

    # Fallbacks for other architectures.
    return (
        "embedding" in lname
        or "lm_head" in lname
        or "unembedding" in lname
    )
