"""Optimizer utilities shared by pretraining pipelines."""

from __future__ import annotations

import torch.distributed as dist
from dion import Muon as DionMuon
from dion import NorMuon as DionNorMuon


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


import torch

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
    distributed_mesh=None,
):
    """Build a single Dion Muon/NorMuon optimizer that handles both muon and adamw param groups."""
    if not muon_params:
        raise ValueError("Muon optimizer requires at least one matrix parameter.")
    if not adamw_params:
        raise ValueError("Muon optimizer requires at least one AdamW parameter.")

    if distributed_mesh is None:
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
