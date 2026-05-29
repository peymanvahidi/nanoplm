"""NOBLE: Nonlinear lOw-rank Branch for Linear Enhancement.

Paper: "NOBLE: Accelerating Transformers with Nonlinear Low-Rank Branches"
(Smith, 2026, arXiv:2603.06492)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModernBertConfig, _NOBLE_TARGET_ROLES


class CosNet(nn.Module):
    """Two-layer cosine nonlinearity with learnable frequency, phase, and mixing.

    CosNet(h) = cos(ω₂ ⊙ (M · cos(ω₁ ⊙ h + φ₁)) + φ₂)

    where h ∈ ℝʳ is the bottleneck representation, M ∈ ℝʳˣʳ is a learned
    mixing matrix, and each dimension i has learnable ωᵢ and φᵢ parameters.
    """

    def __init__(
        self,
        rank: int,
        omega_range: tuple[float, float] = (0.8, 1.2),
        phi_std: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        # First cosine layer: learnable frequency and phase
        self.omega1 = nn.Parameter(torch.empty(rank))
        self.phi1 = nn.Parameter(torch.empty(rank))
        # Mixing matrix (r × r)
        self.M = nn.Parameter(torch.empty(rank, rank))
        # Second cosine layer: learnable frequency and phase
        self.omega2 = nn.Parameter(torch.empty(rank))
        self.phi2 = nn.Parameter(torch.empty(rank))

        # --- Initialization (paper Appendix A) ---
        # Frequencies: Uniform[ω_min, ω_max]
        nn.init.uniform_(self.omega1, omega_range[0], omega_range[1])
        nn.init.uniform_(self.omega2, omega_range[0], omega_range[1])
        # Phases: Normal(0, σ_φ)
        nn.init.normal_(self.phi1, mean=0.0, std=phi_std)
        nn.init.normal_(self.phi2, mean=0.0, std=phi_std)
        # Mixing matrix: Xavier uniform
        nn.init.xavier_uniform_(self.M)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (..., rank)
        h = torch.cos(self.omega1 * h + self.phi1)   # eq (5)
        h = F.linear(h, self.M)                       # eq (6): M @ h (F.linear computes h @ M^T)
        h = torch.cos(self.omega2 * h + self.phi2)   # eq (7)
        return h


class NOBLELinear(nn.Module):
    """Drop-in replacement for nn.Linear with a nonlinear low-rank branch.

    f_NOBLE(x) = xW + b + CosNet(x @ W_down^T) @ W_up^T

    The branch starts near-silent (W_up ≈ 0) and gradually learns
    complementary features that the main linear pathway cannot represent.

    Exposes .weight and .bias properties so existing init code that does
    ``nn.init.uniform_(module.weight, ...)`` works unchanged.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = False,
        alpha: float = 0.01,
        omega_range: tuple[float, float] = (0.8, 1.2),
        phi_std: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Main linear pathway
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Low-rank nonlinear branch
        self.W_down = nn.Linear(in_features, rank, bias=False)
        self.cosnet = CosNet(rank, omega_range=omega_range, phi_std=phi_std)
        self.noble_w_up = nn.Linear(rank, out_features, bias=False)

        # noble_w_up: near-zero init
        nn.init.normal_(self.noble_w_up.weight, mean=0.0, std=alpha / math.sqrt(rank))

    @property
    def weight(self) -> nn.Parameter:
        """Delegate to main linear weight for compatibility with init_weights()."""
        return self.linear.weight

    @property
    def bias(self) -> Optional[nn.Parameter]:
        """Delegate to main linear bias for compatibility with init_weights()."""
        return self.linear.bias

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.linear.bias is not None}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.linear(x)
        branch = self.noble_w_up(self.cosnet(self.W_down(x)))
        return main + branch


def _build_linear(
    in_features: int,
    out_features: int,
    config: ModernBertConfig,
    bias: bool = False,
    role: str = "attn_qkv",
) -> nn.Module:
    """Factory: returns NOBLELinear if NOBLE is enabled for this role, else plain nn.Linear."""
    if config.use_noble:
        targets = _NOBLE_TARGET_ROLES.get(config.noble_targets, _NOBLE_TARGET_ROLES["all"])
        if role in targets:
            return NOBLELinear(
                in_features,
                out_features,
                rank=config.noble_rank,
                bias=bias,
                alpha=config.noble_alpha,
                omega_range=config.noble_omega_range,
                phi_std=config.noble_phi_std,
            )
    return nn.Linear(in_features, out_features, bias=bias)
