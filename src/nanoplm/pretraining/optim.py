"""Optimizer utilities shared by pretraining pipelines."""

from __future__ import annotations

import torch


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for matrix params, AdamW for the rest."""

    def __init__(
        self,
        muon_params: list[torch.nn.Parameter],
        adamw_params: list[torch.nn.Parameter],
        muon_learning_rate: float,
        muon_weight_decay: float,
        muon_momentum: float,
        muon_nesterov: bool,
        muon_eps: float,
        muon_ns_steps: int,
        adamw_learning_rate: float,
        adamw_weight_decay: float,
        adamw_betas: tuple[float, float],
        adamw_epsilon: float,
    ) -> None:
        if not muon_params:
            raise ValueError("Muon optimizer requires at least one matrix parameter.")
        if not adamw_params:
            raise ValueError("Muon optimizer requires at least one AdamW parameter.")

        all_params = list(muon_params) + list(adamw_params)
        super().__init__(all_params, defaults={})

        self.muon = torch.optim.Muon(
            muon_params,
            lr=float(muon_learning_rate),
            weight_decay=float(muon_weight_decay),
            momentum=float(muon_momentum),
            nesterov=bool(muon_nesterov),
            eps=float(muon_eps),
            ns_steps=int(muon_ns_steps),
        )
        self.adamw = torch.optim.AdamW(
            adamw_params,
            lr=float(adamw_learning_rate),
            betas=adamw_betas,
            eps=float(adamw_epsilon),
            weight_decay=float(adamw_weight_decay),
            fused=True
        )
        self.param_groups = self.muon.param_groups + self.adamw.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self.adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = self.muon.param_groups + self.adamw.param_groups
