"""Per-step loop count sampler + curriculum for Huginn-style recycling.

See docs/superpowers/specs/2026-05-24-huggin-style-recycling-design.md §4.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


class RecyclingScheduler:
    """Samples per-step loop count r with a curriculum on the mean.

    Parameters
    ----------
    pretraining_config :
        Object exposing recycling_loop_count_sampling, _sigma, _curriculum,
        and _curriculum_steps attributes.
    target_loop_count :
        The model's recycling_loop_count (target mean for sampling).
    total_training_steps :
        Used to resolve `curriculum_steps == -1` to 25% of total.
    """

    def __init__(self, pretraining_config, target_loop_count: int, total_training_steps: int):
        self.sampling = pretraining_config.recycling_loop_count_sampling
        self.sigma = pretraining_config.recycling_loop_count_sigma
        self.curriculum = pretraining_config.recycling_loop_count_curriculum
        self.target = max(1, int(target_loop_count))

        cs = pretraining_config.recycling_loop_count_curriculum_steps
        if cs == -1:
            self.curriculum_steps = max(1, int(0.25 * total_training_steps))
        else:
            self.curriculum_steps = max(1, int(cs))

        if self.sampling not in {"fixed", "lognormal_poisson"}:
            raise ValueError(
                f"recycling_loop_count_sampling must be 'fixed' or 'lognormal_poisson', "
                f"got {self.sampling!r}"
            )
        if self.curriculum not in {"none", "linear", "1_sqrt"}:
            raise ValueError(
                f"recycling_loop_count_curriculum must be 'none', 'linear', or '1_sqrt', "
                f"got {self.curriculum!r}"
            )

    def current_mean(self, step: int) -> int:
        """Curriculum-scaled target mean k_t at the given step."""
        if self.curriculum == "none":
            return self.target
        T_c = self.curriculum_steps
        if step >= T_c:
            return self.target
        if self.curriculum == "linear":
            k = math.ceil(self.target * step / T_c)
        else:  # "1_sqrt"
            k = math.ceil(self.target * (1.0 - math.sqrt(1.0 - step / T_c)))
        return max(1, int(k))

    def sample(self, step: int, device: torch.device, generator: torch.Generator) -> int:
        """Sample r ~ Poisson(exp(N(log(r_bar) - 0.5 σ², σ))) + 1."""
        k_t = self.current_mean(step)
        if self.sampling == "fixed":
            return k_t
        r_bar = max(1, k_t - 1)
        sigma = self.sigma
        log_lambda = (
            torch.randn(1, device=device, generator=generator).item() * sigma
            + math.log(r_bar)
            - 0.5 * sigma * sigma
        )
        lam = math.exp(log_lambda)
        r_minus_1 = int(torch.poisson(
            torch.tensor([lam], device=device), generator=generator
        ).item())
        return r_minus_1 + 1
