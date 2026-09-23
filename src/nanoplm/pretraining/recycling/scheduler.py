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
    curriculum_complete :
        When True, the curriculum is treated as already finished and
        `current_mean` pins the loop-count mean at `target` from step 0.
        Used by `decay` resumes that anneal a trunk which already ran the
        full curriculum.
    sampling :
        Overrides `recycling_loop_count_sampling`. Loopie stochastic depth
        passes "uniform"; Huginn callers pass nothing and read the config.
    jitter :
        Half-width of the "uniform" support, {target - jitter, ..., target + jitter}.
    """

    def __init__(self, pretraining_config, target_loop_count: int,
                 total_training_steps: int, *, curriculum_complete: bool = False,
                 sampling: Optional[str] = None, jitter: int = 0):
        self.sampling = sampling or pretraining_config.recycling_loop_count_sampling
        self.sigma = pretraining_config.recycling_loop_count_sigma
        self.curriculum = pretraining_config.recycling_loop_count_curriculum
        self.target = max(1, int(target_loop_count))
        self.curriculum_complete = bool(curriculum_complete)
        self.jitter = int(jitter)

        cs = pretraining_config.recycling_loop_count_curriculum_steps
        if cs == -1:
            self.curriculum_steps = max(1, int(0.25 * total_training_steps))
        else:
            self.curriculum_steps = max(1, int(cs))

        if self.sampling not in {"fixed", "lognormal_poisson", "uniform"}:
            raise ValueError(
                f"loop count sampling must be 'fixed', 'lognormal_poisson' or "
                f"'uniform', got {self.sampling!r}"
            )
        # jitter <= target - 1 keeps the support inside [1, ...], so it never
        # clips at 1 and drags the mean above the target.
        if self.sampling == "uniform" and not 1 <= self.jitter <= self.target - 1:
            raise ValueError(
                f"uniform sampling needs 1 <= jitter <= target - 1 "
                f"(target={self.target}), got jitter={self.jitter}"
            )
        if self.curriculum not in {"none", "linear", "1_sqrt"}:
            raise ValueError(
                f"recycling_loop_count_curriculum must be 'none', 'linear', or '1_sqrt', "
                f"got {self.curriculum!r}"
            )

    def current_mean(self, step: int) -> int:
        """Curriculum-scaled target mean k_t at the given step."""
        if self.curriculum_complete or self.curriculum == "none":
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
        """Sample r ~ Poisson(exp(N(log(r_bar) - 0.5 σ², σ))) + 1, or r ~ U{k ± jitter}."""
        k_t = self.current_mean(step)
        if self.sampling == "fixed":
            return k_t
        if self.sampling == "uniform":
            return int(torch.randint(
                k_t - self.jitter, k_t + self.jitter + 1, (1,),
                device=device, generator=generator,
            ).item())
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
