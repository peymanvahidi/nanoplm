"""Tests for RecyclingScheduler (log-normal Poisson sampler + curriculum)."""

import math
import statistics
import pytest
import torch

from nanoplm.pretraining.recycling.scheduler import RecyclingScheduler


class _MockPretrainCfg:
    def __init__(self, **kwargs):
        defaults = dict(
            recycling_loop_count_sampling="lognormal_poisson",
            recycling_loop_count_sigma=0.5,
            recycling_loop_count_curriculum="1_sqrt",
            recycling_loop_count_curriculum_steps=-1,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


def test_curriculum_none_yields_target_mean_immediately():
    sched = RecyclingScheduler(
        _MockPretrainCfg(recycling_loop_count_curriculum="none"),
        target_loop_count=8, total_training_steps=1000,
    )
    assert sched.current_mean(0) == 8
    assert sched.current_mean(500) == 8


def test_curriculum_linear_boundary_conditions():
    sched = RecyclingScheduler(
        _MockPretrainCfg(recycling_loop_count_curriculum="linear",
                         recycling_loop_count_curriculum_steps=100),
        target_loop_count=8, total_training_steps=1000,
    )
    assert sched.current_mean(0) == 1  # max(1, ceil(0))
    assert sched.current_mean(100) == 8  # at end of curriculum
    assert sched.current_mean(500) == 8  # post-curriculum
    prev = 0
    for t in range(0, 100, 5):
        m = sched.current_mean(t)
        assert m >= prev
        prev = m


def test_curriculum_1_sqrt_boundary_conditions():
    sched = RecyclingScheduler(
        _MockPretrainCfg(recycling_loop_count_curriculum="1_sqrt",
                         recycling_loop_count_curriculum_steps=100),
        target_loop_count=8, total_training_steps=1000,
    )
    assert sched.current_mean(0) == 1
    assert sched.current_mean(100) == 8
    assert sched.current_mean(25) >= 2


def test_curriculum_auto_resolves_to_25pct_of_total():
    sched = RecyclingScheduler(
        _MockPretrainCfg(recycling_loop_count_curriculum_steps=-1),
        target_loop_count=8, total_training_steps=1000,
    )
    assert sched.curriculum_steps == 250


def test_fixed_sampling_always_returns_target():
    sched = RecyclingScheduler(
        _MockPretrainCfg(recycling_loop_count_sampling="fixed",
                         recycling_loop_count_curriculum="none"),
        target_loop_count=8, total_training_steps=1000,
    )
    g = torch.Generator().manual_seed(7)
    for t in range(0, 1000, 100):
        assert sched.sample(t, device=torch.device("cpu"), generator=g) == 8


def test_lognormal_poisson_sampler_matches_huginn_statistics():
    sched = RecyclingScheduler(
        _MockPretrainCfg(
            recycling_loop_count_sampling="lognormal_poisson",
            recycling_loop_count_sigma=0.5,
            recycling_loop_count_curriculum="none",
        ),
        target_loop_count=8, total_training_steps=1000,
    )
    g = torch.Generator().manual_seed(42)
    samples = [sched.sample(500, device=torch.device("cpu"), generator=g) for _ in range(10000)]
    sample_mean = statistics.mean(samples)
    assert 7.2 <= sample_mean <= 8.8, f"got mean {sample_mean}, expected ~8"
    assert min(samples) >= 1


def test_sample_with_curriculum_uses_current_mean():
    sched = RecyclingScheduler(
        _MockPretrainCfg(
            recycling_loop_count_sampling="lognormal_poisson",
            recycling_loop_count_curriculum="linear",
            recycling_loop_count_curriculum_steps=1000,
            recycling_loop_count_sigma=0.1,
        ),
        target_loop_count=20, total_training_steps=5000,
    )
    g = torch.Generator().manual_seed(0)
    early_samples = [sched.sample(0, device=torch.device("cpu"), generator=g) for _ in range(200)]
    assert statistics.mean(early_samples) < 3
    late_samples = [sched.sample(5000, device=torch.device("cpu"), generator=g) for _ in range(200)]
    assert statistics.mean(late_samples) > 15


def _uniform_sched(target, jitter):
    """Loopie's construction: bounded uniform, no curriculum."""
    return RecyclingScheduler(
        _MockPretrainCfg(), target_loop_count=target, total_training_steps=1000,
        curriculum_complete=True, sampling="uniform", jitter=jitter,
    )


def test_uniform_mode_is_opt_in_and_leaves_huginn_callers_alone():
    sched = RecyclingScheduler(
        _MockPretrainCfg(), target_loop_count=8, total_training_steps=1000,
    )
    assert sched.sampling == "lognormal_poisson"
    assert sched.jitter == 0


@pytest.mark.parametrize("target,jitter", [(2, 1), (4, 2), (8, 7)])
def test_uniform_sampling_covers_its_support_with_the_target_as_mean(target, jitter):
    sched = _uniform_sched(target, jitter)
    g = torch.Generator().manual_seed(11)
    samples = [sched.sample(0, device=torch.device("cpu"), generator=g)
               for _ in range(20000)]
    assert set(samples) == set(range(target - jitter, target + jitter + 1))
    assert abs(statistics.mean(samples) - target) < 0.1 * (jitter + 1)


@pytest.mark.parametrize("jitter", [0, 4, 5])
def test_uniform_sampling_rejects_a_support_that_would_clip_at_one(jitter):
    # jitter <= target - 1 is what makes the mean exactly the target; jitter 0
    # is fixed depth, which is the "fixed" mode, not this one.
    with pytest.raises(ValueError, match="jitter"):
        _uniform_sched(target=4, jitter=jitter)
