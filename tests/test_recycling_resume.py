from nanoplm.pretraining.recycling.scheduler import RecyclingScheduler


class _Cfg:
    recycling_loop_count_sampling = "fixed"
    recycling_loop_count_sigma = 0.5
    recycling_loop_count_curriculum = "1_sqrt"
    recycling_loop_count_curriculum_steps = -1


def test_curriculum_complete_returns_target_from_step_zero():
    # curriculum_complete=True must pin the mean at target for every step.
    sched = RecyclingScheduler(_Cfg(), target_loop_count=8, total_training_steps=1000,
                               curriculum_complete=True)
    assert sched.current_mean(0) == 8
    assert sched.current_mean(500) == 8


def test_default_curriculum_still_ramps():
    sched = RecyclingScheduler(_Cfg(), target_loop_count=8, total_training_steps=1000)
    assert sched.current_mean(0) < 8   # still ramping early
