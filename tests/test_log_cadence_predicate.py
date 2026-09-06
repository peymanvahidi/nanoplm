from nanoplm.pretraining.pure_pipeline import _should_log_step


def test_logs_the_first_step_of_a_fresh_run():
    assert _should_log_step(1, 25, 0)
    assert not _should_log_step(2, 25, 0)
    assert not _should_log_step(24, 25, 0)
    assert _should_log_step(25, 25, 0)


def test_logs_the_first_step_of_a_resumed_run():
    assert _should_log_step(52501, 25, 52500)
    assert not _should_log_step(52502, 25, 52500)
    assert _should_log_step(52525, 25, 52500)


def test_cadence_of_one_logs_every_step():
    assert all(_should_log_step(s, 1, 0) for s in range(1, 10))


def test_the_first_step_is_the_only_extra_line_on_a_long_run():
    logged = [s for s in range(1, 1001) if _should_log_step(s, 25, 0)]
    assert len(logged) == 41
    assert logged[:3] == [1, 25, 50]
