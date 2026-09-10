import os

import pytest

from nanoplm.pretraining.utils import get_num_workers


def _fake_affinity(monkeypatch, count: int) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(count)))


def test_auto_does_not_grow_with_world_size(monkeypatch):
    _fake_affinity(monkeypatch, 72)
    values = set()
    for world_size in (1, 4, 16, 64):
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
        monkeypatch.setenv("SLURM_NTASKS", str(world_size))
        values.add(get_num_workers("auto"))
    assert values == {4}


def test_auto_rejects_a_world_size_argument():
    with pytest.raises(TypeError):
        get_num_workers("auto", 16)


def test_auto_reads_task_affinity_not_node_cpu_count(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 288)
    _fake_affinity(monkeypatch, 2)
    assert get_num_workers("auto") == 1


def test_auto_shrinks_to_the_cpu_budget(monkeypatch):
    _fake_affinity(monkeypatch, 4)
    assert get_num_workers("auto") == 3


def test_explicit_value_wins_over_auto(monkeypatch):
    _fake_affinity(monkeypatch, 72)
    assert get_num_workers(0) == 0
    assert get_num_workers(16) == 16
    assert get_num_workers("16") == 16
