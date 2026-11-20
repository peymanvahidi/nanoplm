"""The cluster member draw in the start-up log, in training_state, and in the resume
warning when the dataset fingerprint changed."""

import json
from pathlib import Path

import torch

from nanoplm.data.datasets import open_train_dataset
from nanoplm.data.infra import mixture as mx
from nanoplm.data.infra.manifest import read_manifest
from nanoplm.pretraining.pure_pipeline import (
    _dataset_changed_warning,
    _draw_text,
    _save_checkpoint,
    _train_cluster_draw,
)
from tests.data.datasets.mixture_sources import cluster_source, plain_source

BASE = ("Dataset changed since checkpoint (fingerprint aaa → bbb). "
        "Dataloader will start from batch 0 instead of fast-forwarding.")


def test_uniform_to_two_level():
    assert _dataset_changed_warning("aaa", "bbb", "uniform", "two_level") == (
        BASE + " The cluster member draw changed from uniform to two-level.")


def test_two_level_to_uniform():
    assert _dataset_changed_warning("aaa", "bbb", "two_level", "uniform") == (
        BASE + " The cluster member draw changed from two-level to uniform.")


def test_same_draw_other_fingerprint():
    assert _dataset_changed_warning("aaa", "bbb", "two_level", "two_level") == BASE
    assert _dataset_changed_warning("aaa", "bbb", None, None) == BASE   # plain dataset


def test_old_checkpoint_without_the_key_counts_as_uniform():
    assert _dataset_changed_warning("aaa", "bbb", None, "uniform") == BASE
    assert _dataset_changed_warning("aaa", "bbb", None, "two_level") == (
        BASE + " The cluster member draw changed from uniform to two-level.")


def test_mixture_names_the_source():
    got = _dataset_changed_warning("aaa", "bbb", None, {"u50": "two_level", "u90": "uniform"})
    assert got == BASE + " The cluster member draw of u50 changed from uniform to two-level."
    assert _dataset_changed_warning("aaa", "bbb", {"u50": "two_level"},
                                    {"u50": "two_level"}) == BASE


def test_source_missing_from_checkpoint_dict_says_nothing():
    assert _dataset_changed_warning("aaa", "bbb", {"u50": "two_level"},
                                    {"uniref50": "two_level"}) == BASE
    got = _dataset_changed_warning("aaa", "bbb", {"u50": "uniform", "x": "uniform"},
                                   {"u50": "two_level", "new": "two_level"})
    assert got == BASE + " The cluster member draw of u50 changed from uniform to two-level."


def test_switch_between_single_source_and_mixture_says_nothing():
    assert _dataset_changed_warning("aaa", "bbb", {"c": "two_level"}, "two_level") == BASE
    assert _dataset_changed_warning("aaa", "bbb", {"c": "uniform"}, "two_level") == BASE
    assert _dataset_changed_warning("aaa", "bbb", "uniform", {"c": "two_level"}) == BASE


def test_no_cluster_draw_for_plain_data(tmp_path):
    a = plain_source(tmp_path / "a", 50, tag=5, seed=1)
    b = plain_source(tmp_path / "b", 50, tag=6, seed=2)
    assert _train_cluster_draw(open_train_dataset(a, read_manifest(a), seed=0)) is None
    mx.write_mixture(tmp_path / "m", [("a", a, 0.5, ""), ("b", b, 0.5, "")])
    mix = open_train_dataset(tmp_path / "m", read_manifest(tmp_path / "m"), seed=0)
    assert _train_cluster_draw(mix) is None


def test_start_up_description(tmp_path):
    assert _draw_text("two_level") == "two-level (UniRef90 sub-cluster, then member)"
    assert _draw_text("uniform") == "uniform"
    a = plain_source(tmp_path / "a", 50, tag=5, seed=1)
    c, _ = cluster_source(tmp_path / "c", seed=3, n_small=60, two_level=True)
    u, _ = cluster_source(tmp_path / "u", seed=4, n_small=60)
    assert _train_cluster_draw(open_train_dataset(c, read_manifest(c), seed=0)) == "two_level"
    assert _train_cluster_draw(open_train_dataset(u, read_manifest(u), seed=0)) == "uniform"
    assert _train_cluster_draw(open_train_dataset(a, read_manifest(a), seed=0)) is None
    mx.write_mixture(tmp_path / "m", [("a", a, 0.4, ""), ("c", c, 0.3, ""), ("u", u, 0.3, "")])
    mix = open_train_dataset(tmp_path / "m", read_manifest(tmp_path / "m"), seed=0)
    assert _train_cluster_draw(mix) == {"c": "two_level", "u": "uniform"}


def test_training_state_saves_the_draw_only_when_given(tmp_path):
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    common = dict(epoch=0, output_dir=str(tmp_path), logging_steps=1, eval_steps=1,
                  save_steps=1)
    _save_checkpoint(model, opt, sched, global_step=1, cluster_draw="two_level", **common)
    _save_checkpoint(model, opt, sched, global_step=2, **common)
    state = [json.loads((Path(tmp_path) / f"checkpoint-{s}" / "training_state.json").read_text())
             for s in (1, 2)]
    assert state[0]["cluster_draw"] == "two_level" and "cluster_draw" not in state[1]
