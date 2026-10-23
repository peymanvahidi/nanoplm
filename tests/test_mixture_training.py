"""Training on a mixture: eval split and keys, and a CPU smoke run."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nanoplm.data.infra import mixture as mx
from nanoplm.data.infra.manifest import read_manifest
from nanoplm.pretraining import pure_pipeline as pp
from nanoplm.pretraining.config import PretrainingConfig

from tests.data.datasets.mixture_sources import cluster_source, plain_source

HARNESS = Path(__file__).parent / "mixture_smoke_harness.py"


def test_eval_keeps_the_batch_count_and_splits_it_evenly():
    # 30 + 9 + 400 val seqs, micro batch 8: one loader would run 55 batches
    sizes = {"a": 30, "b": 9, "c": 400}
    got = mx.eval_split(sizes, ["a", "b", "c"], 8)
    assert got == {"a": 30, "b": 9, "c": 152}           # 19 batches each, a and b run short
    assert mx.eval_split(sizes, ["c"], 8) == {"c": 400}  # the whole budget to c
    assert mx.eval_split({"a": 100, "b": 100}, ["a", "b"], 10) == {"a": 100, "b": 100}


def test_eval_payload_is_the_share_weighted_mean():
    p = mx.eval_payload({"a": 2.0, "b": 4.0}, {"a": 0.25, "b": 0.75})
    assert p == {"eval/loss/a": 2.0, "eval/loss/b": 4.0, "eval/loss": 3.5}
    # evaluating only b: its weight alone
    assert mx.eval_payload({"b": 4.0}, {"a": 0.25, "b": 0.75})["eval/loss"] == 4.0
    assert mx.eval_payload({"a": 1.0}, {"a": 1.0}, suffix="/r4") == {"eval/loss/a/r4": 1.0}


def _mixture(root):
    a = plain_source(root / "a", 50, tag=5)
    c, _ = cluster_source(root / "c", seed=2)
    mx.write_mixture(root, [("a", a, 0.4, "a.fasta"), ("c", c, 0.6, "c.fasta")])
    return read_manifest(root)


def test_a_single_dataset_keeps_todays_eval(tmp_path):
    a = plain_source(tmp_path / "a", 50, tag=5)
    m = read_manifest(a)
    cfg = PretrainingConfig(dataset_dir=str(a))
    assert pp.MixturePlan.open(a, m, cfg) is None
    cfg.eval_sources = ["a"]
    with pytest.raises(mx.MixtureError, match="not a mixture; set eval_sources: all"):
        pp.MixturePlan.open(a, m, cfg)


class _Wandb:
    def __init__(self):
        self.run, self.logged = object(), []

    def log(self, payload):
        self.logged.append(payload)


@pytest.fixture
def plan(tmp_path, monkeypatch):
    m = _mixture(tmp_path)
    plan = pp.MixturePlan.open(tmp_path, m, PretrainingConfig(dataset_dir=str(tmp_path),
                                                              micro_batch_seqs=4))
    plan.eval_loaders(lambda ds: ds)
    fake = _Wandb()
    monkeypatch.setattr(pp, "wandb", fake)
    losses = {"a": 2.0, "c": 3.0}
    calls = []

    def evaluate(model, loader, device, distributed, amp_dtype, loop_count=None,
                 recycling_bptt=None):
        name = next(n for n, d in plan.loaders.items() if d is loader)
        calls.append((name, loop_count))
        return losses[name] + (loop_count or 0)

    monkeypatch.setattr(pp, "_evaluate", evaluate)
    return plan, fake, calls


def test_mixture_eval_logs_each_source_and_the_weighted_mean(plan):
    plan, fake, calls = plan
    assert plan.eval_names == ["a", "c"]
    assert [len(d) for d in plan.eval_sets.values()] == [8, 7]   # 4 batches of 4, 2 each
    assert pp._mixture_eval(plan, [None], None, None, "cpu", False, None, None, 10, True, True)
    assert calls == [("a", None), ("c", None)]
    assert fake.logged == [{"train/global_step": 10, "eval/loss/a": 2.0, "eval/loss/c": 3.0,
                            "eval/loss": pytest.approx(0.4 * 2 + 0.6 * 3)}]


def test_mixture_eval_with_recycling_logs_per_source_and_r(plan):
    plan, fake, _ = plan
    pp._mixture_eval(plan, [1, 4], 4, None, "cpu", False, None, 2, 10, True, True)
    p = fake.logged[0]
    # the unsuffixed keys are the target r (4), every r adds eval/loss/<source>/r<r>
    assert p["eval/loss/a"] == 6.0 and p["eval/loss/c"] == 7.0
    assert p["eval/loss"] == pytest.approx(0.4 * 6 + 0.6 * 7)
    assert p["eval/loss/a/r1"] == 3.0 and p["eval/loss/c/r4"] == 7.0
    assert p["recycling/eval_inloop/inloop_eval_loss_at_r1"] == pytest.approx(0.4 * 3 + 0.6 * 4)


def test_mixture_training_smoke_on_cpu(tmp_path):
    """A 12-step packed run on a plain + cluster mixture (see mixture_smoke_harness.py)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", WANDB_MODE="offline",
               WANDB_DIR=str(tmp_path))
    r = subprocess.run([sys.executable, str(HARNESS), str(tmp_path / "d")], cwd=tmp_path,
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res = json.loads((tmp_path / "d/result.json").read_text())
    log = res["log"]
    head = log.index(next(m for m in log if m.startswith("Data mixture over a budget")))
    assert log[head + 1].split() == ["source", "share", "size", "tokens", "used", "epochs"]
    assert log[head + 2].split()[:2] == ["a", "0.399"]
    assert log[head + 3].split()[:4] == ["c", "0.601", "246", "clusters"]
    assert any(m.startswith("Mixture eval: a 8 of 9 val seqs, c 7 of 7") for m in log)
    assert res["eval_keys"] and all(
        keys == ["eval/loss", "eval/loss/a", "eval/loss/c", "train/global_step"]
        for keys in res["eval_keys"])
