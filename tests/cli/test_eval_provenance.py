"""Covers the eval provenance sidecar: where it lands, what it records about the
checkpoint that produced a report, and that writing it needs neither a GPU nor
biotrainer.
"""

import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from nanoplm.eval.checkpoint import EvalProvenance, write_eval_provenance

RUN_NAME = "loop-imp-abl-baseline-l20-1564242-3"
RUN_SHA = "9f3c1a2b4d5e6f708192a3b4c5d6e7f809111213"

RUN_MANIFEST = {
    "run_name": RUN_NAME,
    "wandb_run_name": f"{RUN_NAME}-wandb",
    "is_resume": False,
    "resume_step": None,
    "slurm_job_id": "1564242",
    "world_size": 32,
    "git_sha": RUN_SHA,
    "argv": ["nanoplm", "pretrain", "from-yaml", "configs/loop-imp-abl.yaml"],
    "dataset_dir": "/data/uniref50",
    "ckp_dir": "/checkpoints",
}


def _make_checkpoint(
    tmp_path: Path,
    *,
    run_name: str = RUN_NAME,
    checkpoint_name: str = "checkpoint-27500",
    global_step: int | None = 27500,
    run_manifest: dict | None = None,
) -> Path:
    checkpoint_dir = tmp_path / run_name / checkpoint_name
    checkpoint_dir.mkdir(parents=True)
    if global_step is not None:
        (checkpoint_dir / "training_state.json").write_text(
            json.dumps({"global_step": global_step}), encoding="utf-8"
        )
    if run_manifest is not None:
        (checkpoint_dir.parent / "run.json").write_text(
            json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
        )
    return checkpoint_dir


def test_sidecar_sits_beside_the_report_and_shares_its_stem(tmp_path):
    checkpoint_dir = _make_checkpoint(tmp_path, run_manifest=RUN_MANIFEST)
    provenance = EvalProvenance.resolve(
        checkpoint_dir=checkpoint_dir, embedder_name="my-embedder", loop_count=4
    )

    report_path = tmp_path / "out" / "autoeval_report_my-embedder.json"
    sidecar = write_eval_provenance(provenance, report_path)

    assert sidecar == tmp_path / "out" / "autoeval_report_my-embedder.provenance.json"
    assert sidecar.exists()


def test_sidecar_records_checkpoint_step_run_and_loop_count(tmp_path):
    checkpoint_dir = _make_checkpoint(tmp_path, run_manifest=RUN_MANIFEST)
    provenance = EvalProvenance.resolve(
        checkpoint_dir=checkpoint_dir, embedder_name="loop-l20-27500", loop_count=4
    )

    sidecar = write_eval_provenance(
        provenance, tmp_path / "out" / "autoeval_report_loop-l20-27500.json"
    )
    record = json.loads(sidecar.read_text(encoding="utf-8"))

    assert record["checkpoint_dir"] == str(checkpoint_dir.resolve())
    assert record["checkpoint_step"] == 27500
    assert record["run_name"] == RUN_NAME
    assert record["embedder_name"] == "loop-l20-27500"
    assert record["loop_count"] == 4
    assert record["training_run"]["git_sha"] == RUN_SHA


def test_missing_run_manifest_leaves_training_run_null(tmp_path):
    checkpoint_dir = _make_checkpoint(tmp_path, run_manifest=None)
    provenance = EvalProvenance.resolve(
        checkpoint_dir=checkpoint_dir, embedder_name="no-manifest", loop_count=None
    )

    sidecar = write_eval_provenance(
        provenance, tmp_path / "out" / "autoeval_report_no-manifest.json"
    )
    record = json.loads(sidecar.read_text(encoding="utf-8"))

    assert record["training_run"] is None
    assert record["loop_count"] is None
    assert record["checkpoint_step"] == 27500


def test_role_named_checkpoint_takes_its_step_from_training_state(tmp_path):
    checkpoint_dir = _make_checkpoint(
        tmp_path,
        checkpoint_name="checkpoint-final",
        global_step=30000,
        run_manifest=RUN_MANIFEST,
    )
    provenance = EvalProvenance.resolve(
        checkpoint_dir=checkpoint_dir, embedder_name="final", loop_count=8
    )

    assert provenance.checkpoint_step == 30000


def test_eval_started_at_is_a_timezone_aware_iso_timestamp(tmp_path):
    checkpoint_dir = _make_checkpoint(tmp_path, run_manifest=RUN_MANIFEST)
    provenance = EvalProvenance.resolve(
        checkpoint_dir=checkpoint_dir, embedder_name="stamped", loop_count=None
    )

    stamped = datetime.fromisoformat(provenance.eval_started_at)

    assert stamped.tzinfo is not None
    assert stamped.utcoffset() is not None


def test_eval_modules_import_without_biotrainer():
    blocked = ("biotrainer", "biotrainer_core")
    nanoplm_eval = "nanoplm.eval"

    class _Blocker:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in blocked:
                raise ImportError(f"blocked for this test: {fullname}")
            return None

    def _purge():
        for name in list(sys.modules):
            root = name.split(".")[0]
            if root in blocked or name == nanoplm_eval or name.startswith(
                nanoplm_eval + "."
            ):
                sys.modules.pop(name, None)

    blocker = _Blocker()
    saved_meta_path = list(sys.meta_path)
    sys.meta_path.insert(0, blocker)
    try:
        _purge()
        with pytest.raises(ImportError):
            importlib.import_module("biotrainer")
        assert importlib.import_module("nanoplm.eval.checkpoint") is not None
        assert importlib.import_module("nanoplm.eval.pipeline") is not None
    finally:
        sys.meta_path[:] = saved_meta_path
        _purge()
        importlib.import_module("nanoplm.eval.checkpoint")
        importlib.import_module("nanoplm.eval.pipeline")
