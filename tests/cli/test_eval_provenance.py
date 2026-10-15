"""Eval provenance and cache reporting. Must stay runnable with no GPU and no
biotrainer, so use stubs rather than importing either.
"""

import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanoplm.eval.checkpoint import (
    EvalProvenance,
    default_embedder_name,
    write_eval_provenance,
)
from nanoplm.eval.pipeline import (
    FRAMEWORK_REPORT_NAMES,
    _completion_message,
    _refuse_results_from_another_biotrainer,
    _reused_frameworks,
)

LONG_DASH = "\u2014"
PGYM_REPORT_NAME = FRAMEWORK_REPORT_NAMES["pgym"]

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


def test_sidecar_records_the_biotrainer_version(tmp_path):
    checkpoint_dir = _make_checkpoint(tmp_path)
    provenance = EvalProvenance.resolve(
        checkpoint_dir=checkpoint_dir,
        embedder_name="versioned",
        loop_count=None,
        biotrainer_version="2.1.1",
    )

    sidecar = write_eval_provenance(
        provenance, tmp_path / "out" / "autoeval_report_versioned.json"
    )
    record = json.loads(sidecar.read_text(encoding="utf-8"))

    assert record["biotrainer_version"] == "2.1.1"


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


def test_unreadable_run_manifest_leaves_training_run_null(tmp_path):
    checkpoint_dir = _make_checkpoint(tmp_path, run_manifest=None)
    (checkpoint_dir.parent / "run.json").write_text("{ not json", encoding="utf-8")

    provenance = EvalProvenance.resolve(
        checkpoint_dir=checkpoint_dir, embedder_name="corrupt", loop_count=None
    )

    assert provenance.training_run is None


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


def test_default_name_for_a_numeric_checkpoint_is_run_dir_and_step(tmp_path):
    checkpoint_dir = _make_checkpoint(tmp_path, run_manifest=RUN_MANIFEST)

    assert default_embedder_name(checkpoint_dir) == f"{RUN_NAME}-ckpt27500"


def test_two_runs_at_the_same_step_get_different_default_names(tmp_path):
    first = _make_checkpoint(tmp_path, run_name=RUN_NAME)
    second = _make_checkpoint(tmp_path, run_name="loop-imp-abl-l40-1564243-1")
    assert first.name == second.name == "checkpoint-27500"

    first_name = default_embedder_name(first)
    second_name = default_embedder_name(second)

    assert first_name != second_name
    assert first_name != "checkpoint-27500"
    assert second_name != "checkpoint-27500"


def test_role_named_checkpoint_defaults_to_the_step_in_training_state(tmp_path):
    checkpoint_dir = _make_checkpoint(
        tmp_path, checkpoint_name="checkpoint-final", global_step=30000
    )

    assert default_embedder_name(checkpoint_dir) == f"{RUN_NAME}-ckpt30000"


def test_unresolvable_step_falls_back_to_the_checkpoint_role(tmp_path):
    checkpoint_dir = _make_checkpoint(
        tmp_path, checkpoint_name="checkpoint-final", global_step=None
    )

    assert default_embedder_name(checkpoint_dir) == f"{RUN_NAME}-final"


def test_unresolvable_step_is_null_in_the_sidecar(tmp_path):
    checkpoint_dir = _make_checkpoint(
        tmp_path, checkpoint_name="checkpoint-final", global_step=None
    )
    provenance = EvalProvenance.resolve(
        checkpoint_dir=checkpoint_dir, embedder_name=None, loop_count=None
    )

    report_name = f"autoeval_report_{provenance.embedder_name}.json"
    sidecar = write_eval_provenance(provenance, tmp_path / "out" / report_name)
    record = json.loads(sidecar.read_text(encoding="utf-8"))

    assert record["checkpoint_step"] is None


def test_explicit_embedder_name_wins_over_the_default(tmp_path):
    checkpoint_dir = _make_checkpoint(tmp_path)
    provenance = EvalProvenance.resolve(
        checkpoint_dir=checkpoint_dir, embedder_name="hand-picked", loop_count=None
    )

    assert provenance.embedder_name == "hand-picked"


class _FakeResult:
    def __init__(self, *, dev_mode: bool, task_filter_applied: bool = False):
        self._dev_mode = dev_mode
        self.task_filter_applied = task_filter_applied

    def used_development_mode(self) -> bool:
        return self._dev_mode


class _FakeReport:
    def __init__(self, results: dict):
        self._results = results

    def maybe_framework_result(self, framework_name: str):
        return self._results.get(framework_name)


def _fake_biotrainer(results: dict) -> dict:
    return {
        "AutoEvalReport": SimpleNamespace(
            from_json_file=lambda path: _FakeReport(results)
        )
    }


def test_completion_message_names_what_ran_when_nothing_was_reused():
    message = _completion_message(
        computed=["pgym", "pbc_zeroshot_contact"], reused=[], elapsed_s=1837.2
    )

    assert "AUTOEVAL COMPLETE" in message
    assert "COMPUTED NOTHING" not in message
    assert "pgym" in message
    assert "pbc_zeroshot_contact" in message
    assert LONG_DASH not in message


def test_completion_message_names_both_lists_when_some_were_reused():
    message = _completion_message(
        computed=["pbc_zeroshot_contact"], reused=["pgym"], elapsed_s=1837.2
    )

    assert "AUTOEVAL COMPLETE" in message
    assert "pbc_zeroshot_contact" in message
    assert "pgym" in message


def test_completion_message_refuses_to_claim_completion_when_all_were_reused():
    message = _completion_message(
        computed=[], reused=["pgym", "pbc_zeroshot_contact"], elapsed_s=0.04
    )

    assert "AUTOEVAL COMPUTED NOTHING" in message
    assert "AUTOEVAL COMPLETE" not in message
    assert "pgym" in message
    assert "pbc_zeroshot_contact" in message
    assert "output_dir" in message
    assert LONG_DASH not in message


@pytest.mark.parametrize(
    "stored, development_mode, expected",
    [
        (_FakeResult(dev_mode=False), False, ["pgym"]),
        (_FakeResult(dev_mode=False), True, ["pgym"]),
        (_FakeResult(dev_mode=True), True, ["pgym"]),
        (_FakeResult(dev_mode=True), False, []),
        (_FakeResult(dev_mode=False, task_filter_applied=True), False, []),
        (None, False, []),
    ],
    ids=[
        "full_result_full_run_is_reused",
        "full_result_dev_run_is_reused",
        "dev_result_dev_run_is_reused",
        "dev_result_full_run_is_recomputed",
        "filtered_result_is_recomputed",
        "absent_result_is_recomputed",
    ],
)
def test_reuse_follows_biotrainers_rule(tmp_path, stored, development_mode, expected):
    report_path = tmp_path / "autoeval_report_a-run-ckpt27500.json"
    report_path.touch()
    results = {} if stored is None else {PGYM_REPORT_NAME: stored}

    reused = _reused_frameworks(
        biotrainer=_fake_biotrainer(results),
        report_path=report_path,
        frameworks=["pgym"],
        development_mode=development_mode,
    )

    assert reused == expected


def test_nothing_is_reused_when_no_report_exists(tmp_path):
    reused = _reused_frameworks(
        biotrainer=_fake_biotrainer({PGYM_REPORT_NAME: _FakeResult(dev_mode=False)}),
        report_path=tmp_path / "absent" / "autoeval_report_a-run-ckpt27500.json",
        frameworks=["pgym"],
        development_mode=False,
    )

    assert reused == []


EMBEDDER = "a-run-ckpt27500"
REPORT_NAME = f"autoeval_report_{EMBEDDER}.json"
SIDECAR_NAME = f"autoeval_report_{EMBEDDER}.provenance.json"


def _check_version(output_dir: Path, version: str = "2.1.1") -> None:
    _refuse_results_from_another_biotrainer(
        output_dir=output_dir,
        embedder_name=EMBEDDER,
        report_path=output_dir / REPORT_NAME,
        version=version,
    )


def _write_sidecar(output_dir: Path, record: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SIDECAR_NAME).write_text(json.dumps(record), encoding="utf-8")


@pytest.mark.parametrize(
    "sidecar, evidence",
    [
        ({"biotrainer_version": "2.0.0"}, "records biotrainer 2.0.0"),
        ({"embedder_name": EMBEDDER}, "has no biotrainer_version"),
        (None, "there is no provenance sidecar"),
    ],
    ids=["other-version", "no-version-field", "no-sidecar"],
)
def test_results_from_another_biotrainer_are_refused(tmp_path, sidecar, evidence):
    out = tmp_path / "out"
    out.mkdir()
    (out / REPORT_NAME).write_text("{}", encoding="utf-8")
    if sidecar is not None:
        _write_sidecar(out, sidecar)

    with pytest.raises(RuntimeError, match="fresh output_dir") as excinfo:
        _check_version(out)
    assert evidence in str(excinfo.value)


def test_a_contact_cache_without_a_report_is_refused(tmp_path):
    # A crashed run leaves the per-protein cache, which AutoEval reuses, but no report.
    out = tmp_path / "out"
    cache = out / EMBEDDER / "PBC_ZEROSHOT_CONTACT"
    cache.mkdir(parents=True)
    (cache / "zero_shot_contact_cached_results_categorical_jacobian.json").write_text("{}")
    _write_sidecar(out, {"biotrainer_version": "2.0.0"})

    with pytest.raises(RuntimeError, match="2.0.0"):
        _check_version(out)


def test_results_from_the_same_biotrainer_are_reused(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / REPORT_NAME).write_text("{}", encoding="utf-8")
    (out / EMBEDDER / "PGYM").mkdir(parents=True)
    _write_sidecar(out, {"biotrainer_version": "2.1.1"})

    _check_version(out)


def test_empty_or_missing_output_dir_is_accepted(tmp_path):
    (tmp_path / "empty").mkdir()

    _check_version(tmp_path / "empty")
    _check_version(tmp_path / "missing")


def test_other_embedders_in_the_output_dir_do_not_block(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "autoeval_report_old-run-ckpt100.json").write_text("{}", encoding="utf-8")
    (out / "old-run-ckpt100" / "PGYM").mkdir(parents=True)

    _check_version(out)


def test_eval_modules_import_without_biotrainer():
    blocked = ("biotrainer", "biotrainer_core")
    nanoplm_eval = "nanoplm.eval"

    class _Blocker:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in blocked:
                raise ImportError(f"blocked for this test: {fullname}")
            return None

    def _purged(name):
        root = name.split(".")[0]
        return root in blocked or name == nanoplm_eval or name.startswith(
            nanoplm_eval + "."
        )

    def _purge():
        for name in list(sys.modules):
            if _purged(name):
                sys.modules.pop(name, None)

    # Put the original modules back afterwards, and rebind them on their parent
    # packages, so modules imported before this test keep seeing the same
    # objects as later imports.
    saved = {name: mod for name, mod in sys.modules.items() if _purged(name)}
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
        sys.modules.update(saved)
        for name, mod in saved.items():
            parent, _, child = name.rpartition(".")
            if parent in sys.modules:
                setattr(sys.modules[parent], child, mod)
        importlib.import_module("nanoplm.eval.checkpoint")
        importlib.import_module("nanoplm.eval.pipeline")
