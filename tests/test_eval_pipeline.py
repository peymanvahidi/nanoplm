"""run_eval wiring into AutoEval. Runs with no GPU and no biotrainer: every
biotrainer symbol and the checkpoint loading are stubbed.
"""

import json
import sys
from types import SimpleNamespace

import pytest

import nanoplm.eval.pipeline as pipeline
from nanoplm.eval.config import EvalConfig


class _RecordingAutoEval:
    instances: list = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls: list = []
        _RecordingAutoEval.instances.append(self)

    def __getattr__(self, name):
        def record(**kwargs):
            self.calls.append((name, kwargs))
        return record

    def run(self, device):
        return SimpleNamespace(
            all_framework_names=lambda: [], summary=lambda **_: None
        )


@pytest.fixture
def stubbed_pipeline(monkeypatch):
    _RecordingAutoEval.instances = []
    symbols = {
        "AutoEval": _RecordingAutoEval,
        "AutoEvalReport": SimpleNamespace(
            get_file_name=lambda name: f"autoeval_report_{name}.json"
        ),
        "version": "2.1.1",
    }
    monkeypatch.setattr(pipeline, "_load_biotrainer_symbols", lambda: symbols)
    monkeypatch.setattr(
        "nanoplm.eval._quack_untuned.disable_quack_autotune", lambda: False
    )
    monkeypatch.setattr(pipeline, "resolve_checkpoint_dir", lambda d: d)
    monkeypatch.setattr(pipeline, "load_model_config_from_checkpoint", lambda d: {})
    monkeypatch.setattr(
        pipeline,
        "load_checkpoint",
        lambda d, cfg, device: SimpleNamespace(tokenizer=None),
    )
    monkeypatch.setattr(
        pipeline.EvalProvenance,
        "resolve",
        classmethod(lambda cls, **_: SimpleNamespace(embedder_name="stub")),
    )
    monkeypatch.setattr(pipeline, "write_eval_provenance", lambda *_: None)
    monkeypatch.setattr(pipeline.NanoPLMBioEngineer, "build", lambda *a, **k: "engineer")
    monkeypatch.setattr(pipeline.NanoPLMCustomEmbedder, "build", lambda *a, **k: "embedder")
    return _RecordingAutoEval


def test_zero_shot_contact_gets_the_configured_batch_size(stubbed_pipeline, tmp_path):
    pipeline.run_eval(
        EvalConfig(
            checkpoint_dir=str(tmp_path / "ckpt"),
            framework=["pbc_zeroshot_contact"],
            output_dir=str(tmp_path / "out"),
            dataset_dir=str(tmp_path / "data"),
            device="cpu",
            zero_shot_batch_size=123,
            torch_compile=False,
        )
    )

    (autoeval,) = stubbed_pipeline.instances
    assert autoeval.calls == [("pbc_zeroshot_contact", {"batch_size": 123})]


def test_run_eval_disables_quack_autotune_before_loading_the_model(
    stubbed_pipeline, monkeypatch, tmp_path
):
    """from-yaml and programmatic callers reach run_eval without the CLI's
    `run` command, so the patch has to live in run_eval itself."""
    order = []
    monkeypatch.setattr(
        "nanoplm.eval._quack_untuned.disable_quack_autotune",
        lambda: order.append("quack") or True,
    )
    monkeypatch.setattr(
        pipeline,
        "load_checkpoint",
        lambda d, cfg, device: order.append("load") or SimpleNamespace(tokenizer=None),
    )
    pipeline.run_eval(
        EvalConfig(
            checkpoint_dir=str(tmp_path / "ckpt"),
            framework=["pbc_zeroshot_contact"],
            output_dir=str(tmp_path / "out"),
            dataset_dir=str(tmp_path / "data"),
            device="cpu",
            torch_compile=False,
        )
    )
    assert order == ["quack", "load"]


def test_from_yaml_reaches_the_quack_patch(stubbed_pipeline, monkeypatch, tmp_path):
    from click.testing import CliRunner

    import nanoplm.cli.eval as cli_eval

    calls = []
    monkeypatch.setattr(
        "nanoplm.eval._quack_untuned.disable_quack_autotune",
        lambda: calls.append(1) or True,
    )
    monkeypatch.setattr(cli_eval, "_check_biotrainer_available", lambda: None)
    config = tmp_path / "eval.yaml"
    config.write_text(
        "eval:\n"
        f"  checkpoint_dir: {tmp_path / 'ckpt'}\n"
        "  framework: [pbc_zeroshot_contact]\n"
        f"  output_dir: {tmp_path / 'out'}\n"
        f"  dataset_dir: {tmp_path / 'data'}\n"
        "  device: cpu\n"
        "  torch_compile: false\n"
    )
    result = CliRunner().invoke(cli_eval.eval, ["from-yaml", str(config)])
    assert result.exit_code == 0, result.output
    assert calls == [1]


def test_quack_patch_is_idempotent(monkeypatch):
    """Fake quack modules: the real ones would be mutated for the whole session."""
    import types

    import nanoplm.eval._quack_untuned as untuned

    quack = types.ModuleType("quack")
    gi = types.ModuleType("quack.gemm_interface")
    for name in untuned._NAMES:
        setattr(gi, name, lambda *a, **k: k)
    quack.gemm_interface = gi
    monkeypatch.setitem(sys.modules, "quack", quack)
    monkeypatch.setitem(sys.modules, "quack.gemm_interface", gi)
    monkeypatch.setitem(sys.modules, "sonicmoe", None)
    monkeypatch.setitem(sys.modules, "sonicmoe.functional", None)
    monkeypatch.delenv("NANOPLM_EVAL_QUACK_TUNED", raising=False)
    monkeypatch.setattr(untuned, "_applied", False)

    assert untuned.disable_quack_autotune() is True
    patched = gi.gemm
    assert untuned.disable_quack_autotune() is False
    assert gi.gemm is patched
    assert gi.gemm(tuned=True) == {"tuned": False}
    # Even with the flag reset, an already-wrapped entry point is not rewrapped.
    monkeypatch.setattr(untuned, "_applied", False)
    untuned.disable_quack_autotune()
    assert gi.gemm is patched
    assert not hasattr(patched.__wrapped__, "__wrapped__")


# ---------------------------------------------------------------------------
# Stale pre-2.1 dataset cache
# ---------------------------------------------------------------------------

def _make(root, *relpaths):
    for rel in relpaths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")


@pytest.mark.parametrize(
    "files",
    [
        ["CONTACT/contacts/casp14/x.fasta"],
        ["PBC/supervised/scl/scl.fasta"],
    ],
    ids=["old-contact-dir", "pbc-without-contacts"],
)
def test_pre_2_1_cache_layout_is_refused(tmp_path, files):
    _make(tmp_path, *files)
    with pytest.raises(RuntimeError, match="dataset_dir"):
        pipeline._refuse_stale_dataset_cache(tmp_path)


@pytest.mark.parametrize(
    "files",
    [
        [],
        ["PBC/contacts/zeroshot/casp14/x.fasta", "PBC/supervised/scl/scl.fasta"],
    ],
    ids=["empty", "2.1-layout"],
)
def test_empty_or_current_cache_is_accepted(tmp_path, files):
    _make(tmp_path, *files)
    (tmp_path / "PBC").mkdir(exist_ok=True)
    pipeline._refuse_stale_dataset_cache(tmp_path)
    pipeline._refuse_stale_dataset_cache(tmp_path / "missing")


def _write_pgym_reference(root, header):
    path = root / "PGYM" / "reference" / "reference.csv"
    path.parent.mkdir(parents=True)
    path.write_text(header + "\nrow\n")


def test_pgym_only_cache_without_pbc_dev_mode_is_refused(tmp_path):
    # The PR-192 reference.csv lacks the column 2.1 reads per row.
    _write_pgym_reference(tmp_path, "DMS_index,DMS_id,DMS_filename")
    with pytest.raises(RuntimeError, match="reference.csv"):
        pipeline._refuse_stale_dataset_cache(tmp_path)


def test_pgym_reference_with_pbc_dev_mode_is_accepted(tmp_path):
    _write_pgym_reference(tmp_path, "DMS_index,DMS_id,pbc_dev_mode,DMS_filename")
    pipeline._refuse_stale_dataset_cache(tmp_path)


@pytest.mark.parametrize("force_download", [False, True])
def test_run_eval_checks_the_default_cache_only_without_force_download(
    stubbed_pipeline, monkeypatch, tmp_path, force_download
):
    handler = pytest.importorskip("biotrainer.autoeval.core.autoeval_data_handler")
    cache = tmp_path / "cache"
    _make(cache, "CONTACT/contacts/casp14/x.fasta")
    monkeypatch.setattr(handler, "get_autoeval_path", lambda: cache)
    config = EvalConfig(
        checkpoint_dir=str(tmp_path / "ckpt"),
        framework=["pbc_zeroshot_contact"],
        output_dir=str(tmp_path / "out"),
        device="cpu",
        torch_compile=False,
        force_download=force_download,
    )

    if force_download:
        pipeline.run_eval(config)
        assert stubbed_pipeline.instances[0].init_kwargs["force_download"] is True
    else:
        with pytest.raises(RuntimeError, match="layout of biotrainer < 2.1"):
            pipeline.run_eval(config)
        assert stubbed_pipeline.instances == []


def _write_real_provenance(monkeypatch, tmp_path):
    """Swap the fixture's no-op sidecar writer for the real one."""
    from nanoplm.eval.checkpoint import write_eval_provenance

    monkeypatch.setattr(pipeline, "write_eval_provenance", write_eval_provenance)
    monkeypatch.setattr(
        pipeline.EvalProvenance,
        "resolve",
        classmethod(
            lambda cls, **kw: cls(
                embedder_name="stub",
                checkpoint_dir=tmp_path / "ckpt",
                checkpoint_step=None,
                run_name="run",
                loop_count=None,
                eval_git_sha=None,
                eval_started_at="2026-10-15T00:00:00+00:00",
                training_run=None,
                biotrainer_version=kw["biotrainer_version"],
            )
        ),
    )


def test_run_eval_refuses_results_from_another_biotrainer(
    stubbed_pipeline, monkeypatch, tmp_path
):
    # With the real writer, a guard that ran after the sidecar write would
    # overwrite "2.0.0" with this run's version and then let the run through.
    _write_real_provenance(monkeypatch, tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "autoeval_report_stub.json").write_text("{}")
    sidecar = out / "autoeval_report_stub.provenance.json"
    sidecar.write_text('{"biotrainer_version": "2.0.0"}')
    config = EvalConfig(
        checkpoint_dir=str(tmp_path / "ckpt"),
        framework=["pbc_zeroshot_contact"],
        output_dir=str(out),
        dataset_dir=str(tmp_path / "data"),
        device="cpu",
        torch_compile=False,
    )

    with pytest.raises(RuntimeError, match="fresh output_dir"):
        pipeline.run_eval(config)
    assert stubbed_pipeline.instances == []
    assert json.loads(sidecar.read_text())["biotrainer_version"] == "2.0.0"


class _StoredResult:
    task_filter_applied = False

    def used_development_mode(self):
        return True


def test_run_eval_records_the_version_and_reuses_its_own_results(
    stubbed_pipeline, monkeypatch, tmp_path
):
    _write_real_provenance(monkeypatch, tmp_path)
    stored = SimpleNamespace(
        maybe_framework_result=lambda name: (
            _StoredResult() if name == "PBC_ZEROSHOT_CONTACT" else None
        )
    )
    symbols = pipeline._load_biotrainer_symbols()
    symbols["AutoEvalReport"].from_json_file = lambda path: stored
    messages = []
    real_message = pipeline._completion_message
    monkeypatch.setattr(
        pipeline,
        "_completion_message",
        lambda **kw: messages.append(kw) or real_message(**kw),
    )
    out = tmp_path / "out"
    config = EvalConfig(
        checkpoint_dir=str(tmp_path / "ckpt"),
        framework=["pbc_zeroshot_contact"],
        output_dir=str(out),
        dataset_dir=str(tmp_path / "data"),
        device="cpu",
        torch_compile=False,
    )

    pipeline.run_eval(config)
    sidecar = out / "autoeval_report_stub.provenance.json"
    assert json.loads(sidecar.read_text())["biotrainer_version"] == "2.1.1"

    assert (messages[0]["computed"], messages[0]["reused"]) == (
        ["pbc_zeroshot_contact"], []
    )

    (out / "autoeval_report_stub.json").write_text("{}")
    pipeline.run_eval(config)
    assert len(stubbed_pipeline.instances) == 2
    assert (messages[1]["computed"], messages[1]["reused"]) == (
        [], ["pbc_zeroshot_contact"]
    )
