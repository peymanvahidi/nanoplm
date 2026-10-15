"""run_eval wiring into AutoEval. Runs with no GPU and no biotrainer: every
biotrainer symbol and the checkpoint loading are stubbed.
"""

import json
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


def test_run_eval_refuses_results_from_another_biotrainer(stubbed_pipeline, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "autoeval_report_stub.json").write_text("{}")
    (out / "autoeval_report_stub.provenance.json").write_text(
        '{"biotrainer_version": "2.0.0"}'
    )
    config = EvalConfig(
        checkpoint_dir=str(tmp_path / "ckpt"),
        framework=["pgym"],
        output_dir=str(out),
        dataset_dir=str(tmp_path / "data"),
        device="cpu",
        torch_compile=False,
    )

    with pytest.raises(RuntimeError, match="fresh output_dir"):
        pipeline.run_eval(config)
    assert stubbed_pipeline.instances == []


def test_run_eval_records_the_version_and_reuses_its_own_results(
    stubbed_pipeline, monkeypatch, tmp_path
):
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

    (out / "autoeval_report_stub.json").write_text("{}")
    pipeline.run_eval(config)
    assert len(stubbed_pipeline.instances) == 2
