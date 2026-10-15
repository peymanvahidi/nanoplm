"""run_eval wiring into AutoEval. Runs with no GPU and no biotrainer: every
biotrainer symbol and the checkpoint loading are stubbed.
"""

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
