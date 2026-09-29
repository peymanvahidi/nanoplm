from unittest.mock import patch

import pytest
import torch
import yaml
from click.testing import CliRunner

from nanoplm.cli import cli
from nanoplm.hf.verify import ExportVerificationError
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM

TINY = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_kv_heads=2,
    vocab_size=32,
    wo_init="kaiming",
)


def _write_checkpoint(root):
    model = PureProtModernBertMLM(ProtModernBertMLMConfig(**TINY)).eval()
    ckpt = root / "run-1" / "checkpoint-100"
    ckpt.mkdir(parents=True)
    torch.save(model.state_dict(), ckpt / "pytorch_model.bin")
    (ckpt / "model_config.yaml").write_text(yaml.safe_dump(TINY))
    (ckpt / "pretrain_config.yaml").write_text(
        yaml.safe_dump({"optimizer": "normuon", "lr_schedule": "wsd"})
    )
    (ckpt / "data_manifest.yaml").write_text(
        yaml.safe_dump({"max_seq_len": 512, "num_register_tokens": 0})
    )
    (ckpt / "training_state.json").write_text('{"global_step": 100}')
    return ckpt


def test_export_is_registered():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "export" in result.output


def test_export_help_documents_the_flags():
    result = CliRunner().invoke(cli, ["export", "--help"])
    assert result.exit_code == 0
    for flag in ("--out", "--push-to-hub", "--private", "--dtype", "--verify"):
        assert flag in result.output


def test_defaults_are_fp32_and_verify_on(tmp_path):
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    with patch("nanoplm.cli.export.export_checkpoint", return_value=tmp_path) as mock:
        result = CliRunner().invoke(cli, ["export", str(ckpt)])
    assert result.exit_code == 0, result.output
    kwargs = mock.call_args.kwargs
    assert kwargs["dtype"] == "fp32"
    assert kwargs["verify"] is True
    assert kwargs["push_to_hub"] is None


def test_no_verify_flag_disables_verification(tmp_path):
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    with patch("nanoplm.cli.export.export_checkpoint", return_value=tmp_path) as mock:
        result = CliRunner().invoke(cli, ["export", str(ckpt), "--no-verify"])
    assert result.exit_code == 0, result.output
    assert mock.call_args.kwargs["verify"] is False


def test_rejects_an_unknown_dtype(tmp_path):
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    result = CliRunner().invoke(cli, ["export", str(ckpt), "--dtype", "int4"])
    assert result.exit_code != 0
    assert "int4" in result.output


# --- R15-3(a): full forwarding -----------------------------------------------


def test_full_forwarding_of_all_flags(tmp_path):
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    out = tmp_path / "out"
    with patch("nanoplm.cli.export.export_checkpoint", return_value=out) as mock:
        result = CliRunner().invoke(
            cli,
            [
                "export",
                str(ckpt),
                "--out",
                str(out),
                "--dtype",
                "bf16",
                "--push-to-hub",
                "acme/x",
                "--private",
                "--no-verify",
            ],
        )
    assert result.exit_code == 0, result.output
    args, kwargs = mock.call_args
    assert args[0] == str(ckpt)
    assert args[1] == str(out)
    assert kwargs["dtype"] == "bf16"
    assert kwargs["push_to_hub"] == "acme/x"
    assert kwargs["private"] is True
    assert kwargs["verify"] is False
    assert "Wrote" in result.output
    assert "huggingface.co/acme/x" in result.output


# --- R15-3(b): error mapping --------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        FileExistsError("out_dir already exists and is not empty"),
        ExportVerificationError("native and export disagree"),
        RuntimeError("Transformer Engine is not importable"),
        ValueError("dtype must be one of {'fp32'}"),
        FileNotFoundError("model_config.yaml not found"),
    ],
)
def test_refusals_are_mapped_to_clean_cli_errors(tmp_path, exc):
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    with patch("nanoplm.cli.export.export_checkpoint", side_effect=exc):
        result = CliRunner().invoke(cli, ["export", str(ckpt)])
    assert result.exit_code == 1
    assert str(exc) in result.output
    assert isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output


# --- R15-3(c): one real invocation -------------------------------------------


def test_real_export_then_refuses_on_nonempty_out_dir(tmp_path):
    ckpt = _write_checkpoint(tmp_path)
    out = tmp_path / "hf"

    result = CliRunner().invoke(cli, ["export", str(ckpt), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "config.json").exists()
    assert (out / "model.safetensors").exists()
    assert "Wrote" in result.output

    result = CliRunner().invoke(cli, ["export", str(ckpt), "--out", str(out)])
    assert result.exit_code == 1
    assert "exists" in result.output
    assert "Traceback" not in result.output


def test_push_failure_is_a_clean_error_that_points_at_the_written_export(
    tmp_path, monkeypatch
):
    import huggingface_hub

    class _FailingHfApi:
        def __init__(self, *a, **k):
            pass

        def create_repo(self, *a, **k):
            raise OSError("401 Unauthorized: not logged in")

    monkeypatch.setattr(huggingface_hub, "HfApi", _FailingHfApi)
    ckpt = _write_checkpoint(tmp_path)
    out = tmp_path / "hf"

    result = CliRunner().invoke(
        cli, ["export", str(ckpt), "--out", str(out), "--push-to-hub", "acme/x"]
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "401 Unauthorized: not logged in" in result.output
    assert f"hf upload acme/x {out}" in result.output
    assert (out / "config.json").exists()
