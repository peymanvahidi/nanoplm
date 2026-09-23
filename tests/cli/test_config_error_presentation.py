import click
import pytest
from click.testing import CliRunner

from nanoplm.cli import cli
from nanoplm.cli._common import unknown_keys_message
from nanoplm.cli.eval import eval as eval_group
from nanoplm.cli.pretrain import pretrain


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def pretrain_typo_yaml(tmp_path):
    return _write(
        tmp_path / "pretrain.yaml",
        'pretraining:\n  dataset_dir: "data"\n  run_nmae: "run"\n',
    )


@pytest.fixture
def eval_typo_yaml(tmp_path):
    return _write(
        tmp_path / "eval.yaml",
        'eval:\n  checkpoint_dir: "ckpt"\n  framewrok: [pgym]\n',
    )


def test_pretrain_typo_prints_one_error_line(pretrain_typo_yaml):
    result = CliRunner().invoke(pretrain, ["from-yaml", pretrain_typo_yaml])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert result.output.strip() == (
        "Error: Unexpected training configuration keys: "
        "run_nmae (did you mean run_name?)"
    )


def test_eval_typo_prints_one_error_line(eval_typo_yaml):
    result = CliRunner().invoke(eval_group, ["from-yaml", eval_typo_yaml])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert result.output.strip() == (
        "Error: Unexpected eval configuration keys: "
        "framewrok (did you mean framework?)"
    )


def test_post_init_validation_error_also_collapses(tmp_path):
    config = _write(
        tmp_path / "pretrain.yaml",
        'pretraining:\n  dataset_dir: "data"\n  lr_schedule: "cosin"\n',
    )
    result = CliRunner().invoke(pretrain, ["from-yaml", config])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert result.output.startswith("Error: lr_schedule must be one of")


def test_debug_env_keeps_the_traceback(pretrain_typo_yaml, monkeypatch):
    monkeypatch.setenv("NANOPLM_DEBUG", "1")
    result = CliRunner().invoke(pretrain, ["from-yaml", pretrain_typo_yaml])
    assert isinstance(result.exception, ValueError)
    assert not isinstance(result.exception, click.ClickException)


@pytest.mark.parametrize("group", ["data", "deps", "distill", "eval", "pretrain"])
def test_group_help_exits_zero(group):
    """A group that collapses errors must not swallow click's clean Exit.

    click.exceptions.Exit and click.Abort both subclass RuntimeError, so a
    guard that catches RuntimeError turns '--help' into 'Error: 0' and exit 1.
    """
    result = CliRunner().invoke(cli, [group, "--help"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("Usage:")


@pytest.mark.parametrize(
    "group,subcommand",
    [("pretrain", "from-yaml"), ("pretrain", "get-yaml"), ("pretrain", "run"),
     ("eval", "from-yaml"), ("eval", "get-yaml"), ("eval", "run")],
)
def test_subcommand_help_exits_zero(group, subcommand):
    result = CliRunner().invoke(cli, [group, subcommand, "--help"])
    assert result.exit_code == 0, result.output


def test_unknown_keys_message_declines_a_weak_match():
    valid = ["adam_epsilon", "adam_learning_rate", "run_name"]
    assert unknown_keys_message("training", ["adam_lr"], valid) == (
        "Unexpected training configuration keys: adam_lr"
    )
    assert unknown_keys_message("training", ["run_nmae"], valid) == (
        "Unexpected training configuration keys: run_nmae (did you mean run_name?)"
    )


def test_unknown_keys_message_lists_every_key_sorted():
    valid = ["dataset_dir", "run_name", "seed"]
    assert unknown_keys_message("training", ["seedd", "zzqq"], valid) == (
        "Unexpected training configuration keys: seedd (did you mean seed?), zzqq"
    )
