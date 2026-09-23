import pytest
from click.testing import CliRunner

from nanoplm.cli import cli
from nanoplm.cli._common import LOG_LEVELS, YAML_FIRST_SECTIONS


def _help(*args):
    result = CliRunner().invoke(cli, [*args, "--help"])
    assert result.exit_code == 0, result.output
    return result.output


def test_root_help_does_not_truncate_a_group_description():
    output = _help()
    listing = output.partition("Commands:")[2]
    assert listing, output
    truncated = [line for line in listing.splitlines() if line.rstrip().endswith("...")]
    assert truncated == []
    assert "Manage protein datasets for nanoPLM pretraining and distillation." in output


@pytest.mark.parametrize("group", ["pretrain", "eval"])
def test_group_help_lists_commands_under_section_headers(group):
    output = _help(group)
    for header, _ in YAML_FIRST_SECTIONS:
        assert f"{header}:" in output


@pytest.mark.parametrize("group", ["pretrain", "eval"])
def test_every_subcommand_is_placed_in_a_section(group):
    """A new subcommand must join a section, not land in 'Other commands'."""
    assert "Other commands" not in _help(group)


@pytest.mark.parametrize(
    "group,subcommand",
    [("pretrain", "from-yaml"), ("pretrain", "run"),
     ("eval", "from-yaml"), ("eval", "run")],
)
def test_log_level_is_wired_to_the_running_commands(group, subcommand):
    output = _help(group, subcommand)
    assert "--log-level" in output
    for level in LOG_LEVELS:
        assert level in output


@pytest.mark.parametrize("group,subcommand", [("pretrain", "run"), ("eval", "run")])
def test_log_level_rejects_an_unknown_level(group, subcommand):
    result = CliRunner().invoke(cli, [group, subcommand, "--log-level", "shout"])
    assert result.exit_code == 2
    assert "is not one of" in result.output
