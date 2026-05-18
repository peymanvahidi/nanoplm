# tests/data/dvc/test_help_grouping.py
from click.testing import CliRunner
from nanoplm.cli.data import data


def test_help_groups_commands():
    runner = CliRunner()
    result = runner.invoke(data, ["--help"])
    out = result.output
    assert "Main commands" in out
    assert "Pipeline stages" in out
    assert "Pretraining-specific" in out
    assert "Knowledge-distillation-specific" in out
    # get-yaml / from-yaml in Main, get-file in stages
    main_idx = out.index("Main commands")
    stages_idx = out.index("Pipeline stages")
    pre_idx = out.index("Pretraining-specific")
    kd_idx = out.index("Knowledge-distillation-specific")
    assert main_idx < stages_idx < pre_idx < kd_idx
    assert "get-yaml" in out[main_idx:stages_idx]
    assert "get-file" in out[stages_idx:pre_idx]
    assert "pre-tokenize" in out[pre_idx:kd_idx]
    assert "kd-embed" in out[kd_idx:]
