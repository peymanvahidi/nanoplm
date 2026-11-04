"""``pbc_supervised_datasets`` normalisation in ``EvalConfig``.

YAML and ``eval run`` both build an ``EvalConfig``, so the spellings of "every
dataset" ('all', []) and the bare-string form are resolved in one place.
"""

import pytest
from click.testing import CliRunner

import nanoplm.cli.eval as cli_eval
from nanoplm.cli.eval import _load_eval_config
from nanoplm.eval.config import PBC_SUPERVISED_DATASETS, EvalConfig
from nanoplm.utils.common import read_yaml


def _from_yaml(tmp_path, value: str) -> EvalConfig:
    path = tmp_path / "eval.yaml"
    path.write_text(
        'eval:\n  checkpoint_dir: "ckpt"\n'
        f"  pbc_supervised_datasets: {value}\n"
    )
    return _load_eval_config(read_yaml(path)["eval"])


@pytest.mark.parametrize("value", ["all", "ALL", "[all]", "[]"])
def test_all_or_empty_means_every_dataset(tmp_path, value):
    assert _from_yaml(tmp_path, value).pbc_supervised_datasets is None


def test_bare_string_becomes_a_list(tmp_path):
    assert _from_yaml(tmp_path, "scl").pbc_supervised_datasets == ["scl"]


def test_list_is_kept(tmp_path):
    config = _from_yaml(tmp_path, "[secondary_structure, frustration-regression]")
    assert config.pbc_supervised_datasets == [
        "secondary_structure", "frustration-regression",
    ]


def test_null_keeps_the_default(tmp_path):
    config = _from_yaml(tmp_path, "null")
    assert config.pbc_supervised_datasets == ["secondary_structure", "scl"]


def test_names_are_matched_case_insensitively(tmp_path):
    config = _from_yaml(tmp_path, "[SCL, Secondary_Structure]")
    assert config.pbc_supervised_datasets == ["scl", "secondary_structure"]


def test_all_mixed_with_names_is_refused(tmp_path):
    with pytest.raises(ValueError, match="mixes 'all' with dataset names"):
        _from_yaml(tmp_path, "[scl, all]")


@pytest.mark.parametrize("value", ["5", "true", "{scl: 1}"])
def test_a_scalar_that_is_not_a_name_is_a_value_error(tmp_path, value):
    with pytest.raises(ValueError, match="must be 'all', a dataset name or a list"):
        _from_yaml(tmp_path, value)


def test_unknown_name_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Unknown pbc_supervised_datasets.*'ssp'"):
        _from_yaml(tmp_path, "[scl, ssp]")


@pytest.mark.parametrize(
    "flags, expected",
    [
        ([], ["secondary_structure", "scl"]),
        (["--pbc-supervised-dataset", "all"], None),
        (["--pbc-supervised-dataset", "phages"], ["phages"]),
    ],
)
def test_run_passes_the_flag_through(monkeypatch, flags, expected):
    seen = []
    monkeypatch.setattr(cli_eval, "_check_biotrainer_available", lambda: None)
    monkeypatch.setattr(
        "nanoplm.eval.pipeline.run_eval", lambda config: seen.append(config)
    )
    result = CliRunner().invoke(
        cli_eval.eval, ["run", "--checkpoint-dir", "ckpt", *flags]
    )
    assert result.exit_code == 0, result.output
    assert seen[0].pbc_supervised_datasets == expected


def test_run_reports_an_unknown_name_as_one_error_line(monkeypatch):
    monkeypatch.setattr(cli_eval, "_check_biotrainer_available", lambda: None)
    result = CliRunner().invoke(
        cli_eval.eval,
        ["run", "--checkpoint-dir", "ckpt", "--pbc-supervised-dataset", "ssp"],
    )
    assert result.exit_code != 0
    assert "Error: Unknown pbc_supervised_datasets ['ssp']" in result.output


def test_names_match_biotrainer():
    autoeval = pytest.importorskip(
        "biotrainer_core.data_classes.autoeval",
        reason="requires biotrainer",
    )
    assert set(PBC_SUPERVISED_DATASETS) == set(
        autoeval.all_pbc_supervised_datasets()
    )
