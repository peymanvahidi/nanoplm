"""Drive ``prepare_run_and_steps`` and compare the run directory it creates with
the layout ``pretrain.yaml`` documents for ``run_name``.

``tests/docs_schema/test_run_dir_pattern.py`` proves the same claim by reading
the source, so the torch-free CI schema job can gate it on every PR. This is
the end-to-end version: it also catches the layout being built somewhere the
static reader does not look.
"""

import re
from pathlib import Path

import pytest

from nanoplm.pretraining.config import PretrainingConfig
from nanoplm.pretraining.utils import prepare_run_and_steps
from tests.docs_schema.test_run_dir_pattern import _STAMP_ENV_VARS, _documented_layout


def _prepare(tmp_path, monkeypatch, env):
    for key in (*_STAMP_ENV_VARS, "WORLD_SIZE", "RANK"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config = PretrainingConfig(
        dataset_dir=str(tmp_path / "data"),
        ckp_dir=str(tmp_path / "ckpts"),
        run_name="myrun",
    )
    return config, prepare_run_and_steps(config, None, 1)


@pytest.mark.parametrize(
    "env,expected_stamp",
    [
        ({"SLURM_JOB_ID": "1684851"}, "1684851"),
        ({"SLURM_JOB_ID": "1684851", "TORCHELASTIC_RUN_ID": "ignored"}, "1684851"),
        ({"TORCHELASTIC_RUN_ID": "elastic7"}, "elastic7"),
    ],
)
def test_run_dir_matches_the_documented_layout(tmp_path, monkeypatch, env, expected_stamp):
    config, run = _prepare(tmp_path, monkeypatch, env)
    documented = (
        _documented_layout()
        .replace("<run_name>", config.run_name)
        .replace("<stamp>", expected_stamp)
    )
    assert run.run_root.name == documented
    assert run.run_root.parent == Path(config.ckp_dir)


def test_run_dir_stamp_falls_back_to_an_eight_digit_clock_reading(tmp_path, monkeypatch):
    config, run = _prepare(tmp_path, monkeypatch, {})
    documented = _documented_layout().replace("<run_name>", config.run_name)
    prefix, _, suffix = documented.partition("<stamp>")
    assert suffix == ""
    stamp = run.run_root.name[len(prefix):]
    assert re.fullmatch(r"\d{8}", stamp), stamp
