"""Gate the run-directory pattern documented in pretrain.yaml against the code.

The docs_schema parity gate checks that a template documents every field that
exists. It cannot check that a comment is *true*, which is how the run_name
comment came to describe a ``DDMMHHMM-<run_name>`` layout that the naming code
has never produced. These tests close that gap for the one comment whose claim
is mechanically checkable.
"""

import inspect
import re
from pathlib import Path

import pytest

from nanoplm.cli.templates import load_template
from nanoplm.pretraining import utils as pretraining_utils
from nanoplm.pretraining.config import PretrainingConfig

_STAMP_ENV_VARS = ("SLURM_JOB_ID", "TORCHELASTIC_RUN_ID")


def _run_name_comment() -> str:
    """Every comment documenting ``run_name:``, whether above it or end-of-line.

    Both placements are read so that moving the comment cannot quietly disable
    this gate.
    """
    lines = load_template("pretrain").splitlines()
    index = next(i for i, line in enumerate(lines) if line.strip().startswith("run_name:"))
    block = []
    _, _, trailing = lines[index].partition("#")
    if trailing.strip():
        block.append(trailing.strip())
    above = index
    while above > 0 and lines[above - 1].strip().startswith("#"):
        above -= 1
        block.insert(0, lines[above].strip().lstrip("#").strip())
    assert block, "run_name carries no comment, so nothing documents the run directory"
    return " ".join(block)


def _documented_layout() -> str:
    """The ``<ckp_dir>/...`` path shape the comment claims, e.g. ``<run_name>-<stamp>``."""
    match = re.search(r"<ckp_dir>/(\S+?)\.", _run_name_comment())
    assert match, f"comment names no <ckp_dir>/... layout: {_run_name_comment()!r}"
    return match.group(1)


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
    run = pretraining_utils.prepare_run_and_steps(config, None, 1)
    return config, run


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


def test_run_dir_stamp_falls_back_to_the_documented_clock_format(tmp_path, monkeypatch):
    config, run = _prepare(tmp_path, monkeypatch, {})
    assert "DDMMHHMM" in _run_name_comment()
    documented = _documented_layout().replace("<run_name>", config.run_name)
    prefix, _, suffix = documented.partition("<stamp>")
    assert suffix == ""
    stamp = run.run_root.name[len(prefix):]
    assert re.fullmatch(r"\d{8}", stamp), stamp


def _stamp_helper():
    """The helper that turns the environment into the run-directory stamp.

    Found by name suffix rather than exact name so that making it public, or
    private again, does not silently disable the check below.
    """
    named = [
        getattr(pretraining_utils, name)
        for name in dir(pretraining_utils)
        if name.endswith("job_stamp")
    ]
    assert len(named) == 1, f"expected one *job_stamp helper, found {len(named)}"
    return named[0]


def test_comment_names_every_stamp_source_the_code_reads():
    """A stamp source added to the code must be documented, not just added."""
    source = inspect.getsource(_stamp_helper())
    read_from_env = set(re.findall(r'"([A-Z][A-Z0-9_]+)"', source)) - {"WORLD_SIZE"}
    assert read_from_env == set(_STAMP_ENV_VARS), read_from_env
    comment = _run_name_comment()
    for name in read_from_env:
        assert name in comment, f"{name} sets the stamp but the template omits it"
