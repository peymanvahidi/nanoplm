import click
import pytest
from click.testing import CliRunner

from nanoplm.cli.pretrain import _check_pure_torch_key, pretrain


@pytest.fixture
def warnings_seen(monkeypatch):
    # The nanoplm logger has propagate=False, so caplog cannot see it: spy instead.
    seen = []
    from nanoplm.utils.logger import logger as _nplm_logger
    monkeypatch.setattr(_nplm_logger, "warning",
                        lambda msg, *a, **k: seen.append(msg % a if a else msg))
    return seen


class TestPureTorchKey:
    def test_absent_is_silent(self, warnings_seen):
        _check_pure_torch_key({"model": {}})
        assert warnings_seen == []

    def test_true_warns_once(self, warnings_seen):
        _check_pure_torch_key({"pure_torch": True})
        assert len(warnings_seen) == 1
        assert "pure_torch is no longer needed" in warnings_seen[0]

    def test_false_is_rejected(self, warnings_seen):
        with pytest.raises(click.ClickException, match="no longer supported"):
            _check_pure_torch_key({"pure_torch": False})


@pytest.mark.parametrize("cmd", ["from-yaml", "run"])
def test_pure_torch_flag_is_hidden(cmd):
    r = CliRunner().invoke(pretrain, [cmd, "--help"])
    assert r.exit_code == 0
    assert "--pure-torch" not in r.output
