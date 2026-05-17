import importlib
from pathlib import Path


def test_no_log_dir_created_at_import(tmp_path, monkeypatch):
    """Importing the logger module must NOT create output/logs/."""
    monkeypatch.chdir(tmp_path)
    # Force fresh import
    import sys
    sys.modules.pop("nanoplm.utils.logger", None)
    import nanoplm.utils.logger as lm  # noqa: F401
    assert not (tmp_path / "output" / "logs").exists()


def test_log_dir_created_lazily_on_first_emit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import sys
    sys.modules.pop("nanoplm.utils.logger", None)
    from nanoplm.utils.logger import logger
    logger.info("first message")
    assert (tmp_path / "output" / "logs").exists()
    files = list((tmp_path / "output" / "logs").glob("nanoplm_*.log"))
    assert len(files) == 1


def test_log_filename_is_timestamped_per_pid(tmp_path, monkeypatch):
    import re
    monkeypatch.chdir(tmp_path)
    import sys
    sys.modules.pop("nanoplm.utils.logger", None)
    from nanoplm.utils.logger import logger
    logger.info("hello")
    files = list((tmp_path / "output" / "logs").glob("*.log"))
    assert len(files) == 1
    pattern = re.compile(r"nanoplm_\d{8}_\d{6}_\d+\.log")
    assert pattern.match(files[0].name), f"unexpected filename: {files[0].name}"


def test_handlers_are_idempotent_across_reimports(tmp_path, monkeypatch):
    """Re-importing the logger module must not accumulate duplicate handlers."""
    monkeypatch.chdir(tmp_path)
    import sys
    sys.modules.pop("nanoplm.utils.logger", None)
    from nanoplm.utils.logger import logger as l1
    initial = len(l1.handlers)
    sys.modules.pop("nanoplm.utils.logger", None)
    from nanoplm.utils.logger import logger as l2
    # Same global Logger instance + same handler count
    assert l1 is l2
    assert len(l2.handlers) == initial
