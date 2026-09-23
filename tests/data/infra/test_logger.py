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


def test_log_filename_is_the_job_id_not_the_pid(tmp_path, monkeypatch):
    """One file per job, so the 16 ranks of one job share it instead of each
    opening its own PID-named copy."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    import sys
    sys.modules.pop("nanoplm.utils.logger", None)
    from nanoplm.utils.logger import logger
    logger.info("hello")
    files = list((tmp_path / "output" / "logs").glob("*.log"))
    assert [f.name for f in files] == ["nanoplm_4242.log"]


def test_log_filename_falls_back_to_a_timestamp_off_slurm(tmp_path, monkeypatch):
    import re
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    import sys
    sys.modules.pop("nanoplm.utils.logger", None)
    from nanoplm.utils.logger import logger
    logger.info("hello")
    files = list((tmp_path / "output" / "logs").glob("*.log"))
    assert len(files) == 1
    assert re.fullmatch(r"nanoplm_\d{8}_\d{6}\.log", files[0].name), files[0].name


def test_non_zero_rank_opens_no_log_file(tmp_path, monkeypatch):
    """The console is what SLURM captures, so only the speaking rank keeps a file."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    import sys
    sys.modules.pop("nanoplm.utils.logger", None)
    from nanoplm.utils.logger import logger
    logger.warning("a warning still reaches the console")
    assert not (tmp_path / "output" / "logs").exists()


def test_all_ranks_opt_in_names_the_file_by_rank(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("NANOPLM_LOG_ALL_RANKS", "1")
    import sys
    sys.modules.pop("nanoplm.utils.logger", None)
    from nanoplm.utils.logger import logger
    logger.info("hello")
    files = list((tmp_path / "output" / "logs").glob("*.log"))
    assert [f.name for f in files] == ["nanoplm_4242_rank3.log"]


def test_log_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NANOPLM_LOG_DIR", str(tmp_path / "elsewhere"))
    import sys
    sys.modules.pop("nanoplm.utils.logger", None)
    from nanoplm.utils.logger import logger
    logger.info("hello")
    assert list((tmp_path / "elsewhere").glob("*.log"))
    assert not (tmp_path / "output").exists()


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
