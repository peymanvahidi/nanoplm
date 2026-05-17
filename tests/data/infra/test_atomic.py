# tests/data/infra/test_atomic.py
from pathlib import Path
import pytest
from nanoplm.data.infra.atomic import atomic_write_path


def test_atomic_write_succeeds(tmp_path):
    target = tmp_path / "out.txt"
    with atomic_write_path(target) as tmp:
        tmp.write_text("hello")
    assert target.read_text() == "hello"
    # No leftover temp files
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_cleans_up_on_error(tmp_path):
    target = tmp_path / "out.txt"
    with pytest.raises(RuntimeError):
        with atomic_write_path(target) as tmp:
            tmp.write_text("partial")
            raise RuntimeError("simulated failure")
    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_to_directory(tmp_path):
    """atomic_write_dir lets the caller populate a directory then rename atomically."""
    from nanoplm.data.infra.atomic import atomic_write_dir
    target = tmp_path / "out_dir"
    with atomic_write_dir(target) as tmp_dir:
        (tmp_dir / "a.txt").write_text("a")
        (tmp_dir / "b.txt").write_text("b")
    assert target.is_dir()
    assert (target / "a.txt").read_text() == "a"
    assert (target / "b.txt").read_text() == "b"
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())


def test_atomic_write_dir_cleans_up_on_error(tmp_path):
    from nanoplm.data.infra.atomic import atomic_write_dir
    target = tmp_path / "out_dir"
    with pytest.raises(RuntimeError):
        with atomic_write_dir(target) as tmp_dir:
            (tmp_dir / "a.txt").write_text("a")
            raise RuntimeError("boom")
    assert not target.exists()
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())
