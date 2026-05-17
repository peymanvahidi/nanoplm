"""
Atomic write helpers. Provides context managers that write to a .tmp sibling
then rename on success; cleanup on error. Same-filesystem rename is atomic.
"""
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def atomic_write_path(target: Path) -> Iterator[Path]:
    """
    Yields a temporary Path; on successful exit, renames it to target.
    On exception, removes the temp file. Target's parent must exist.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        yield tmp
        tmp.replace(target)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


@contextmanager
def atomic_write_dir(target: Path) -> Iterator[Path]:
    """
    Yields a temporary directory; on successful exit, renames it to target.
    On exception, removes the temp directory.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        yield tmp
        if target.exists():
            shutil.rmtree(target)
        tmp.rename(target)
    except BaseException:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise
