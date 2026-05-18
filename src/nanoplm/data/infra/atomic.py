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

    The rename is atomic on POSIX when the temp file and target live on the
    same filesystem; that holds here because the temp is a sibling of target.
    If target.parent crosses a filesystem boundary (bind mount / symlink),
    Path.replace falls back to copy+remove and atomicity is lost.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    # is_symlink covers dangling symlinks (exists() follows the link and may
    # return False even when the symlink path itself is present).
    if tmp.is_symlink() or tmp.exists():
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

    Replacement is NOT atomic: if target already exists it is rmtree'd before
    the rename, leaving a brief window in which target does not exist. Safe
    for single-writer, no-concurrent-reader pipelines (the nanoplm use case).
    Same-filesystem caveat from atomic_write_path applies to the rename here.
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
