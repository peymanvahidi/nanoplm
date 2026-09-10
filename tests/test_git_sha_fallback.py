"""`git_sha` must resolve a commit on a host with no `git` binary.

JUPITER compute nodes have no `git` on PATH, so `subprocess.run(["git", ...])`
raised OSError and every `run.json` written by a training job recorded
`"git_sha": null`, losing the record of which commit produced the checkpoint.
The fallback reads `.git` directly, which cannot see uncommitted changes, so
those shas are marked `-dirty-unknown`.
"""

import subprocess
from pathlib import Path

from nanoplm.utils.common import _head_sha_from_git_files, git_sha

SHA = "9f4d1c0a3b7e5d2f8a6c4b1e0d9f7a3c5b2e8d10"
OTHER = "1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d"
MASTER = "refs/heads/master"
SLASHED = "refs/heads/feat/ux-logging-and-output-layout"


def _checkout(root: Path, head: str = f"ref: {MASTER}\n") -> Path:
    git_dir = root / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text(head, encoding="utf-8")
    return git_dir


def _worktree(root: Path, name: str, head: str) -> Path:
    common_dir = _checkout(root / "main")
    git_dir = common_dir / "worktrees" / name
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text(head, encoding="utf-8")
    (git_dir / "commondir").write_text("../..\n", encoding="utf-8")

    linked = root / name
    linked.mkdir()
    (linked / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
    return linked


def _loose(git_dir: Path, ref: str, sha: str) -> None:
    path = git_dir / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sha + "\n", encoding="utf-8")


def _packed(git_dir: Path, ref: str, sha: str) -> None:
    (git_dir / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted \n"
        f"{OTHER} refs/heads/feat/decoy\n"
        f"{OTHER} {ref.replace('refs/heads/', 'refs/remotes/origin/')}\n"
        f"{OTHER} refs/tags/v1\n"
        f"^{SHA}\n"
        f"{sha} {ref}\n",
        encoding="utf-8",
    )
    assert not (git_dir / ref).exists()


def _fake_git(rev_parse, status):
    def run(cmd, **kwargs):
        if cmd[1] == "rev-parse":
            return subprocess.CompletedProcess(cmd, *rev_parse)
        assert cmd[1:] == ["status", "--porcelain"], cmd
        return subprocess.CompletedProcess(cmd, *status)

    return run


def test_loose_ref_in_a_plain_git_dir(tmp_path):
    _loose(_checkout(tmp_path), MASTER, SHA)
    assert _head_sha_from_git_files(tmp_path) == SHA


def test_detached_head_is_a_raw_sha(tmp_path):
    _checkout(tmp_path, head=f"{SHA}\n")
    assert _head_sha_from_git_files(tmp_path) == SHA


def test_packed_ref_when_no_loose_file_exists(tmp_path):
    _packed(_checkout(tmp_path), MASTER, SHA)
    assert _head_sha_from_git_files(tmp_path) == SHA


def test_branch_name_with_a_slash(tmp_path):
    _loose(_checkout(tmp_path, head=f"ref: {SLASHED}\n"), SLASHED, SHA)
    assert _head_sha_from_git_files(tmp_path) == SHA


def test_nested_subdirectory_resolves_the_same_sha(tmp_path):
    _loose(_checkout(tmp_path), MASTER, SHA)
    nested = tmp_path / "src" / "nanoplm" / "pretraining" / "models"
    nested.mkdir(parents=True)
    assert _head_sha_from_git_files(nested) == SHA


def test_worktree_pointer_file_resolves_a_loose_ref_from_the_common_dir(tmp_path):
    linked = _worktree(tmp_path, "wt", head=f"ref: {SLASHED}\n")
    _loose(tmp_path / "main" / ".git", SLASHED, SHA)
    assert _head_sha_from_git_files(linked) == SHA
    assert _head_sha_from_git_files(linked / "src") == SHA


def test_worktree_pointer_file_resolves_a_packed_ref_from_the_common_dir(tmp_path):
    linked = _worktree(tmp_path, "wt", head=f"ref: {SLASHED}\n")
    _packed(tmp_path / "main" / ".git", SLASHED, SHA)
    assert _head_sha_from_git_files(linked) == SHA


def test_worktree_pointer_file_written_as_a_relative_path(tmp_path):
    linked = _worktree(tmp_path, "wt", head=f"ref: {SLASHED}\n")
    (linked / ".git").write_text("gitdir: ../main/.git/worktrees/wt\n", encoding="utf-8")
    _loose(tmp_path / "main" / ".git", SLASHED, SHA)
    assert _head_sha_from_git_files(linked) == SHA


def test_worktree_pointer_file_resolves_a_detached_head(tmp_path):
    linked = _worktree(tmp_path, "wt", head=f"{SHA}\n")
    _loose(tmp_path / "main" / ".git", MASTER, OTHER)
    assert _head_sha_from_git_files(linked) == SHA


def test_no_git_directory_anywhere_returns_none(tmp_path):
    bare = tmp_path / "not" / "a" / "checkout"
    bare.mkdir(parents=True)
    assert _head_sha_from_git_files(bare) is None


def test_unresolvable_ref_returns_none(tmp_path):
    _checkout(tmp_path, head=f"ref: {SLASHED}\n")
    assert _head_sha_from_git_files(tmp_path) is None


def test_git_sha_falls_back_when_the_git_binary_is_missing(tmp_path, monkeypatch):
    _loose(_checkout(tmp_path, head=f"ref: {SLASHED}\n"), SLASHED, SHA)
    nested = tmp_path / "src" / "nanoplm" / "pretraining"
    nested.mkdir(parents=True)

    def no_git(*args, **kwargs):
        raise OSError(2, "No such file or directory: 'git'")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert git_sha(nested) == f"{SHA}-dirty-unknown"


def test_git_sha_returns_none_without_git_or_git_files(tmp_path, monkeypatch):
    def no_git(*args, **kwargs):
        raise OSError(2, "No such file or directory: 'git'")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert git_sha(tmp_path) is None


def test_git_sha_reports_a_clean_tree_as_a_bare_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_git((0, SHA + "\n"), (0, "")))
    assert git_sha(tmp_path) == SHA


def test_git_sha_reports_uncommitted_changes_as_dirty(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", _fake_git((0, SHA + "\n"), (0, " M src/nanoplm/x.py\n"))
    )
    assert git_sha(tmp_path) == f"{SHA}-dirty"


def test_git_sha_reports_a_failed_status_as_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_git((0, SHA + "\n"), (128, "")))
    assert git_sha(tmp_path) == f"{SHA}-dirty-unknown"
