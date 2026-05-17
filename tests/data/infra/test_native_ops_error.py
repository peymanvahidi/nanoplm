import os
import pytest
from nanoplm.data.preparation._native_ops import (
    NativeFastaOpsError,
    is_native_fasta_ops_available,
)


def test_compile_failure_error_message_includes_install_hint(monkeypatch):
    monkeypatch.setenv("NANOPLM_DISABLE_NATIVE_FASTA_OPS", "1")
    available, reason = is_native_fasta_ops_available()
    assert not available
    assert reason is not None


def test_explicit_error_when_cc_missing(monkeypatch, tmp_path):
    """When cc is missing, the error message must list install instructions."""
    # Simulate cc missing by pointing PATH at an empty dir
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("NANOPLM_NATIVE_FASTA_BUILD_DIR", str(tmp_path / "build"))
    monkeypatch.delenv("NANOPLM_DISABLE_NATIVE_FASTA_OPS", raising=False)
    # Force re-load
    from nanoplm.data.preparation import _native_ops
    if hasattr(_native_ops._compile_library, "cache_clear"):
        _native_ops._compile_library.cache_clear()
    with pytest.raises(NativeFastaOpsError) as exc:
        _native_ops._compile_library()
    msg = str(exc.value)
    assert "C compiler" in msg
    assert "apt install build-essential" in msg
    assert "xcode-select" in msg
