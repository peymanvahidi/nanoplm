from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable


_ERROR_BUFFER_BYTES = 4096
_NATIVE_RC_CANCELLED = -2
_DISABLE_NATIVE_FASTA_OPS_ENV = "NANOPLM_DISABLE_NATIVE_FASTA_OPS"
_NATIVE_FASTA_BUILD_DIR_ENV = "NANOPLM_NATIVE_FASTA_BUILD_DIR"
_SHUFFLE_PROGRESS_CB = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_longlong,
    ctypes.c_longlong,
    ctypes.c_longlong,
)
_PROGRESS_CB = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_longlong,
    ctypes.c_longlong,
)


class NativeFastaOpsError(RuntimeError):
    """Raised when native FASTA operation compilation/execution fails."""


@dataclass(frozen=True)
class NativeFilterResult:
    processed_seqs: int
    passed_seqs: int


@dataclass(frozen=True)
class NativeSplitResult:
    train_size: int
    val_size: int


@dataclass(frozen=True)
class NativeShuffleResult:
    record_count: int


@dataclass(frozen=True)
class NativeValidateResult:
    total_sequences: int
    invalid_count: int
    empty_count: int
    duplicate_count: int
    dedup_truncated: bool


def _is_disabled() -> bool:
    return os.environ.get(_DISABLE_NATIVE_FASTA_OPS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _source_path() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "fast_fasta_ops.c"


def _build_dir() -> Path:
    env_dir = os.environ.get(_NATIVE_FASTA_BUILD_DIR_ENV, "").strip()
    if env_dir:
        out = Path(env_dir)
    else:
        out = Path(tempfile.gettempdir()) / "nanoplm_native_fasta_ops"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _library_path() -> Path:
    source_id = hashlib.sha1(str(_source_path()).encode("utf-8")).hexdigest()[:12]
    return _build_dir() / f"libnanoplm_fast_fasta_ops_{source_id}.so"


_INSTALL_HINT = (
    "Install a C compiler:\n"
    "  Debian/Ubuntu:  apt install build-essential\n"
    "  RHEL/CentOS:    dnf install gcc\n"
    "  macOS:          xcode-select --install\n"
    "  conda:          conda install -c conda-forge gcc\n"
)


def _compile_library() -> Path:
    source = _source_path()
    if not source.exists():
        raise NativeFastaOpsError(f"Native FASTA ops source file not found: {source}")

    # Check for compiler before consulting the on-disk cache so that a missing
    # compiler always produces an actionable error even when a stale .so exists.
    cc = os.environ.get("CC", "gcc")
    if shutil.which(cc) is None:
        raise NativeFastaOpsError(
            f"Native FASTA backend requires a C compiler. Compilation failed:\n"
            f"  FileNotFoundError: compiler '{cc}' not found on PATH\n\n"
            + _INSTALL_HINT
        )

    out_path = _library_path()
    if out_path.exists() and out_path.stat().st_mtime >= source.stat().st_mtime:
        return out_path

    tmp_out = out_path.with_suffix(f".{os.getpid()}.tmp.so")
    base_cmd = [
        cc,
        "-O3",
        "-march=native",
        "-fPIC",
        "-shared",
        "-std=c11",
        str(source),
        "-o",
        str(tmp_out),
    ]
    attempts = [[*base_cmd, "-fopenmp"], base_cmd]
    errors: list[str] = []

    for cmd in attempts:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise NativeFastaOpsError(
                f"Native FASTA backend requires a C compiler. Compilation failed:\n"
                f"  {type(e).__name__}: {e}\n\n"
                + _INSTALL_HINT
            ) from e
        if proc.returncode == 0:
            tmp_out.replace(out_path)
            return out_path
        errors.append(
            "Failed to compile native FASTA ops library.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    raise NativeFastaOpsError("\n\n".join(errors))


@lru_cache(maxsize=1)
def _load_library() -> ctypes.CDLL:
    if _is_disabled():
        raise NativeFastaOpsError(
            f"Native FASTA ops disabled by env var {_DISABLE_NATIVE_FASTA_OPS_ENV}=1"
        )
    return ctypes.CDLL(str(_compile_library()))


def is_native_fasta_ops_available() -> tuple[bool, str | None]:
    if _is_disabled():
        return (
            False,
            f"Native FASTA ops disabled by env var {_DISABLE_NATIVE_FASTA_OPS_ENV}=1",
        )
    try:
        _load_library()
        return True, None
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, str(exc)


@contextmanager
def _sigint_cancel_flag():
    """
    Yield a c_int flag that the C library polls for cooperative cancellation.

    The flag is initialized to 0. On entry to each native call the C side
    installs its own sigaction-based SIGINT handler (see cancel_scope_enter
    in fast_fasta_ops.c) so Ctrl+C is observed synchronously while ctypes
    is blocked in C; that handler sets the flag to 1. The C loops poll the
    flag and exit cleanly on cancellation.

    Python does NOT install a signal.signal handler here — this context
    manager only allocates and yields the flag the C side will write to.
    """
    yield ctypes.c_int(0)


def _wrap_progress_cb(progress_cb, cb_type):
    """Wrap a user progress callback so exceptions don't crash the C->Py bridge."""
    if progress_cb is None:
        return ctypes.cast(None, cb_type)

    def _safe(*args):
        try:
            progress_cb(*args)
        except BaseException:
            pass

    return cb_type(_safe)


def filter_fasta_native(
    input_path: str | Path,
    output_path: str | Path,
    min_seq_len: int,
    max_seq_len: int,
    seqs_num: int,
    skip_n: int = 0,
    exclude_ids: str | Path | None = None,
    progress_cb: Callable[[int, float, int, int], None] | None = None,
) -> NativeFilterResult:
    if _is_disabled():
        raise NativeFastaOpsError(
            f"Native FASTA ops disabled by env var {_DISABLE_NATIVE_FASTA_OPS_ENV}=1"
        )
    lib = _load_library()
    func = lib.nanoplm_filter_fasta
    func.argtypes = [
        ctypes.c_char_p,  # input_path
        ctypes.c_char_p,  # output_path
        ctypes.c_int,  # min_seq_len
        ctypes.c_int,  # max_seq_len
        ctypes.c_longlong,  # seqs_num
        ctypes.c_longlong,  # skip_n
        ctypes.c_char_p,  # exclude_ids_path (NEW — may be NULL)
        _PROGRESS_CB,  # progress callback
        ctypes.POINTER(ctypes.c_int),  # cancel_requested
        ctypes.POINTER(ctypes.c_longlong),  # out_processed
        ctypes.POINTER(ctypes.c_longlong),  # out_passed
        ctypes.c_char_p,  # error_msg
        ctypes.c_size_t,  # error_cap
    ]
    func.restype = ctypes.c_int

    out_processed = ctypes.c_longlong(0)
    out_passed = ctypes.c_longlong(0)
    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
    c_progress_cb = _wrap_progress_cb(progress_cb, _PROGRESS_CB)
    c_exclude_ids = str(Path(exclude_ids)).encode("utf-8") if exclude_ids is not None else None

    with _sigint_cancel_flag() as cancel_flag:
        rc = func(
            str(Path(input_path)).encode("utf-8"),
            str(Path(output_path)).encode("utf-8"),
            int(min_seq_len),
            int(max_seq_len),
            int(seqs_num),
            int(skip_n),
            c_exclude_ids,
            c_progress_cb,
            ctypes.byref(cancel_flag),
            ctypes.byref(out_processed),
            ctypes.byref(out_passed),
            error_buffer,
            ctypes.c_size_t(_ERROR_BUFFER_BYTES),
        )
        cancelled = cancel_flag.value != 0 or rc == _NATIVE_RC_CANCELLED

    if cancelled:
        raise KeyboardInterrupt("Native FASTA filter cancelled")
    if rc != 0:
        msg = error_buffer.value.decode("utf-8", errors="replace").strip()
        if not msg:
            msg = "Unknown native filter error"
        raise NativeFastaOpsError(msg)

    return NativeFilterResult(
        processed_seqs=int(out_processed.value),
        passed_seqs=int(out_passed.value),
    )


def split_fasta_native(
    input_path: str | Path,
    train_path: str | Path,
    val_path: str | Path,
    val_ratio: float,
    progress_cb: Callable[[int, float, int, int], None] | None = None,
) -> NativeSplitResult:
    if _is_disabled():
        raise NativeFastaOpsError(
            f"Native FASTA ops disabled by env var {_DISABLE_NATIVE_FASTA_OPS_ENV}=1"
        )
    lib = _load_library()
    func = lib.nanoplm_split_fasta
    func.argtypes = [
        ctypes.c_char_p,  # input_path
        ctypes.c_char_p,  # train_path
        ctypes.c_char_p,  # val_path
        ctypes.c_double,  # val_ratio
        _PROGRESS_CB,  # progress callback
        ctypes.POINTER(ctypes.c_int),  # cancel_requested
        ctypes.POINTER(ctypes.c_longlong),  # out_train
        ctypes.POINTER(ctypes.c_longlong),  # out_val
        ctypes.c_char_p,  # error_msg
        ctypes.c_size_t,  # error_cap
    ]
    func.restype = ctypes.c_int

    out_train = ctypes.c_longlong(0)
    out_val = ctypes.c_longlong(0)
    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
    c_progress_cb = _wrap_progress_cb(progress_cb, _PROGRESS_CB)

    with _sigint_cancel_flag() as cancel_flag:
        rc = func(
            str(Path(input_path)).encode("utf-8"),
            str(Path(train_path)).encode("utf-8"),
            str(Path(val_path)).encode("utf-8"),
            float(val_ratio),
            c_progress_cb,
            ctypes.byref(cancel_flag),
            ctypes.byref(out_train),
            ctypes.byref(out_val),
            error_buffer,
            ctypes.c_size_t(_ERROR_BUFFER_BYTES),
        )
        cancelled = cancel_flag.value != 0 or rc == _NATIVE_RC_CANCELLED

    if cancelled:
        raise KeyboardInterrupt("Native FASTA split cancelled")
    if rc != 0:
        msg = error_buffer.value.decode("utf-8", errors="replace").strip()
        if not msg:
            msg = "Unknown native split error"
        raise NativeFastaOpsError(msg)

    return NativeSplitResult(
        train_size=int(out_train.value),
        val_size=int(out_val.value),
    )


def shuffle_fasta_native(
    input_path: str | Path,
    output_path: str | Path,
    seed: int,
    num_threads: int,
    batch_records: int = 262144,
    progress_cb: Callable[[int, float, int, int, int], None] | None = None,
) -> NativeShuffleResult:
    if _is_disabled():
        raise NativeFastaOpsError(
            f"Native FASTA ops disabled by env var {_DISABLE_NATIVE_FASTA_OPS_ENV}=1"
        )
    lib = _load_library()
    func = lib.nanoplm_shuffle_fasta
    func.argtypes = [
        ctypes.c_char_p,  # input_path
        ctypes.c_char_p,  # output_path
        ctypes.c_ulonglong,  # seed
        ctypes.c_int,  # num_threads
        ctypes.c_longlong,  # batch_records
        _SHUFFLE_PROGRESS_CB,  # progress callback
        ctypes.POINTER(ctypes.c_int),  # cancel_requested
        ctypes.POINTER(ctypes.c_longlong),  # out_num_records
        ctypes.c_char_p,  # error_msg
        ctypes.c_size_t,  # error_cap
    ]
    func.restype = ctypes.c_int

    out_count = ctypes.c_longlong(0)
    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
    c_progress_cb = _wrap_progress_cb(progress_cb, _SHUFFLE_PROGRESS_CB)

    with _sigint_cancel_flag() as cancel_flag:
        rc = func(
            str(Path(input_path)).encode("utf-8"),
            str(Path(output_path)).encode("utf-8"),
            int(seed) & ((1 << 64) - 1),
            int(num_threads),
            int(batch_records),
            c_progress_cb,
            ctypes.byref(cancel_flag),
            ctypes.byref(out_count),
            error_buffer,
            ctypes.c_size_t(_ERROR_BUFFER_BYTES),
        )
        cancelled = cancel_flag.value != 0 or rc == _NATIVE_RC_CANCELLED

    if cancelled:
        raise KeyboardInterrupt("Native FASTA shuffle cancelled")
    if rc != 0:
        msg = error_buffer.value.decode("utf-8", errors="replace").strip()
        if not msg:
            msg = "Unknown native shuffle error"
        raise NativeFastaOpsError(msg)

    return NativeShuffleResult(record_count=int(out_count.value))


def validate_fasta_native(
    input_path: Path,
    invalid_fasta_path: Path,
    invalid_ids_path: Path,
    report_json_path: Path,
    *,
    detect_duplicates: bool = True,
    progress_cb=None,
) -> NativeValidateResult:
    """Validate a FASTA file via the native C backend.

    Writes invalid records to `invalid_fasta_path`, their IDs to
    `invalid_ids_path`, and a JSON summary to `report_json_path`.

    progress_cb is reserved for future use and currently ignored;
    the C function does not surface progress today.
    """
    if _is_disabled():
        raise NativeFastaOpsError(
            f"Native FASTA ops disabled by env var {_DISABLE_NATIVE_FASTA_OPS_ENV}=1"
        )
    lib = _load_library()
    fn = lib.validate_fasta_native
    fn.argtypes = [
        ctypes.c_char_p,               # input_path
        ctypes.c_char_p,               # invalid_fasta_path
        ctypes.c_char_p,               # invalid_ids_path
        ctypes.c_char_p,               # report_json_path
        ctypes.c_int,                  # detect_duplicates
        ctypes.POINTER(ctypes.c_int),  # cancel_flag
        ctypes.c_char_p,               # err_buf
        ctypes.c_size_t,               # err_buf_size
    ]
    fn.restype = ctypes.c_int

    err_buf = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
    with _sigint_cancel_flag() as cancel_flag:
        rc = fn(
            str(input_path).encode(),
            str(invalid_fasta_path).encode(),
            str(invalid_ids_path).encode(),
            str(report_json_path).encode(),
            1 if detect_duplicates else 0,
            ctypes.byref(cancel_flag),
            err_buf,
            _ERROR_BUFFER_BYTES,
        )
    if rc == _NATIVE_RC_CANCELLED:
        raise KeyboardInterrupt()
    if rc != 0:
        raise NativeFastaOpsError(
            err_buf.value.decode(errors="replace") or "validate_fasta_native failed"
        )

    try:
        report = json.loads(Path(report_json_path).read_text())
        return NativeValidateResult(
            total_sequences=report["total_sequences"],
            invalid_count=report["invalid_count"],
            empty_count=report["empty_count"],
            duplicate_count=report["duplicate_count"],
            dedup_truncated=bool(report.get("dedup_truncated", False)),
        )
    except (json.JSONDecodeError, OSError, KeyError, TypeError) as exc:
        raise NativeFastaOpsError(
            f"validate_fasta_native: malformed report at {report_json_path}: {exc}"
        ) from exc
