from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable


_ERROR_BUFFER_BYTES = 4096
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


def _compile_library() -> Path:
    source = _source_path()
    if not source.exists():
        raise NativeFastaOpsError(f"Native FASTA ops source file not found: {source}")

    out_path = _library_path()
    if out_path.exists() and out_path.stat().st_mtime >= source.stat().st_mtime:
        return out_path

    tmp_out = out_path.with_suffix(f".{os.getpid()}.tmp.so")
    base_cmd = [
        "gcc",
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
        proc = subprocess.run(cmd, capture_output=True, text=True)
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


def filter_fasta_native(
    input_path: str | Path,
    output_path: str | Path,
    min_seq_len: int,
    max_seq_len: int,
    seqs_num: int,
    skip_n: int = 0,
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
        _PROGRESS_CB,  # progress callback
        ctypes.POINTER(ctypes.c_longlong),  # out_processed
        ctypes.POINTER(ctypes.c_longlong),  # out_passed
        ctypes.c_char_p,  # error_msg
        ctypes.c_size_t,  # error_cap
    ]
    func.restype = ctypes.c_int

    out_processed = ctypes.c_longlong(0)
    out_passed = ctypes.c_longlong(0)
    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
    c_progress_cb = (
        _PROGRESS_CB(progress_cb)
        if progress_cb is not None
        else ctypes.cast(None, _PROGRESS_CB)
    )

    rc = func(
        str(Path(input_path)).encode("utf-8"),
        str(Path(output_path)).encode("utf-8"),
        int(min_seq_len),
        int(max_seq_len),
        int(seqs_num),
        int(skip_n),
        c_progress_cb,
        ctypes.byref(out_processed),
        ctypes.byref(out_passed),
        error_buffer,
        ctypes.c_size_t(_ERROR_BUFFER_BYTES),
    )
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
        ctypes.POINTER(ctypes.c_longlong),  # out_train
        ctypes.POINTER(ctypes.c_longlong),  # out_val
        ctypes.c_char_p,  # error_msg
        ctypes.c_size_t,  # error_cap
    ]
    func.restype = ctypes.c_int

    out_train = ctypes.c_longlong(0)
    out_val = ctypes.c_longlong(0)
    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
    c_progress_cb = (
        _PROGRESS_CB(progress_cb)
        if progress_cb is not None
        else ctypes.cast(None, _PROGRESS_CB)
    )

    rc = func(
        str(Path(input_path)).encode("utf-8"),
        str(Path(train_path)).encode("utf-8"),
        str(Path(val_path)).encode("utf-8"),
        float(val_ratio),
        c_progress_cb,
        ctypes.byref(out_train),
        ctypes.byref(out_val),
        error_buffer,
        ctypes.c_size_t(_ERROR_BUFFER_BYTES),
    )
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
        ctypes.POINTER(ctypes.c_longlong),  # out_num_records
        ctypes.c_char_p,  # error_msg
        ctypes.c_size_t,  # error_cap
    ]
    func.restype = ctypes.c_int

    out_count = ctypes.c_longlong(0)
    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
    c_progress_cb = (
        _SHUFFLE_PROGRESS_CB(progress_cb)
        if progress_cb is not None
        else ctypes.cast(None, _SHUFFLE_PROGRESS_CB)
    )

    rc = func(
        str(Path(input_path)).encode("utf-8"),
        str(Path(output_path)).encode("utf-8"),
        int(seed) & ((1 << 64) - 1),
        int(num_threads),
        int(batch_records),
        c_progress_cb,
        ctypes.byref(out_count),
        error_buffer,
        ctypes.c_size_t(_ERROR_BUFFER_BYTES),
    )
    if rc != 0:
        msg = error_buffer.value.decode("utf-8", errors="replace").strip()
        if not msg:
            msg = "Unknown native shuffle error"
        raise NativeFastaOpsError(msg)

    return NativeShuffleResult(record_count=int(out_count.value))
