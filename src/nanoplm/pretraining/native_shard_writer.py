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

from nanoplm.utils import logger


_ERROR_BUFFER_BYTES = 4096
_DISABLE_NATIVE_ENV = "NANOPLM_DISABLE_NATIVE_SHARD_WRITER"
_BUILD_DIR_ENV = "NANOPLM_NATIVE_BUILD_DIR"
_PROGRESS_CB = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_longlong,
    ctypes.c_longlong,
)


class NativeShardWriterError(RuntimeError):
    """Raised when the native shard writer cannot be compiled or executed."""


@dataclass(frozen=True)
class NativeShardResult:
    shard_paths: list[Path]
    sequence_count: int


def _is_disabled_by_env() -> bool:
    return os.environ.get(_DISABLE_NATIVE_ENV, "").strip().lower() in {"1", "true", "yes"}


def _native_source_path() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "fast_fasta_shard_writer.c"


def _native_build_dir() -> Path:
    env_dir = os.environ.get(_BUILD_DIR_ENV, "").strip()
    if env_dir:
        build_dir = Path(env_dir)
    else:
        build_dir = Path(tempfile.gettempdir()) / "nanoplm_native"
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir


def _native_library_path() -> Path:
    source = _native_source_path()
    source_id = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
    return _native_build_dir() / f"libnanoplm_fast_shard_{source_id}.so"


def _compile_native_library() -> Path:
    source = _native_source_path()
    if not source.exists():
        raise NativeShardWriterError(f"Native source file not found: {source}")

    out_path = _native_library_path()
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
    openmp_cmd = [*base_cmd, "-fopenmp"]
    attempts = [openmp_cmd, base_cmd]
    errors: list[str] = []

    for cmd in attempts:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            tmp_out.replace(out_path)
            return out_path
        errors.append(
            f"Command failed ({' '.join(cmd)}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    raise NativeShardWriterError(
        "Failed to compile native shard writer.\n" + "\n\n".join(errors)
    )


@lru_cache(maxsize=1)
def _load_native_function():
    if _is_disabled_by_env():
        raise NativeShardWriterError(
            f"Native shard writer disabled by env var {_DISABLE_NATIVE_ENV}=1"
        )

    lib_path = _compile_native_library()
    lib = ctypes.CDLL(str(lib_path))
    func = lib.nanoplm_create_fasta_shards
    func.argtypes = [
        ctypes.c_char_p,  # fasta_path
        ctypes.c_char_p,  # output_dir
        ctypes.c_int,  # max_length
        ctypes.c_int,  # samples_per_shard
        ctypes.c_int,  # num_threads
        ctypes.c_int,  # use_bos_token
        _PROGRESS_CB,  # progress callback
        ctypes.POINTER(ctypes.c_int),  # out_num_shards
        ctypes.POINTER(ctypes.c_longlong),  # out_num_sequences
        ctypes.c_char_p,  # error_message buffer
        ctypes.c_size_t,  # error_message capacity
    ]
    func.restype = ctypes.c_int
    return func


def is_native_shard_writer_available() -> tuple[bool, str | None]:
    if _is_disabled_by_env():
        return False, f"Native shard writer disabled by env var {_DISABLE_NATIVE_ENV}=1"
    try:
        _load_native_function()
        return True, None
    except Exception as exc:  # pragma: no cover - exercised in integration
        return False, str(exc)


def create_shards_native(
    fasta_path: str | Path,
    output_dir: str | Path,
    max_length: int,
    samples_per_shard: int,
    num_threads: int,
    use_bos_token: bool,
    progress_cb: Callable[[int, float, int, int], None] | None = None,
) -> NativeShardResult:
    if _is_disabled_by_env():
        raise NativeShardWriterError(
            f"Native shard writer disabled by env var {_DISABLE_NATIVE_ENV}=1"
        )
    func = _load_native_function()

    fasta = str(Path(fasta_path))
    out_dir = str(Path(output_dir))

    out_num_shards = ctypes.c_int(0)
    out_num_sequences = ctypes.c_longlong(0)
    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
    c_progress_cb = (
        _PROGRESS_CB(progress_cb)
        if progress_cb is not None
        else ctypes.cast(None, _PROGRESS_CB)
    )

    rc = func(
        fasta.encode("utf-8"),
        out_dir.encode("utf-8"),
        int(max_length),
        int(samples_per_shard),
        int(num_threads),
        int(bool(use_bos_token)),
        c_progress_cb,
        ctypes.byref(out_num_shards),
        ctypes.byref(out_num_sequences),
        error_buffer,
        ctypes.c_size_t(_ERROR_BUFFER_BYTES),
    )
    if rc != 0:
        error_text = error_buffer.value.decode("utf-8", errors="replace").strip()
        if not error_text:
            error_text = "Unknown native writer error"
        raise NativeShardWriterError(error_text)

    shard_paths = [
        Path(out_dir) / f"shard_{idx:04d}.bin"
        for idx in range(int(out_num_shards.value))
    ]
    missing = [str(path) for path in shard_paths if not path.exists()]
    if missing:
        raise NativeShardWriterError(
            "Native writer reported success but shard files are missing: "
            + ", ".join(missing[:5])
        )

    logger.info(
        "Native shard writer created %d shards (%d sequences).",
        out_num_shards.value,
        out_num_sequences.value,
    )

    return NativeShardResult(
        shard_paths=shard_paths,
        sequence_count=int(out_num_sequences.value),
    )
