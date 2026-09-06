import logging
import os
import sys
from datetime import datetime
from pathlib import Path

_RANK = int(os.environ.get("RANK") or 0)
_LOG_ALL_RANKS = os.environ.get("NANOPLM_LOG_ALL_RANKS", "").strip().lower() not in ("", "0", "false", "no")
_SPEAKS = _RANK == 0 or _LOG_ALL_RANKS
_RANK_TAG = "" if _RANK == 0 else f"[rank {_RANK}] "


class _LazyFileHandler(logging.FileHandler):
    """FileHandler that defers opening the file (and creating its directory)
    until the first emit() call. Avoids creating output/logs/ at import time.
    """

    def __init__(self):
        # Defer parent __init__ until first emit
        logging.Handler.__init__(self)
        self.baseFilename = None
        self.stream = None
        self._initialized = False

    def _initialize(self) -> None:
        directory = Path(os.environ.get("NANOPLM_LOG_DIR") or "output/logs")
        directory.mkdir(parents=True, exist_ok=True)
        job = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_rank{_RANK}" if _LOG_ALL_RANKS else ""
        self.baseFilename = str(directory / f"nanoplm_{job}{suffix}.log")
        self.mode = "a"
        self.encoding = "utf-8"
        self.delay = False
        self.errors = None
        self.stream = open(self.baseFilename, "a", encoding="utf-8")
        self._initialized = True

    def emit(self, record):
        if not self._initialized:
            self._initialize()
        super().emit(record)


class _RankFilter(logging.Filter):
    # Lives on the handler, not the logger: records from child loggers such as
    # nanoplm.collator reach us by propagation, which runs handler filters only.
    def filter(self, record) -> bool:
        if _SPEAKS:
            return True
        return record.levelno >= logging.WARNING


file_handler = _LazyFileHandler()
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
file_handler._nanoplm_role = "file"

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter(
        f"%(asctime)s %(levelname)-7s {_RANK_TAG}%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
)
console_handler.addFilter(_RankFilter())
console_handler._nanoplm_role = "console"

logger = logging.getLogger("nanoplm")
logger.setLevel(logging.INFO)
# logging.getLogger returns a process-global singleton, so re-imports would otherwise
# accumulate handlers. isinstance cannot identify ours: sys.modules.pop re-creates the
# class object, making handlers from an earlier import look like a different type.
for _h in list(logger.handlers):
    if getattr(_h, "_nanoplm_role", None) in ("file", "console"):
        logger.removeHandler(_h)
logger.addHandler(console_handler)
if _SPEAKS:
    logger.addHandler(file_handler)
logger.propagate = False


def log_stage(stage_name: str) -> None:
    separator = "=" * 20
    logger.info(f"{separator}STAGE: {stage_name}{separator}")


def set_log_level(level: str) -> None:
    """Set the logger level by string ('debug'|'info'|'warning'|'error')."""
    numeric = getattr(logging, level.upper(), None)
    if numeric is None:
        raise ValueError(f"Unknown log level: {level}")
    logger.setLevel(numeric)
    for h in logger.handlers:
        h.setLevel(numeric)
