import logging
import os
import sys
from datetime import datetime
from pathlib import Path


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
        log_dir = Path("output") / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.baseFilename = str(log_dir / f"nanoplm_{ts}_{os.getpid()}.log")
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


# File handler (lazy)
file_handler = _LazyFileHandler()
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
file_handler._nanoplm_role = "file"  # sentinel for idempotency check across re-imports

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(message)s"))
console_handler._nanoplm_role = "console"  # sentinel for idempotency check across re-imports

logger = logging.getLogger("nanoplm")
logger.setLevel(logging.INFO)
# Idempotent: avoid accumulating duplicate handlers across re-imports.
# (logging.getLogger returns a process-global singleton; isinstance checks
# cannot be used here because sys.modules.pop re-creates the class object,
# making old instances appear as a different type to the new import.)
# For the file handler, replace any existing one so each import starts fresh
# (the lazy handler is uninitialized and will resolve its path on first emit).
for _h in list(logger.handlers):
    if getattr(_h, "_nanoplm_role", None) == "file":
        logger.removeHandler(_h)
        break
logger.addHandler(file_handler)
if not any(getattr(h, "_nanoplm_role", None) == "console" for h in logger.handlers):
    logger.addHandler(console_handler)
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
