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
        self.encoding = None
        self.delay = False
        self.errors = None
        self.stream = open(self.baseFilename, "a", encoding=None)
        self._initialized = True

    def emit(self, record):
        if not self._initialized:
            self._initialize()
        super().emit(record)


# File handler (lazy)
file_handler = _LazyFileHandler()
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(message)s"))

logger = logging.getLogger("nanoplm")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
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
