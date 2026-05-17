"""
Thread-safe file handle pool for HDF5 datasets.

Provides LRU-based file handle pooling to prevent "Too many open files" errors
when loading large sharded HDF5 datasets with PyTorch DataLoader.
"""

import resource
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Union

import h5py

from nanoplm.utils import logger


def detect_file_limits(num_workers: int = 4, reserve_percentage: float = 0.7) -> int:
    """
    Detect safe file handle limits based on system ulimit.

    Auto-detects the system's soft file descriptor limit and calculates
    a safe per-worker limit by reserving a percentage for application use.

    Args:
        num_workers: Number of DataLoader workers
        reserve_percentage: Percentage of soft limit to use (default: 70%)

    Returns:
        int: Maximum number of files to keep open per worker

    Example:
        >>> limit = detect_file_limits(num_workers=4)
        >>> print(f"Safe limit: {limit} files per worker")
    """
    try:
        # Query system ulimit for file descriptors
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

        # Cap at a reasonable maximum (some systems report unlimited as very large values)
        # 65536 is a common high limit on production systems
        MAX_REASONABLE_LIMIT = 65536
        effective_limit = min(soft_limit, MAX_REASONABLE_LIMIT)

        # Calculate usable limit (reserve percentage of soft limit)
        usable_limit = int(effective_limit * reserve_percentage)

        # Split across workers, ensuring at least 5 files per worker
        per_worker_limit = max(5, usable_limit // max(1, num_workers))

        logger.info(f"File descriptor limits:")
        logger.info(f"  System soft limit: {soft_limit}")
        logger.info(f"  System hard limit: {hard_limit}")
        logger.info(f"  Usable ({int(reserve_percentage * 100)}%): {usable_limit}")
        logger.info(f"  Per worker ({num_workers} workers): {per_worker_limit}")

        return per_worker_limit

    except Exception as e:
        logger.warning(f"Could not detect file limits: {e}. Using conservative default of 10.")
        return 10


class ThreadSafeFileHandlePool:
    """
    LRU cache for HDF5 file handles with thread-safety.

    Maintains a pool of open file handles with automatic LRU eviction when
    the maximum number of open files is reached. Thread-safe for use with
    PyTorch DataLoader's multi-worker data loading.

    The pool uses an OrderedDict to track access order, moving recently
    accessed files to the end and evicting from the front when full.

    Args:
        max_open_files: Maximum number of file handles to keep open simultaneously

    Example:
        >>> pool = ThreadSafeFileHandlePool(max_open_files=5)
        >>> file_handle = pool.get_file("data/shard_0000.h5")
        >>> data = file_handle['input_ids'][0]
        >>> pool.close_all()
    """

    def __init__(self, max_open_files: int = 10):
        self.max_open_files = max_open_files
        self._cache = OrderedDict()  # {path: h5py.File}
        self._lock = threading.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0

        logger.debug(f"Initialized ThreadSafeFileHandlePool with max_open_files={max_open_files}")

    def get_file(self, path: Union[str, Path]) -> h5py.File:
        """
        Get HDF5 file handle, opening if needed with LRU eviction.

        Thread-safe method to retrieve a file handle. If the file is already
        open, it's marked as recently used. If not, it's opened and added to
        the cache, potentially evicting the least recently used file.

        Args:
            path: Path to HDF5 file

        Returns:
            h5py.File: Open file handle in read mode

        Raises:
            OSError: If file cannot be opened
        """
        path = str(path)

        with self._lock:
            # Cache hit - file already open
            if path in self._cache:
                # Mark as recently used (move to end of OrderedDict)
                self._cache.move_to_end(path)
                self._hits += 1
                return self._cache[path]

            # Cache miss - need to open file
            self._misses += 1

            # Evict least recently used file if at capacity
            if len(self._cache) >= self.max_open_files:
                old_path, old_file = self._cache.popitem(last=False)
                try:
                    old_file.close()
                    logger.debug(f"LRU evicted: {Path(old_path).name}")
                except Exception as e:
                    logger.warning(f"Error closing {old_path}: {e}")

            # Open new file and add to cache
            try:
                new_file = h5py.File(path, 'r')
                self._cache[path] = new_file
                logger.debug(f"Opened file: {Path(path).name} (cache size: {len(self._cache)})")
                return new_file
            except Exception as e:
                raise OSError(f"Failed to open HDF5 file {path}: {e}")

    def close_all(self):
        """
        Close all cached file handles.

        Safely closes all open file handles in the pool. Handles exceptions
        gracefully to ensure all files are attempted to be closed even if
        some fail.
        """
        with self._lock:
            for path, file_handle in list(self._cache.items()):
                try:
                    file_handle.close()
                    logger.debug(f"Closed: {Path(path).name}")
                except Exception as e:
                    logger.warning(f"Error closing {path}: {e}")
            self._cache.clear()
            logger.debug("All cached files closed")

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            dict: Statistics including:
                - hits: Number of cache hits
                - misses: Number of cache misses
                - hit_rate: Cache hit rate (0.0 to 1.0)
                - open_files: Number of currently open files
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'open_files': len(self._cache),
        }

    def __getstate__(self):
        """
        Prepare state for pickling (DataLoader multiprocessing).

        File handles cannot be pickled, so we clear the cache and set the
        lock to None. The pool will be recreated per-worker via worker_init_fn.
        """
        state = self.__dict__.copy()
        # Remove unpicklable objects
        state['_cache'] = OrderedDict()  # Don't pickle file handles
        state['_lock'] = None  # Locks can't be pickled
        return state

    def __setstate__(self, state):
        """
        Restore state after unpickling.

        Recreates the lock and initializes an empty cache. Per-worker pools
        should be created via worker_init_fn, not through unpickling.
        """
        self.__dict__.update(state)
        # Recreate lock and empty cache
        self._lock = threading.Lock()
        self._cache = OrderedDict()
