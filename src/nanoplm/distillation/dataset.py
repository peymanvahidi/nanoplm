"""
Deprecation shim. KD dataset classes moved into nanoplm.data/.
"""
import warnings

warnings.warn(
    "Importing from nanoplm.distillation.dataset is deprecated. "
    "Use nanoplm.data.datasets.distill for LoadKDDataset/KDDatasetOnTheFly "
    "and nanoplm.data.sharding.distill_shard_writer for SaveKDDataset/shard_h5_file.",
    DeprecationWarning,
    stacklevel=2,
)

from nanoplm.data.datasets.distill import (  # noqa: F401
    KDDatasetOnTheFly,
    LoadKDDataset,
)
from nanoplm.data.sharding.distill_shard_writer import (  # noqa: F401
    SaveKDDataset,
    shard_h5_file,
)
