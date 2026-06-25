"""
Deprecation shim. The dataset classes and ShardWriter moved into nanoplm.data/.
This file is kept for one release for backwards compatibility.
"""
import warnings

warnings.warn(
    "Importing from nanoplm.pretraining.dataset is deprecated. "
    "Use nanoplm.data.datasets.pretrain for ShardedDataset/TokenPackingDataset "
    "and nanoplm.data.sharding.pretrain_shard_writer for ShardWriter.",
    DeprecationWarning,
    stacklevel=2,
)

from nanoplm.data.datasets.pretrain import (  # noqa: F401
    ShardedDataset,
    TokenPackingDataset,
)
from nanoplm.data.sharding.pretrain_shard_writer import (  # noqa: F401
    ShardWriter,
)
