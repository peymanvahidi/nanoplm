import numpy as np
import pytest

from nanoplm.data.datasets.pretrain import ShardedDataset
from nanoplm.data.infra.validation import ValidationError


def _write_shard(dirpath, name, sizes, bin_bytes=None):
    dirpath.mkdir(parents=True, exist_ok=True)
    sizes = np.asarray(sizes, dtype=np.int32)
    np.save(dirpath / f"{name}.idx.npy", sizes)
    if bin_bytes is None:
        bin_bytes = bytes(range(1, int(sizes.sum()) + 1))
    (dirpath / f"{name}.bin").write_bytes(bin_bytes)


def test_matching_bin_and_idx_construct_and_read(tmp_path):
    _write_shard(tmp_path, "shard_0000", [3, 4, 5])

    ds = ShardedDataset(str(tmp_path))

    assert len(ds) == 3
    assert [len(ds[i]["input_ids"]) for i in range(3)] == [3, 4, 5]


def test_truncated_bin_raises(tmp_path):
    _write_shard(tmp_path, "shard_0000", [10, 10, 10], bin_bytes=bytes(range(1, 16)))

    with pytest.raises(ValidationError, match=r"shard_0000\.bin.*30 bytes.*15 bytes"):
        ShardedDataset(str(tmp_path))


def test_empty_bin_raises(tmp_path):
    _write_shard(tmp_path, "shard_0000", [5], bin_bytes=b"")

    with pytest.raises(ValidationError, match=r"shard_0000\.bin.*5 bytes.*0 bytes"):
        ShardedDataset(str(tmp_path))


def test_oversized_bin_raises(tmp_path):
    _write_shard(tmp_path, "shard_0000", [5, 5], bin_bytes=bytes(range(1, 13)))

    with pytest.raises(ValidationError, match=r"shard_0000\.bin.*10 bytes.*12 bytes"):
        ShardedDataset(str(tmp_path))


def test_second_shard_mismatch_is_still_caught(tmp_path):
    _write_shard(tmp_path, "shard_0000", [4, 4])
    _write_shard(tmp_path, "shard_0001", [6, 6], bin_bytes=bytes(range(1, 10)))

    with pytest.raises(ValidationError, match=r"shard_0001\.bin"):
        ShardedDataset(str(tmp_path))
