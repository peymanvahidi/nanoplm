"""
Tests for data loading infrastructure.

Tests cover:
1. File handle pool (LRU eviction, thread safety, pickling)
2. Manifest validation (missing files, corrupt data, clear errors)
3. Dataset classes (pretraining and distillation with pooling)
4. DataLoader integration (multi-worker, per-worker pools)
"""

import pytest
import tempfile
import h5py
import numpy as np
import yaml
from pathlib import Path
from unittest.mock import patch


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def pretrain_shards(temp_dir):
    """Create pretraining-style HDF5 shards."""
    shard_dir = temp_dir / 'pretrain_shards'
    shard_dir.mkdir()

    for i in range(5):
        with h5py.File(shard_dir / f'shard_{i:04d}.h5', 'w') as f:
            dt = h5py.special_dtype(vlen=np.uint8)
            ds = f.create_dataset('input_ids', (20,), dtype=dt)
            for j in range(20):
                ds[j] = np.random.randint(1, 32, size=50, dtype=np.uint8)

    return shard_dir


@pytest.fixture
def distill_shards(temp_dir):
    """Create distillation-style HDF5 shards.

    Returns a file prefix path (e.g., /path/to/data.h5) that the dataset
    will use to find shards matching pattern: data_shard_*.h5
    """
    shard_dir = temp_dir / 'distill_shards'
    shard_dir.mkdir()

    for i in range(5):
        with h5py.File(shard_dir / f'data_shard_{i}.h5', 'w') as f:
            for j in range(20):
                grp = f.create_group(str(j))
                grp.create_dataset('input_ids', data=np.arange(50, dtype=np.int8))
                grp.create_dataset('attention_mask', data=np.ones(50, dtype=np.int8))
                grp.create_dataset('teacher_embeddings',
                                   data=np.random.randn(50, 1024).astype(np.float16))

    # Return file prefix path (dataset looks for {prefix.stem}_shard_*.h5)
    return shard_dir / 'data.h5'


@pytest.fixture
def pretrain_manifest(temp_dir, pretrain_shards):
    """Create a valid pretraining manifest."""
    manifest_path = pretrain_shards / '.data_manifest'
    manifest = {
        'pipeline_mode': 'pretrain',
        'seqs_num': 100,
        'min_seq_len': 10,
        'max_seq_len': 512,
        'val_ratio': 0.1,
        'train_dir': 'train',
        'val_dir': 'val',
        'train_sequences': 90,
        'val_sequences': 10,
        'sharded': True,
        'samples_per_shard': 20,
    }
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest, f)
    return manifest_path


# ==============================================================================
# File Pool Tests
# ==============================================================================

class TestFileHandlePool:
    """Tests for ThreadSafeFileHandlePool."""

    def test_detect_file_limits(self):
        """Test that file limit detection returns reasonable values."""
        from nanoplm.data.file_pool import detect_file_limits

        limit = detect_file_limits(num_workers=4)
        assert limit >= 5, "Should return at least 5 files per worker"
        # With cap at 65536 and 70% reserve, max per worker is ~11468 for 4 workers
        assert limit <= 20000, "Sanity check: limit should be reasonable"

    def test_lru_eviction(self, temp_dir):
        """Test that LRU eviction works when pool is full."""
        from nanoplm.data.file_pool import ThreadSafeFileHandlePool

        # Create test files
        for i in range(5):
            with h5py.File(temp_dir / f'test_{i}.h5', 'w') as f:
                f.create_dataset('data', data=np.arange(10))

        pool = ThreadSafeFileHandlePool(max_open_files=3)

        # Open 5 files with pool that only allows 3
        for i in range(5):
            pool.get_file(temp_dir / f'test_{i}.h5')

        stats = pool.get_stats()
        assert stats['open_files'] <= 3, "LRU should evict when pool is full"
        assert stats['misses'] == 5, "All files should be cache misses initially"

        pool.close_all()

    def test_cache_hit(self, temp_dir):
        """Test that repeated access uses cache."""
        from nanoplm.data.file_pool import ThreadSafeFileHandlePool

        with h5py.File(temp_dir / 'test.h5', 'w') as f:
            f.create_dataset('data', data=np.arange(10))

        pool = ThreadSafeFileHandlePool(max_open_files=3)

        # Access same file twice
        f1 = pool.get_file(temp_dir / 'test.h5')
        f2 = pool.get_file(temp_dir / 'test.h5')

        assert f1 is f2, "Same file should return cached handle"
        stats = pool.get_stats()
        assert stats['hits'] == 1, "Second access should be cache hit"
        assert stats['misses'] == 1, "First access should be cache miss"

        pool.close_all()

    def test_pickle_support(self, temp_dir):
        """Test that pool can be pickled (for DataLoader workers)."""
        import pickle
        from nanoplm.data.file_pool import ThreadSafeFileHandlePool

        with h5py.File(temp_dir / 'test.h5', 'w') as f:
            f.create_dataset('data', data=np.arange(10))

        pool = ThreadSafeFileHandlePool(max_open_files=3)
        pool.get_file(temp_dir / 'test.h5')

        # Pickle and unpickle
        pickled = pickle.dumps(pool)
        restored = pickle.loads(pickled)

        # Restored pool should be empty (file handles can't be pickled)
        stats = restored.get_stats()
        assert stats['open_files'] == 0, "Restored pool should have no open files"


# ==============================================================================
# Validation Tests
# ==============================================================================

class TestValidation:
    """Tests for manifest and shard validation."""

    def test_missing_manifest_error(self, temp_dir):
        """Test that missing manifest gives helpful error."""
        from nanoplm.data.validation import validate_dataset_manifest

        with pytest.raises(FileNotFoundError) as exc_info:
            validate_dataset_manifest(temp_dir / '.data_manifest')

        assert 'nanoplm data from-yaml' in str(exc_info.value)

    def test_valid_manifest_parsing(self, pretrain_manifest):
        """Test that valid manifest is parsed correctly."""
        from nanoplm.data.validation import validate_dataset_manifest

        result = validate_dataset_manifest(pretrain_manifest)
        assert result['pipeline_mode'] == 'pretrain'
        assert result['samples_per_shard'] == 20

    def test_shard_validation_success(self, pretrain_shards):
        """Test that valid shards pass validation."""
        from nanoplm.data.validation import validate_shard_files

        paths = validate_shard_files(pretrain_shards)
        assert len(paths) == 5

    def test_shard_validation_missing_file(self, pretrain_shards):
        """Test that missing shard is detected."""
        from nanoplm.data.validation import validate_shard_files, ValidationError

        # Remove one shard
        (pretrain_shards / 'shard_0002.h5').unlink()

        # Should still work without expected_count
        paths = validate_shard_files(pretrain_shards)
        assert len(paths) == 4

        # But fail with expected_count
        with pytest.raises(ValidationError):
            validate_shard_files(pretrain_shards, expected_count=5)

    def test_shard_validation_empty_dataset(self, temp_dir):
        """Test that empty shard is detected."""
        from nanoplm.data.validation import validate_shard_files, ValidationError

        # Create shard with empty dataset
        with h5py.File(temp_dir / 'shard_0000.h5', 'w') as f:
            dt = h5py.special_dtype(vlen=np.uint8)
            f.create_dataset('input_ids', shape=(0,), dtype=dt)

        with pytest.raises(ValidationError) as exc_info:
            validate_shard_files(temp_dir)

        assert 'empty' in str(exc_info.value).lower() or '0' in str(exc_info.value)


# ==============================================================================
# Dataset Tests
# ==============================================================================

class TestPretrainingDataset:
    """Tests for LoadShardedFastaMLMDataset."""

    def test_dataset_creation(self, pretrain_shards):
        """Test that dataset can be created."""
        from nanoplm.pretraining.dataset import LoadShardedFastaMLMDataset

        dataset = LoadShardedFastaMLMDataset(str(pretrain_shards), max_open_files=3)
        assert len(dataset) == 100  # 5 shards * 20 samples

    def test_item_access(self, pretrain_shards):
        """Test that items can be accessed."""
        from nanoplm.pretraining.dataset import LoadShardedFastaMLMDataset

        dataset = LoadShardedFastaMLMDataset(str(pretrain_shards))
        item = dataset[0]

        assert 'input_ids' in item
        assert 'attention_mask' in item

    def test_lru_pooling(self, pretrain_shards):
        """Test that LRU pooling is used."""
        from nanoplm.pretraining.dataset import LoadShardedFastaMLMDataset

        dataset = LoadShardedFastaMLMDataset(str(pretrain_shards), max_open_files=2)

        # Access items from different shards
        for i in [0, 25, 50, 75, 10]:
            dataset[i]

        pool = dataset._get_worker_pool()
        stats = pool.get_stats()
        assert stats['open_files'] <= 2

    def test_pickle_support(self, pretrain_shards):
        """Test that dataset can be pickled."""
        import pickle
        from nanoplm.pretraining.dataset import LoadShardedFastaMLMDataset

        dataset = LoadShardedFastaMLMDataset(str(pretrain_shards))
        dataset[0]  # Create pool

        pickled = pickle.dumps(dataset)
        restored = pickle.loads(pickled)

        # Should work after unpickling
        item = restored[0]
        assert 'input_ids' in item


class TestDistillationDataset:
    """Tests for LoadKDDataset."""

    def test_dataset_creation(self, distill_shards):
        """Test that dataset can be created."""
        from nanoplm.distillation.dataset import LoadKDDataset

        dataset = LoadKDDataset(
            h5_path=str(distill_shards),
            device='cpu',
            max_open_files=3,
            sharded=True,
        )
        assert len(dataset) == 100  # 5 shards * 20 samples

    def test_item_access(self, distill_shards):
        """Test that items can be accessed."""
        from nanoplm.distillation.dataset import LoadKDDataset

        dataset = LoadKDDataset(
            h5_path=str(distill_shards),
            device='cpu',
            sharded=True,
        )
        item = dataset[0]

        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'teacher_embeddings' in item


# ==============================================================================
# DataLoader Integration Tests
# ==============================================================================

class TestDataLoaderIntegration:
    """Tests for DataLoader integration with worker_init_fn."""

    def test_single_worker(self, pretrain_shards):
        """Test DataLoader with single worker."""
        import torch
        from torch.utils.data import DataLoader
        from nanoplm.pretraining.dataset import LoadShardedFastaMLMDataset

        dataset = LoadShardedFastaMLMDataset(str(pretrain_shards))
        loader = DataLoader(dataset, batch_size=8, num_workers=0)

        batch = next(iter(loader))
        assert batch['input_ids'].shape[0] == 8

    def test_multi_worker(self, pretrain_shards):
        """Test DataLoader with multiple workers."""
        import torch
        from torch.utils.data import DataLoader
        from nanoplm.pretraining.dataset import (
            LoadShardedFastaMLMDataset,
            get_pretraining_worker_init_fn
        )

        dataset = LoadShardedFastaMLMDataset(str(pretrain_shards), max_open_files=2)
        worker_init = get_pretraining_worker_init_fn(dataset)

        loader = DataLoader(
            dataset,
            batch_size=8,
            num_workers=2,
            worker_init_fn=worker_init,
            persistent_workers=False
        )

        batches = list(loader)
        total = sum(b['input_ids'].shape[0] for b in batches)
        assert total == 100


# ==============================================================================
# Stress Tests (Optional - can be slow)
# ==============================================================================

@pytest.mark.slow
class TestStress:
    """Stress tests with many shards."""

    def test_many_shards(self, temp_dir):
        """Test with 100+ shards to verify no file handle exhaustion."""
        from nanoplm.pretraining.dataset import LoadShardedFastaMLMDataset

        # Create 100 shards
        shard_dir = temp_dir / 'many_shards'
        shard_dir.mkdir()

        for i in range(100):
            with h5py.File(shard_dir / f'shard_{i:04d}.h5', 'w') as f:
                dt = h5py.special_dtype(vlen=np.uint8)
                ds = f.create_dataset('input_ids', (10,), dtype=dt)
                for j in range(10):
                    ds[j] = np.arange(20, dtype=np.uint8)

        # Should not raise "Too many open files"
        dataset = LoadShardedFastaMLMDataset(str(shard_dir), max_open_files=10)

        # Access items from various shards
        for i in range(0, 1000, 50):  # Jump around shards
            dataset[i]

        pool = dataset._get_worker_pool()
        stats = pool.get_stats()
        assert stats['open_files'] <= 10, "Should stay within limit with 100 shards"
