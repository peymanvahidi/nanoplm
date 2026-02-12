import pytest
import tempfile
import os
import torch
import numpy as np
from pathlib import Path

from nanoplm.pretraining.dataset import (
    LazyFastaDataset,
    ShardWriter,
    ShardedDataset,
)
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer


class TestLazyFastaDataset:
    """Test suite for LazyFastaDataset."""

    @pytest.fixture
    def sample_fasta_content(self):
        """Create sample FASTA content for testing."""
        return """>seq1
MKALCLLLLPVLGLLTGSSGSGSGSGSGS
>seq2
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR
>seq3
MAIGT
"""

    @pytest.fixture
    def temp_fasta_file(self, sample_fasta_content):
        """Create a temporary FASTA file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(sample_fasta_content)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer for testing."""
        return ProtModernBertTokenizer()

    def test_dataset_creation(self, temp_fasta_file, tokenizer):
        """Test basic dataset creation and properties."""
        dataset = LazyFastaDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=128,
        )

        assert len(dataset) == 3, "Should have 3 sequences"
        assert dataset.max_length == 128, "Should store max_length"

    def test_dataset_getitem(self, temp_fasta_file, tokenizer):
        """Test accessing individual sequences."""
        dataset = LazyFastaDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=128,
        )

        sample = dataset[0]
        assert 'input_ids' in sample, "Should have input_ids"
        assert 'attention_mask' in sample, "Should have attention_mask"
        assert isinstance(sample['input_ids'], torch.Tensor), "input_ids should be tensor"
        assert isinstance(sample['attention_mask'], torch.Tensor), "attention_mask should be tensor"
        assert len(sample['input_ids']) == len(sample['attention_mask']), "Lengths should match"

    def test_dataset_out_of_bounds(self, temp_fasta_file, tokenizer):
        """Test error handling for out-of-bounds access."""
        dataset = LazyFastaDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=128,
        )

        with pytest.raises(IndexError):
            dataset[-1]

        with pytest.raises(IndexError):
            dataset[len(dataset)]

        with pytest.raises(IndexError):
            dataset[1000]

    def test_dataset_empty_fasta(self, tokenizer):
        """Test handling of empty FASTA file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write("")
            empty_path = f.name

        try:
            with pytest.raises(ValueError, match="FASTA file is empty"):
                LazyFastaDataset(
                    fasta_path=empty_path,
                    tokenizer=tokenizer,
                    max_length=128,
                )
        finally:
            os.unlink(empty_path)


class TestBinaryShardCreationAndLoading:
    """End-to-end tests for binary shard creation and loading."""

    @pytest.fixture
    def sample_fasta_content(self):
        """Create sample FASTA content for testing."""
        return """>seq1
MKALCLLLLPVLGLLTGSSGS
>seq2
MVLSPADKTNVKAAWGKVGAH
>seq3
MAIGTMAIGTMAIGT
"""

    @pytest.fixture
    def temp_fasta_file(self, sample_fasta_content):
        """Create a temporary FASTA file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(sample_fasta_content)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer for testing."""
        return ProtModernBertTokenizer()

    def test_create_and_load_shards(self, temp_fasta_file, tokenizer):
        """Test creating binary shards then loading them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = ShardWriter(
                fasta_path=temp_fasta_file,
                tokenizer=tokenizer,
                max_length=128,
                output_dir=tmpdir,
                samples_per_shard=2,
                max_workers=1,
            )
            shard_paths = saver.create_shards()

            # Should create 2 shards (3 seqs / 2 per shard = 2)
            assert len(shard_paths) == 2

            # Verify .bin and .idx files exist
            for sp in shard_paths:
                assert sp.exists(), f"Bin file should exist: {sp}"
                idx_file = sp.with_name(sp.stem + ".idx.npy")
                assert idx_file.exists(), f"Idx file should exist: {idx_file}"

            # Load the shards
            dataset = ShardedDataset(data_dir=tmpdir)
            assert len(dataset) == 3

            # Access each sample
            for i in range(len(dataset)):
                sample = dataset[i]
                assert 'input_ids' in sample
                assert 'attention_mask' in sample
                assert isinstance(sample['input_ids'], torch.Tensor)
                assert len(sample['input_ids']) > 0

    def test_force_overwrite(self, temp_fasta_file, tokenizer):
        """Test force overwrite of existing shards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = ShardWriter(
                fasta_path=temp_fasta_file,
                tokenizer=tokenizer,
                max_length=128,
                output_dir=tmpdir,
                samples_per_shard=10,
                max_workers=1,
            )
            saver.create_shards()

            # Without force, should raise
            with pytest.raises(FileExistsError):
                saver2 = ShardWriter(
                    fasta_path=temp_fasta_file,
                    tokenizer=tokenizer,
                    max_length=128,
                    output_dir=tmpdir,
                    samples_per_shard=10,
                    max_workers=1,
                    force=False,
                )
                saver2.create_shards()

            # With force, should succeed
            saver3 = ShardWriter(
                fasta_path=temp_fasta_file,
                tokenizer=tokenizer,
                max_length=128,
                output_dir=tmpdir,
                samples_per_shard=10,
                max_workers=1,
                force=True,
            )
            saver3.create_shards()

    def test_missing_directory(self):
        """Test loading from non-existent directory."""
        with pytest.raises(FileNotFoundError):
            ShardedDataset(data_dir="/nonexistent/path")

    def test_empty_directory(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No binary shard files"):
                ShardedDataset(data_dir=tmpdir)

    def test_out_of_bounds(self, temp_fasta_file, tokenizer):
        """Test out-of-bounds access on loaded shards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = ShardWriter(
                fasta_path=temp_fasta_file,
                tokenizer=tokenizer,
                max_length=128,
                output_dir=tmpdir,
                samples_per_shard=10,
                max_workers=1,
            )
            saver.create_shards()
            dataset = ShardedDataset(data_dir=tmpdir)

            with pytest.raises(IndexError):
                dataset[-1]

            with pytest.raises(IndexError):
                dataset[len(dataset)]

    def test_consistency_with_lazy_dataset(self, temp_fasta_file, tokenizer):
        """Test that binary shards produce the same tokens as lazy dataset."""
        lazy_ds = LazyFastaDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=128,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            saver = ShardWriter(
                fasta_path=temp_fasta_file,
                tokenizer=tokenizer,
                max_length=128,
                output_dir=tmpdir,
                samples_per_shard=10,
                max_workers=1,
            )
            saver.create_shards()
            shard_ds = ShardedDataset(data_dir=tmpdir)

            assert len(lazy_ds) == len(shard_ds), "Datasets should have same length"

            for i in range(len(lazy_ds)):
                lazy_sample = lazy_ds[i]
                shard_sample = shard_ds[i]

                # Compare token values (shard stores uint8, lazy returns pt tensors)
                assert torch.equal(
                    lazy_sample['input_ids'].long(),
                    shard_sample['input_ids'].long(),
                ), f"input_ids should match for index {i}"
