import pytest
import tempfile
import os
import torch
from pathlib import Path

from nanoplm.pretraining.dataset import FastaMLMDataset
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer


class TestFastaMLMDataset:
    """Test suite for FastaMLMDataset."""

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
        # Cleanup
        os.unlink(temp_path)

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer for testing."""
        return ProtModernBertTokenizer()

    def test_dataset_creation(self, temp_fasta_file, tokenizer):
        """Test basic dataset creation and properties."""
        dataset = FastaMLMDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=128,
            lazy=True
        )

        assert len(dataset) == 3, "Should have 3 sequences"
        assert dataset.max_length == 128, "Should store max_length"

    def test_dataset_sequence_access(self, temp_fasta_file, tokenizer):
        """Test accessing individual sequences."""
        dataset = FastaMLMDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=128,
            lazy=True
        )

        # Test first sequence
        sample = dataset[0]
        assert 'input_ids' in sample, "Should have input_ids"
        assert 'attention_mask' in sample, "Should have attention_mask"
        assert isinstance(sample['input_ids'], torch.Tensor), "input_ids should be tensor"
        assert isinstance(sample['attention_mask'], torch.Tensor), "attention_mask should be tensor"

        # Verify lengths match
        assert len(sample['input_ids']) == len(sample['attention_mask']), "Lengths should match"

        # Verify attention mask is correct (all 1s for non-padded sequence)
        expected_attention = torch.ones_like(sample['attention_mask'])
        assert torch.equal(sample['attention_mask'], expected_attention), "Attention mask should be all 1s"

    def test_dataset_truncation(self, temp_fasta_file, tokenizer):
        """Test sequence truncation when exceeding max_length."""
        # Create a very long sequence
        long_sequence = "M" + "A" * 200  # Very long protein sequence
        long_fasta = f">long_seq\n{long_sequence}"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(long_fasta)
            long_fasta_path = f.name

        try:
            dataset = FastaMLMDataset(
                fasta_path=long_fasta_path,
                tokenizer=tokenizer,
                max_length=50,  # Short max_length to force truncation
                lazy=True
            )

            sample = dataset[0]
            assert len(sample['input_ids']) <= 50, f"Should truncate to max_length, got {len(sample['input_ids'])}"
        finally:
            os.unlink(long_fasta_path)

    def test_dataset_out_of_bounds(self, temp_fasta_file, tokenizer):
        """Test error handling for out-of-bounds access."""
        dataset = FastaMLMDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=128,
            lazy=True
        )

        # Test negative index
        with pytest.raises(IndexError):
            dataset[-1]

        # Test index too large
        with pytest.raises(IndexError):
            dataset[len(dataset)]

        # Test large positive index
        with pytest.raises(IndexError):
            dataset[1000]

    def test_dataset_empty_fasta(self, tokenizer):
        """Test handling of empty FASTA file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write("")  # Empty file
            empty_path = f.name

        try:
            with pytest.raises(ValueError, match="FASTA file is empty"):
                FastaMLMDataset(
                    fasta_path=empty_path,
                    tokenizer=tokenizer,
                    max_length=128,
                    lazy=True
                )
        finally:
            os.unlink(empty_path)

    def test_dataset_special_tokens(self, tokenizer):
        """Test that special tokens are added correctly."""
        # Create a simple sequence
        simple_fasta = ">test\nMA"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(simple_fasta)
            simple_path = f.name

        try:
            dataset = FastaMLMDataset(
                fasta_path=simple_path,
                tokenizer=tokenizer,
                max_length=128,
                lazy=True
            )

            sample = dataset[0]

            # Should have at least 3 tokens: [CLS], sequence tokens, [SEP]
            # input_ids is now a 1D tensor after squeezing
            assert len(sample['input_ids']) >= 3, f"Should have special tokens, got {len(sample['input_ids'])}"

            # First token should be CLS (usually 1 or similar)
            # Last token should be SEP (usually 2 or similar)
            # This depends on the specific tokenizer implementation

        finally:
            os.unlink(simple_path)

    def test_dataset_index_persistence(self, temp_fasta_file, tokenizer):
        """Test that the SQLite index is created and reused."""
        dataset = FastaMLMDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=128,
            lazy=True
        )

        # Check that index file was created
        index_path = f"{temp_fasta_file}.idx"
        assert os.path.exists(index_path), "Index file should be created"

        # Create another dataset instance - should reuse the index
        dataset2 = FastaMLMDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=128,
            lazy=True
        )

        # Should work without recreating index
        assert len(dataset2) == len(dataset), "Should have same length"

        # Cleanup index file
        if os.path.exists(index_path):
            os.unlink(index_path)


class TestDatasetIntegration:
    """Integration tests combining dataset with other components."""

    @pytest.fixture
    def sample_fasta_content(self):
        """Create sample FASTA content for integration testing."""
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
        # Cleanup
        os.unlink(temp_path)

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer for testing."""
        return ProtModernBertTokenizer()

    def test_dataset_with_data_collator(self, temp_fasta_file, tokenizer):
        """Test dataset integration with DataCollatorForLanguageModeling."""
        from transformers import DataCollatorForLanguageModeling

        dataset = FastaMLMDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=64,
            lazy=True
        )

        # Get a batch of samples
        samples = [dataset[i] for i in range(len(dataset))]

        # Test collator
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15,
            pad_to_multiple_of=8  # Enable padding for batched tensors
        )

        batch = collator(samples)

        # Verify batch properties
        assert 'input_ids' in batch, "Batch should have input_ids"
        assert 'attention_mask' in batch, "Batch should have attention_mask"
        assert 'labels' in batch, "Batch should have labels for MLM"

        # Verify shapes
        assert batch['input_ids'].shape[0] == len(samples), "Should have correct batch size"
        assert batch['attention_mask'].shape[0] == len(samples), "Attention mask should match batch size"
        assert batch['labels'].shape[0] == len(samples), "Labels should match batch size"

        # All sequences should be padded to the same length
        assert batch['input_ids'].shape[1] == batch['attention_mask'].shape[1], "Input and attention should have same sequence length"
        assert batch['labels'].shape[1] == batch['input_ids'].shape[1], "Labels should match input_ids length"

    def test_dataset_iteration(self, temp_fasta_file, tokenizer):
        """Test iterating through the dataset."""
        dataset = FastaMLMDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=128,
            lazy=True
        )

        # Test iteration
        sequences = list(dataset)
        assert len(sequences) == len(dataset), "Iteration should yield all sequences"

        # Verify each sequence has required keys
        for seq in sequences:
            assert 'input_ids' in seq, "Each sequence should have input_ids"
            assert 'attention_mask' in seq, "Each sequence should have attention_mask"
            assert isinstance(seq['input_ids'], torch.Tensor), "input_ids should be tensor"
            assert isinstance(seq['attention_mask'], torch.Tensor), "attention_mask should be tensor"

    def test_dataset_lazy_vs_non_lazy(self, temp_fasta_file, tokenizer):
        """Test lazy vs non-lazy loading modes."""
        import tempfile
        
        # Test lazy mode (default)
        lazy_dataset = FastaMLMDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=128,
            lazy=True
        )

        # Create temporary directory for HDF5 shards
        with tempfile.TemporaryDirectory() as temp_hdf5_dir:
            # Test non-lazy mode
            non_lazy_dataset = FastaMLMDataset(
                fasta_path=temp_fasta_file,
                tokenizer=tokenizer,
                max_length=128,
                lazy=False,
                hdf5_dir=temp_hdf5_dir
            )

            # Both should have same length
            assert len(lazy_dataset) == len(non_lazy_dataset), "Datasets should have same length"

            # Non-lazy dataset should have HDF5 shards
            assert hasattr(non_lazy_dataset, 'shard_paths'), "Non-lazy dataset should have shard_paths"
            assert len(non_lazy_dataset.shard_paths) > 0, "Should have created HDF5 shards"

            # Lazy dataset should not have shard_paths
            assert not hasattr(lazy_dataset, 'shard_paths'), "Lazy dataset should not have shard_paths"

            # Results should be identical
            for i in range(len(lazy_dataset)):
                lazy_sample = lazy_dataset[i]
                non_lazy_sample = non_lazy_dataset[i]

                # Compare tensors
                assert torch.equal(lazy_sample['input_ids'], non_lazy_sample['input_ids']), f"input_ids should match for index {i}"
                assert torch.equal(lazy_sample['attention_mask'], non_lazy_sample['attention_mask']), f"attention_mask should match for index {i}"

    def test_dataset_memory_efficiency(self, temp_fasta_file, tokenizer):
        """Test that dataset doesn't load all sequences into memory."""
        dataset = FastaMLMDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=128,
            lazy=True
        )

        # Access all sequences
        for i in range(len(dataset)):
            seq = dataset[i]
            assert 'input_ids' in seq, f"Sequence {i} should have input_ids"

        # Dataset should not store sequences in memory
        # (This is a basic check - in practice we'd need memory profiling)
        assert not hasattr(dataset, '_sequences'), "Dataset should not cache sequences in memory"
