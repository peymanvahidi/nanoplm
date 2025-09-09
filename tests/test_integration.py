import pytest
import tempfile
import os
import torch
from transformers import DataCollatorForLanguageModeling

from nanoplm.pretraining.dataset import FastaMLMDataset
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer


class TestFullPipelineIntegration:
    """Integration tests for the complete pretraining pipeline."""

    @pytest.fixture
    def sample_fasta_content(self):
        """Create sample FASTA content for testing."""
        return """>protein1
MKALCLLLLPVLGLLTGSSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGS
>protein2
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLS
>protein3
MAIGTMAIGTMAIGTMAIGTMAIGTMAIGTMAIGTMAIGTMAIGTMAIGT
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

    @pytest.fixture
    def small_model(self, tokenizer):
        """Create a small model for testing."""
        model = ProtModernBertMLM(
            hidden_size=128,  # Small for testing
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=tokenizer.vocab_size,
            mlp_activation="swiglu",
            mlp_dropout=0.0,
            mlp_bias=False,
            attention_bias=False,
            attention_dropout=0.0,
            classifier_activation="gelu"
        )
        return model

    def test_full_pipeline_forward_pass(self, temp_fasta_file, tokenizer, small_model):
        """Test complete forward pass through dataset -> collator -> model."""
        # Create dataset
        dataset = FastaMLMDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=64
        )

        # Get batch from dataset
        batch_size = min(3, len(dataset))
        samples = [dataset[i] for i in range(batch_size)]

        # Create collator
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        # Create batch
        batch = collator(samples)

        # Test model forward pass
        small_model.eval()
        with torch.no_grad():
            outputs = small_model(**batch)

        # Verify outputs
        assert 'loss' in outputs, "Should have loss for MLM task"
        assert 'logits' in outputs, "Should have logits"
        assert outputs['logits'].shape[0] == batch_size, "Batch size should match"
        assert outputs['logits'].shape[2] == tokenizer.vocab_size, "Vocab size should match"

        # Loss should be a scalar tensor
        assert outputs['loss'].dim() == 0, "Loss should be scalar"
        assert outputs['loss'].item() > 0, "Loss should be positive"

    def test_modernbert_unpadding_mechanism(self, temp_fasta_file, tokenizer, small_model):
        """Test that ModernBERT's unpadding mechanism works correctly."""
        # Create dataset with different sequence lengths
        dataset = FastaMLMDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=64
        )

        # Get samples and verify different lengths
        samples = [dataset[i] for i in range(len(dataset))]
        lengths = [len(sample['input_ids']) for sample in samples]
        assert len(set(lengths)) > 1, "Should have different sequence lengths"

        # Create batch with padding
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.0  # No masking for this test
        )
        batch = collator(samples)

        # Verify padding occurred
        max_len = batch['input_ids'].shape[1]
        assert all(length < max_len for length in lengths), "Should have padding"

        # Test model can handle the padded batch
        small_model.eval()
        with torch.no_grad():
            outputs = small_model(**batch)

        # Should not crash and should produce outputs
        assert outputs['logits'].shape[0] == len(samples), "Should handle all samples"

    def test_batch_training_step(self, temp_fasta_file, tokenizer, small_model):
        """Test a complete training step simulation."""
        # Create dataset and collator
        dataset = FastaMLMDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=64
        )

        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        # Simulate getting a batch
        samples = [dataset[i] for i in range(min(3, len(dataset)))]
        batch = collator(samples)

        # Set model to training mode
        small_model.train()

        # Forward pass
        outputs = small_model(**batch)
        loss = outputs['loss']

        # Backward pass (simulation)
        loss.backward()

        # Check gradients exist
        has_gradients = any(p.grad is not None for p in small_model.parameters())
        assert has_gradients, "Should have gradients after backward pass"

        # Clear gradients
        small_model.zero_grad()

    def test_different_batch_sizes(self, temp_fasta_file, tokenizer, small_model):
        """Test model handles different batch sizes correctly."""
        dataset = FastaMLMDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=64
        )

        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.1
        )

        small_model.eval()

        # Test different batch sizes
        for batch_size in [1, 2, 3]:
            if batch_size > len(dataset):
                continue

            samples = [dataset[i] for i in range(batch_size)]
            batch = collator(samples)

            with torch.no_grad():
                outputs = small_model(**batch)

            assert outputs['logits'].shape[0] == batch_size, f"Should handle batch_size={batch_size}"

    def test_gradient_accumulation_simulation(self, temp_fasta_file, tokenizer, small_model):
        """Test gradient accumulation simulation."""
        dataset = FastaMLMDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=64
        )

        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        small_model.train()

        # Simulate gradient accumulation
        accumulation_steps = 2
        accumulated_loss = 0.0

        for step in range(accumulation_steps):
            # Get batch
            batch_size = min(2, len(dataset))
            samples = [dataset[i % len(dataset)] for i in range(batch_size)]
            batch = collator(samples)

            # Forward pass
            outputs = small_model(**batch)
            loss = outputs['loss'] / accumulation_steps  # Normalize for accumulation

            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()

        # Check accumulated gradients
        total_grad_norm = 0.0
        param_count = 0
        for param in small_model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
                param_count += 1

        total_grad_norm = total_grad_norm ** 0.5
        assert total_grad_norm > 0, "Should have accumulated gradients"
        assert param_count > 0, "Should have parameters with gradients"

        # Clear gradients
        small_model.zero_grad()

    def test_memory_efficiency(self, temp_fasta_file, tokenizer, small_model):
        """Test memory efficiency of the pipeline."""
        import gc

        dataset = FastaMLMDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=64
        )

        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.1
        )

        small_model.eval()

        # Get initial memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = 0

        # Process multiple batches
        for i in range(min(5, len(dataset))):
            samples = [dataset[i]]
            batch = collator(samples)

            with torch.no_grad():
                outputs = small_model(**batch)

            # Force cleanup
            del batch, outputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Check that memory usage is reasonable
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_increase = final_memory - initial_memory
            # Should not have excessive memory growth
            assert memory_increase < 100 * 1024 * 1024, f"Memory increase too large: {memory_increase} bytes"


class TestModernBertCompatibility:
    """Tests specifically for ModernBERT compatibility."""

    @pytest.fixture
    def simple_fasta(self):
        """Create simple FASTA for compatibility testing."""
        return """>seq1
MKALCL
>seq2
MVLSPA
"""

    @pytest.fixture
    def temp_fasta_file(self, simple_fasta):
        """Create temporary FASTA file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(simple_fasta)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer."""
        return ProtModernBertTokenizer()

    @pytest.fixture
    def tiny_model(self, tokenizer):
        """Create tiny model for fast testing."""
        model = ProtModernBertMLM(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=tokenizer.vocab_size,
            mlp_activation="swiglu",
            mlp_dropout=0.0,
            mlp_bias=False,
            attention_bias=False,
            attention_dropout=0.0,
            classifier_activation="gelu"
        )
        return model

    def test_modernbert_attention_mask_handling(self, temp_fasta_file, tokenizer, tiny_model):
        """Test that attention masks are handled correctly by ModernBERT."""
        dataset = FastaMLMDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=32
        )

        samples = [dataset[i] for i in range(len(dataset))]
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.0  # No masking for this test
        )
        batch = collator(samples)

        tiny_model.eval()
        with torch.no_grad():
            # Test with attention_mask
            outputs_with_mask = tiny_model(**batch)

            # Test without attention_mask (should still work)
            batch_no_mask = {k: v for k, v in batch.items() if k != 'attention_mask'}
            outputs_no_mask = tiny_model(**batch_no_mask)

        # Both should produce outputs
        assert 'logits' in outputs_with_mask, "Should work with attention mask"
        assert 'logits' in outputs_no_mask, "Should work without attention mask"

        # Shapes should be the same
        assert outputs_with_mask['logits'].shape == outputs_no_mask['logits'].shape, "Shapes should match"

    def test_modernbert_position_ids(self, temp_fasta_file, tokenizer, tiny_model):
        """Test position_ids handling."""
        dataset = FastaMLMDataset(
            fasta_path=temp_fasta_file,
            tokenizer=tokenizer,
            max_length=32
        )

        samples = [dataset[i] for i in range(len(dataset))]
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.0
        )
        batch = collator(samples)

        tiny_model.eval()
        with torch.no_grad():
            # Test with automatic position_ids
            outputs_auto = tiny_model(**batch)

            # Test with explicit position_ids
            seq_len = batch['input_ids'].shape[1]
            position_ids = torch.arange(seq_len, device=batch['input_ids'].device).unsqueeze(0).repeat(len(samples), 1)
            batch_with_pos = {**batch, 'position_ids': position_ids}
            outputs_explicit = tiny_model(**batch_with_pos)

        # Both should work
        assert outputs_auto['logits'].shape == outputs_explicit['logits'].shape, "Position IDs should not affect output shape"
