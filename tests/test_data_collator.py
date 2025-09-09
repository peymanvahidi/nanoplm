import pytest
import torch
from transformers import DataCollatorForLanguageModeling, AutoTokenizer

from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer


class TestDataCollator:
    """Test suite for data collator functionality with ModernBERT."""

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer for testing."""
        return ProtModernBertTokenizer()

    @pytest.fixture
    def standard_tokenizer(self):
        """Create a standard tokenizer for comparison tests."""
        return AutoTokenizer.from_pretrained('bert-base-uncased')

    def test_data_collator_basic_padding(self, standard_tokenizer):
        """Test that DataCollatorForLanguageModeling properly pads sequences."""
        # Create test sequences of different lengths
        sequences = [
            'Hello world',
            'This is a longer sequence with more tokens',
            'Short'
        ]

        # Tokenize without padding (like dataset does)
        tokenized = []
        for seq in sequences:
            encoding = standard_tokenizer(
                seq,
                padding=False,
                truncation=True,
                max_length=20,
                return_tensors=None
            )
            tokenized.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask']
            })

        # Verify different lengths before padding
        lengths = [len(seq['input_ids']) for seq in tokenized]
        assert lengths == [4, 11, 3], f"Expected lengths [4, 11, 3], got {lengths}"

        # Test collator
        collator = DataCollatorForLanguageModeling(
            tokenizer=standard_tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
        batch = collator(tokenized)

        # Verify padding worked correctly
        assert batch['input_ids'].shape == torch.Size([3, 11]), "Should pad to max length in batch"
        assert batch['attention_mask'].shape == torch.Size([3, 11]), "Attention mask should match input_ids shape"

        # Check attention mask correctness
        expected_attention = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # First sequence (4 tokens + 7 padding)
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Second sequence (11 tokens, no padding)
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]   # Third sequence (3 tokens + 8 padding)
        ])
        assert torch.equal(batch['attention_mask'], expected_attention), "Attention mask should be correct"

    def test_data_collator_mlm_masking(self, standard_tokenizer):
        """Test that MLM masking is applied correctly."""
        sequences = ['Hello world this is a test']
        tokenized = [standard_tokenizer(seq, padding=False, return_tensors=None) for seq in sequences]

        # Test with masking
        collator_masked = DataCollatorForLanguageModeling(
            tokenizer=standard_tokenizer,
            mlm=True,
            mlm_probability=1.0  # Mask everything for testing
        )
        batch_masked = collator_masked(tokenized)

        # Test without masking
        collator_no_mask = DataCollatorForLanguageModeling(
            tokenizer=standard_tokenizer,
            mlm=False
        )
        batch_no_mask = collator_no_mask(tokenized)

        # With masking, should have some [MASK] tokens (103 for BERT)
        mask_token_id = standard_tokenizer.mask_token_id
        has_mask_tokens = (batch_masked['input_ids'] == mask_token_id).any()
        assert has_mask_tokens, "Should have mask tokens when mlm=True and mlm_probability=1.0"

        # Without masking, should not have mask tokens in original positions
        original_tokens = batch_no_mask['input_ids'][0]
        masked_tokens = batch_masked['input_ids'][0]
        # Find positions where masking occurred (where tokens changed)
        changed_positions = (original_tokens != masked_tokens) & (original_tokens != 0)  # Exclude padding
        assert changed_positions.any(), "Some tokens should be masked"

    def test_data_collator_labels_shape(self, standard_tokenizer):
        """Test that labels have correct shape and content."""
        sequences = ['Hello world', 'This is a test']
        tokenized = [standard_tokenizer(seq, padding=False, return_tensors=None) for seq in sequences]

        collator = DataCollatorForLanguageModeling(
            tokenizer=standard_tokenizer,
            mlm=True,
            mlm_probability=0.5
        )
        batch = collator(tokenized)

        # Should have labels when mlm=True
        assert 'labels' in batch, "Should have labels when mlm=True"
        assert batch['labels'].shape == batch['input_ids'].shape, "Labels should match input_ids shape"

        # Labels should be -100 for non-masked tokens and original token ids for masked tokens
        labels = batch['labels']
        input_ids = batch['input_ids']

        # Find masked positions (where input_ids has mask token)
        mask_positions = (input_ids == standard_tokenizer.mask_token_id)
        non_mask_positions = ~mask_positions

        # Non-masked positions should have -100 in labels
        assert (labels[non_mask_positions] == -100).all(), "Non-masked positions should have -100 labels"

        # Masked positions should have original token ids in labels
        assert mask_positions.any(), "Should have some masked positions"
        # Note: This is a basic check - in practice, the labels would contain the original tokens


class TestProtModernBertCollator:
    """Test suite specifically for ProtModernBert tokenizer and collator."""

    @pytest.fixture
    def tokenizer(self):
        """Create ProtModernBert tokenizer."""
        return ProtModernBertTokenizer()

    def test_protein_sequence_padding(self, tokenizer):
        """Test padding with actual protein sequences."""
        # Simulate protein sequences of different lengths
        sequences = [
            "MKALCLLLLPVLGLLTGSSGS",  # Short sequence
            "MKALCLLLLPVLGLLTGSSGSGSGSGSGSGSGSGSGSGSGSGSGSGS",  # Medium sequence
            "M"  # Very short sequence
        ]

        tokenized = []
        for seq in sequences:
            encoding = tokenizer(
                seq,
                padding=False,
                truncation=True,
                max_length=50,
                return_tensors=None
            )
            tokenized.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask']
            })

        # Verify different lengths
        lengths = [len(seq['input_ids']) for seq in tokenized]
        assert len(set(lengths)) > 1, "Should have different sequence lengths"

        # Test collator
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
        batch = collator(tokenized)

        # Verify batch shape
        max_len = max(lengths)
        assert batch['input_ids'].shape == torch.Size([len(sequences), max_len])
        assert batch['attention_mask'].shape == torch.Size([len(sequences), max_len])

        # Verify attention mask is correct
        for i, seq_len in enumerate(lengths):
            # First seq_len positions should be 1, rest should be 0
            expected_attention = torch.cat([
                torch.ones(seq_len),
                torch.zeros(max_len - seq_len)
            ])
            assert torch.equal(batch['attention_mask'][i], expected_attention)

    def test_empty_and_edge_cases(self, tokenizer):
        """Test edge cases like empty sequences or very long sequences."""
        # Test with empty sequence (should be handled gracefully)
        sequences = ["", "MKALCL", "M"]

        tokenized = []
        for seq in sequences:
            encoding = tokenizer(
                seq,
                padding=False,
                truncation=True,
                max_length=20,
                return_tensors=None
            )
            tokenized.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask']
            })

        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        # Should not crash even with empty sequence
        batch = collator(tokenized)
        assert batch['input_ids'].shape[0] == len(sequences)
        assert batch['attention_mask'].shape[0] == len(sequences)
