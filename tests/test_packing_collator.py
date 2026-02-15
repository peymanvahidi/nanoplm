#!/usr/bin/env python3
"""
Tests for the PackingCollator (sequence packing for MLM pretraining).
"""

import torch
import pytest
from nanoplm.pretraining.collator import PackingCollator
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
from nanoplm.pretraining.models.modern_bert.modeling import _position_ids_from_cu_seqlens


@pytest.fixture
def tokenizer():
    return ProtModernBertTokenizer()


@pytest.fixture
def collator(tokenizer):
    return PackingCollator(
        tokenizer=tokenizer,
        max_seq_len=64,
        mlm_probability=0.0,  # disable masking for deterministic structure tests
    )


@pytest.fixture
def masking_collator(tokenizer):
    return PackingCollator(
        tokenizer=tokenizer,
        max_seq_len=64,
        mlm_probability=0.3,
        mask_token_probability=0.8,
        random_token_probability=0.1,
        keep_probability=0.1,
    )


def _tokenize(tokenizer, sequence: str) -> dict:
    """Tokenize a sequence and return dict with input_ids and attention_mask."""
    enc = tokenizer(sequence, padding=False, truncation=True, max_length=512, return_tensors=None)
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }


def _tokenize_list(tokenizer, sequences: list[str]) -> list[dict]:
    return [_tokenize(tokenizer, seq) for seq in sequences]


# ──────────────────────────────────────────────────────────────────────
# Structure / packing
# ──────────────────────────────────────────────────────────────────────


class TestPackingStructure:
    """Verify that packing produces correct shapes and cu_seqlens."""

    def test_basic_packing(self, tokenizer, collator):
        """Multiple short sequences should be packed into a flat 1-D tensor."""
        seqs = ["MKAL", "GVSA", "PQNF"]
        examples = _tokenize_list(tokenizer, seqs)
        batch = collator(examples)

        assert "input_ids" in batch
        assert "labels" in batch
        assert "cu_seqlens" in batch
        assert "max_seqlen" in batch
        assert "position_ids" in batch

        # Dynamic mode: flat 1-D output with no padding
        assert batch["input_ids"].dim() == 1
        total_input_tokens = sum(len(ex["input_ids"]) for ex in examples)
        assert batch["input_ids"].shape[0] == total_input_tokens

    def test_cu_seqlens_consistency(self, tokenizer, collator):
        """cu_seqlens should have len = num_sequences + 1, and its last value
        should equal the total number of non-padding tokens."""
        seqs = ["MKAL", "GVSARLPQNFYMHWC", "PQ"]
        examples = _tokenize_list(tokenizer, seqs)
        batch = collator(examples)

        cu = batch["cu_seqlens"]
        assert cu[0] == 0, "cu_seqlens must start with 0"
        assert len(cu) == len(seqs) + 1, (
            f"cu_seqlens should have {len(seqs)+1} entries, got {len(cu)}"
        )

        total_real_tokens = batch["input_ids"].shape[0]
        assert cu[-1].item() == total_real_tokens, (
            f"cu_seqlens[-1]={cu[-1].item()} != total tokens={total_real_tokens}"
        )

    def test_seq_lengths_match_cu_seqlens(self, tokenizer, collator):
        """Per-sequence lengths derived from cu_seqlens should match original tokenized lengths."""
        seqs = ["MKAL", "GVSARLPQNFYMHWC", "PQ"]
        examples = _tokenize_list(tokenizer, seqs)

        # Expected lengths (including special tokens like EOS)
        expected_lengths = sorted(
            [len(ex["input_ids"]) for ex in examples], reverse=True
        )

        batch = collator(examples)
        cu = batch["cu_seqlens"]
        actual_lengths = sorted(
            [(cu[i + 1] - cu[i]).item() for i in range(len(cu) - 1)], reverse=True
        )

        assert actual_lengths == expected_lengths, (
            f"Lengths mismatch: expected {expected_lengths}, got {actual_lengths}"
        )

    def test_max_seqlen(self, tokenizer, collator):
        """max_seqlen should equal the length of the longest individual sequence."""
        seqs = ["M", "MKALCLLLLPVLGLLTGSSGS", "PQ"]
        examples = _tokenize_list(tokenizer, seqs)
        batch = collator(examples)

        longest = max(len(ex["input_ids"]) for ex in examples)
        assert batch["max_seqlen"] == longest

    def test_no_token_loss(self, tokenizer, collator):
        """Total tokens should equal sum of all input sequence lengths (no waste)."""
        seqs = ["MKAL", "GVSARLPQNFYMHWC", "PQ", "ACDEFG", "HIKLMNPQRSTVWY"]
        examples = _tokenize_list(tokenizer, seqs)
        batch = collator(examples)

        total_input_tokens = sum(len(ex["input_ids"]) for ex in examples)
        total_packed_tokens = batch["input_ids"].shape[0]
        assert total_packed_tokens == total_input_tokens, (
            f"Token count mismatch: input={total_input_tokens}, packed={total_packed_tokens}"
        )

    def test_single_sequence(self, tokenizer, collator):
        """A single sequence should produce cu_seqlens length 2."""
        seqs = ["MKALCLLLL"]
        examples = _tokenize_list(tokenizer, seqs)
        batch = collator(examples)

        assert batch["input_ids"].dim() == 1
        assert len(batch["cu_seqlens"]) == 2

    def test_sequence_longer_than_max_seq_len(self, tokenizer):
        """A sequence longer than max_seq_len should be truncated."""
        col = PackingCollator(tokenizer=tokenizer, max_seq_len=10, mlm_probability=0.0)
        # This sequence will have more than 10 tokens
        seqs = ["MKALCLLLLPVLGLLTGSSGSGSGSGSGS"]
        examples = _tokenize_list(tokenizer, seqs)
        batch = col(examples)

        total_tokens = batch["input_ids"].shape[0]
        assert total_tokens <= 10

    def test_packing_reduces_tokens(self, tokenizer):
        """With 10 short sequences, flat output should have no padding waste."""
        col = PackingCollator(tokenizer=tokenizer, max_seq_len=64, mlm_probability=0.0)
        seqs = ["MKAL"] * 10  # each ~5 tokens → ~50 tokens total
        examples = _tokenize_list(tokenizer, seqs)
        batch = col(examples)

        total_input = sum(len(ex["input_ids"]) for ex in examples)
        assert batch["input_ids"].shape[0] == total_input, "Flat output should have zero waste"
        assert len(batch["cu_seqlens"]) == 11  # 10 sequences + 1


# ──────────────────────────────────────────────────────────────────────
# Position IDs
# ──────────────────────────────────────────────────────────────────────


class TestPositionIds:
    """Verify position IDs reset correctly at sequence boundaries."""

    def test_position_ids_reset_at_boundaries(self, tokenizer, collator):
        """position_ids should reset to 0 at each new sequence in a packed row."""
        seqs = ["MKAL", "GV", "PQN"]
        examples = _tokenize_list(tokenizer, seqs)
        batch = collator(examples)

        cu = batch["cu_seqlens"]
        total = int(cu[-1].item())
        pos_ids = _position_ids_from_cu_seqlens(cu, total, cu.device)

        # Check that each sequence starts with position 0
        for i in range(len(cu) - 1):
            start = cu[i].item()
            assert pos_ids[start].item() == 0, (
                f"Sequence {i}: position_id at offset {start} should be 0, got {pos_ids[start].item()}"
            )

        # Check that positions are sequential within each sequence
        for i in range(len(cu) - 1):
            s = cu[i].item()
            e = cu[i + 1].item()
            expected = list(range(e - s))
            actual = pos_ids[s:e].tolist()
            assert actual == expected, (
                f"Sequence {i}: expected positions {expected}, got {actual}"
            )


# ──────────────────────────────────────────────────────────────────────
# MLM masking
# ──────────────────────────────────────────────────────────────────────


class TestPackingMLM:
    """Verify MLM masking respects packing constraints."""

    def test_no_padding_in_flat_output(self, tokenizer, masking_collator):
        """Flat output should contain no padding — only real tokens."""
        seqs = ["MKAL", "GV"]
        examples = _tokenize_list(tokenizer, seqs)
        batch = masking_collator(examples)

        total_input = sum(len(ex["input_ids"]) for ex in examples)
        assert batch["input_ids"].shape[0] == total_input, "Flat output should have no padding"

    def test_special_tokens_not_masked(self, tokenizer, masking_collator):
        """EOS tokens should never be corrupted."""
        seqs = ["MKAL", "GVSARLPQNFYMHWC"] * 5
        examples = _tokenize_list(tokenizer, seqs)
        batch = masking_collator(examples)

        eos_id = tokenizer.eos_token_id
        ids = batch["input_ids"]
        labels = batch["labels"]
        # Flat output: all tokens are real, check EOS positions directly
        eos_positions = ids == eos_id
        if eos_positions.any():
            # labels at EOS should be -100 (excluded from MLM)
            assert (labels[eos_positions] == -100).all(), (
                "EOS positions should have label -100 (not selected for masking)"
            )

    def test_some_tokens_masked(self, tokenizer, masking_collator):
        """With mlm_probability=0.3, some non-special tokens should be masked."""
        seqs = ["MKALCLLLLPVLGLLTGSSGSACDEFGHIKLMNPQRSTVWY"] * 5
        examples = _tokenize_list(tokenizer, seqs)
        batch = masking_collator(examples)

        # At least some labels should not be -100
        active_labels = batch["labels"][batch["labels"] != -100]
        assert len(active_labels) > 0, "Expected some tokens to be masked"


# ──────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────


class TestPackingEdgeCases:
    """Edge cases for the packing collator."""

    def test_all_same_length(self, tokenizer, collator):
        """All sequences of the same length should pack perfectly."""
        seqs = ["MKALCL"] * 6
        examples = _tokenize_list(tokenizer, seqs)
        batch = collator(examples)

        total_input = sum(len(ex["input_ids"]) for ex in examples)
        assert batch["input_ids"].shape[0] == total_input

    def test_flat_output_no_padding(self, tokenizer, collator):
        """Flat 1-D output should have no padding tokens."""
        seqs = ["MK"]  # very short
        examples = _tokenize_list(tokenizer, seqs)
        batch = collator(examples)

        total_input = sum(len(ex["input_ids"]) for ex in examples)
        assert batch["input_ids"].shape[0] == total_input
        assert batch["input_ids"].dim() == 1

    def test_dtype_correctness(self, tokenizer, collator):
        """Check that output tensor dtypes are correct."""
        seqs = ["MKAL", "GV"]
        examples = _tokenize_list(tokenizer, seqs)
        batch = collator(examples)

        assert batch["input_ids"].dtype == torch.long
        assert batch["labels"].dtype == torch.long
        assert batch["cu_seqlens"].dtype == torch.int32
        assert batch["position_ids"].dtype == torch.int32
        assert isinstance(batch["max_seqlen"], int)


# ──────────────────────────────────────────────────────────────────────
# Integration: cu_seqlens → position_ids → model compatibility
# ──────────────────────────────────────────────────────────────────────


class TestPackingIntegration:
    """End-to-end checks for packed batches with the model's varlen path."""

    def test_cu_seqlens_position_ids_roundtrip(self, tokenizer, collator):
        """Packed cu_seqlens → position_ids should produce valid indexing into RoPE tables."""
        seqs = ["MKAL", "GVSARLPQNF", "PQ", "ACDEFGHIKL"]
        examples = _tokenize_list(tokenizer, seqs)
        batch = collator(examples)

        cu = batch["cu_seqlens"]
        total = int(cu[-1].item())
        pos_ids = _position_ids_from_cu_seqlens(cu, total, cu.device)

        # All position IDs must be non-negative
        assert (pos_ids >= 0).all()

        # No position ID should exceed max_seqlen - 1
        assert pos_ids.max().item() < batch["max_seqlen"]

    def test_packing_utilization(self, tokenizer):
        """Flat packing should achieve 100% utilization (zero waste)."""
        max_seq_len = 64
        col = PackingCollator(
            tokenizer=tokenizer, max_seq_len=max_seq_len, mlm_probability=0.0
        )
        # 20 short sequences
        seqs = ["MKAL"] * 20
        examples = _tokenize_list(tokenizer, seqs)
        batch = col(examples)

        total_input = sum(len(ex["input_ids"]) for ex in examples)
        total_output = batch["input_ids"].shape[0]
        assert total_output == total_input, (
            f"Flat output should have zero waste: input={total_input}, output={total_output}"
        )
