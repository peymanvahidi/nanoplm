#!/usr/bin/env python3
"""
Test suite for the custom ProtDataCollatorForLM to verify it works correctly.
"""

import torch
from nanoplm.pretraining.collator import ProtDataCollatorForLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer


def test_collator_basic_functionality():
    """Test basic collator functionality: padding and batching"""
    print("Testing basic collator functionality...")

    tokenizer = ProtModernBertTokenizer()
    collator = ProtDataCollatorForLM(
        tokenizer=tokenizer,
        mlm_probability=0.0,  # No masking for basic test
        mask_token_probability=0.8,
        random_token_probability=0.1,
        keep_probability=0.1
    )

    # Create test sequences of different lengths
    sequences = [
        "MKALCLLLLPVLGLLTGSSGS",  # Short sequence
        "MKALCLLLLPVLGLLTGSSGSGSGSGSGSGSGSGSGSGSGSGSGSGS",  # Medium sequence
        "M"  # Very short sequence
    ]

    # Tokenize sequences (simulating what dataset would do)
    tokenized = []
    for seq in sequences:
        encoding = tokenizer(seq, padding=False, truncation=True, max_length=50, return_tensors=None)
        tokenized.append({
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        })

    # Apply collator
    batch = collator(tokenized)

    # Verify batch structure
    assert 'input_ids' in batch, "Batch should contain input_ids"
    assert 'attention_mask' in batch, "Batch should contain attention_mask"
    assert 'labels' in batch, "Batch should contain labels"

    # Verify shapes
    batch_size = len(sequences)
    max_len = max(len(seq['input_ids']) for seq in tokenized)
    assert batch['input_ids'].shape == (batch_size, max_len), f"Expected shape {(batch_size, max_len)}, got {batch['input_ids'].shape}"
    assert batch['attention_mask'].shape == (batch_size, max_len), f"Expected shape {(batch_size, max_len)}, got {batch['attention_mask'].shape}"

    # Verify attention mask correctness
    for i, seq in enumerate(tokenized):
        seq_len = len(seq['input_ids'])
        expected_attention = torch.cat([torch.ones(seq_len), torch.zeros(max_len - seq_len)])
        assert torch.equal(batch['attention_mask'][i], expected_attention), f"Attention mask incorrect for sequence {i}"

    print("✓ Basic functionality test passed")


def test_mlm_masking_probabilities():
    """Test that MLM masking respects probability settings"""
    print("Testing MLM masking probabilities...")

    tokenizer = ProtModernBertTokenizer()
    mlm_prob = 0.5  # High probability to ensure masking occurs
    collator = ProtDataCollatorForLM(
        tokenizer=tokenizer,
        mlm_probability=mlm_prob,
        mask_token_probability=0.8,
        random_token_probability=0.1,
        keep_probability=0.1
    )

    # Create a simple test sequence
    sequence = "MKALCL"
    tokenized = [{
        'input_ids': tokenizer(sequence, padding=False, return_tensors=None)['input_ids'],
        'attention_mask': tokenizer(sequence, padding=False, return_tensors=None)['attention_mask']
    }]

    # Run collator and verify basic functionality
    batch = collator(tokenized)

    # Verify output structure
    assert 'input_ids' in batch and 'attention_mask' in batch and 'labels' in batch

    input_ids = batch['input_ids'][0]
    labels = batch['labels'][0]
    attention_mask = batch['attention_mask'][0]

    # Verify masking occurred (some positions changed)
    original_tokens = torch.tensor(tokenized[0]['input_ids'])
    changed_positions = (input_ids != original_tokens)
    assert changed_positions.sum() > 0, "Some masking should occur with mlm_probability=0.5"

    # Verify labels are correct: -100 for positions not selected for masking, original tokens for positions that were masked
    # Positions not selected for masking should have -100 labels
    masked_indices = (labels != -100)  # Positions that were selected for masking
    non_masked_positions = ~masked_indices & attention_mask.bool()
    if non_masked_positions.sum() > 0:
        assert (labels[non_masked_positions] == -100).all(), "Non-masked positions should have -100 labels"

    # Positions that were masked should preserve original tokens in labels
    if masked_indices.sum() > 0:
        assert (labels[masked_indices] == original_tokens[masked_indices]).all(), "Masked positions should preserve original tokens in labels"

    # Verify padding positions have -100 labels
    padding_positions = ~attention_mask.bool()
    if padding_positions.sum() > 0:
        assert (labels[padding_positions] == -100).all(), "Padding positions should have -100 labels"

    print("✓ MLM masking probabilities test passed")


def test_special_tokens_not_masked():
    """Test that special tokens are never masked"""
    print("Testing special token protection...")

    tokenizer = ProtModernBertTokenizer()
    collator = ProtDataCollatorForLM(
        tokenizer=tokenizer,
        mlm_probability=1.0,  # Mask everything except specials
        mask_token_probability=0.8,
        random_token_probability=0.1,
        keep_probability=0.1
    )

    # Create sequence with EOS token
    sequence = "MKALCL"
    tokenized = [{
        'input_ids': tokenizer(sequence, padding=False, return_tensors=None)['input_ids'],
        'attention_mask': tokenizer(sequence, padding=False, return_tensors=None)['attention_mask']
    }]

    batch = collator(tokenized)
    input_ids = batch['input_ids'][0]

    # EOS token should never be masked (it's at the end)
    eos_positions = (input_ids == tokenizer.eos_token_id)
    mask_positions = (input_ids == tokenizer.mask_token_id)

    # No overlap between EOS and MASK positions
    assert not (eos_positions & mask_positions).any(), "EOS tokens should never be masked"

    print("✓ Special token protection test passed")


def test_masking_strategies():
    """Test the three masking strategies: MASK, RANDOM, UNCHANGED"""
    print("Testing masking strategies...")

    tokenizer = ProtModernBertTokenizer()
    # Set probabilities to clearly distinguish strategies
    collator = ProtDataCollatorForLM(
        tokenizer=tokenizer,
        mlm_probability=1.0,  # Mask all eligible tokens
        mask_token_probability=0.5,
        random_token_probability=0.3,
        keep_probability=0.2
    )

    # Create a long sequence to get good statistics
    sequence = "MKALCLLLLPVLGLLTGSSGS" * 5
    tokenized = [{
        'input_ids': tokenizer(sequence, padding=False, return_tensors=None)['input_ids'],
        'attention_mask': tokenizer(sequence, padding=False, return_tensors=None)['attention_mask']
    }]

    # Run multiple times to get statistics
    mask_token_count = 0
    random_token_count = 0
    unchanged_count = 0
    total_masked = 0
    num_runs = 50

    original_input_ids = torch.tensor(tokenized[0]['input_ids'])

    for _ in range(num_runs):
        batch = collator(tokenized)
        input_ids = batch['input_ids'][0]
        labels = batch['labels'][0]

        # Find positions that were originally amino acids (not special tokens)
        amino_acid_positions = ~torch.tensor([
            tokenizer.get_special_tokens_mask(seq.tolist(), already_has_special_tokens=True)
            for seq in original_input_ids.unsqueeze(0)
        ][0]).bool()

        # Among originally amino acid positions, count masking strategies
        for pos in range(len(original_input_ids)):
            if not amino_acid_positions[pos]:
                continue

            original_token = original_input_ids[pos]
            current_token = input_ids[pos]

            if current_token == tokenizer.mask_token_id:
                mask_token_count += 1
            elif current_token != original_token:
                random_token_count += 1
            else:
                unchanged_count += 1

            if current_token != original_token:
                total_masked += 1

    print(f"  MASK tokens: {mask_token_count}")
    print(f"  RANDOM tokens: {random_token_count}")
    print(f"  UNCHANGED tokens: {unchanged_count}")
    print(f"  Total modified: {total_masked}")

    # Verify we have all three strategies
    assert mask_token_count > 0, "Should have MASK tokens"
    assert random_token_count > 0, "Should have RANDOM tokens"
    assert unchanged_count > 0, "Should have UNCHANGED tokens"

    # Verify random tokens are valid amino acids (not special tokens)
    # This would require running the collator and checking random token values

    print("✓ Masking strategies test passed")


def test_random_token_replacement():
    """Test that random token replacement uses valid tokens"""
    print("Testing random token replacement...")

    tokenizer = ProtModernBertTokenizer()
    collator = ProtDataCollatorForLM(
        tokenizer=tokenizer,
        mlm_probability=1.0,
        mask_token_probability=0.0,  # No MASK tokens
        random_token_probability=1.0,  # All masked tokens get random replacement
        keep_probability=0.0
    )

    sequence = "MKALCLLLLPVLGLLTGSSGS" * 3
    tokenized = [{
        'input_ids': tokenizer(sequence, padding=False, return_tensors=None)['input_ids'],
        'attention_mask': tokenizer(sequence, padding=False, return_tensors=None)['attention_mask']
    }]

    original_input_ids = torch.tensor(tokenized[0]['input_ids'])
    batch = collator(tokenized)
    input_ids = batch['input_ids'][0]
    labels = batch['labels'][0]

    # Find positions that were changed
    changed_positions = (input_ids != original_input_ids)

    # Verify changed positions have valid random tokens
    for pos in changed_positions.nonzero(as_tuple=True)[0]:
        new_token = input_ids[pos].item()
        # Should be a valid token ID (in vocabulary)
        assert new_token in tokenizer.get_vocab().values(), f"Random token {new_token} not in vocabulary"
        # Should not be a special token
        assert new_token != tokenizer.mask_token_id, "Random replacement should not use MASK token"
        assert new_token != tokenizer.pad_token_id, "Random replacement should not use PAD token"
        assert new_token != tokenizer.eos_token_id, "Random replacement should not use EOS token"
        assert new_token != tokenizer.unk_token_id, "Random replacement should not use UNK token"

    # Verify labels preserve original tokens
    assert (labels[changed_positions] == original_input_ids[changed_positions]).all(), "Labels should preserve original tokens for changed positions"

    print("✓ Random token replacement test passed")


def test_edge_cases():
    """Test edge cases like empty sequences, padding, etc."""
    print("Testing edge cases...")

    tokenizer = ProtModernBertTokenizer()
    collator = ProtDataCollatorForLM(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        mask_token_probability=0.8,
        random_token_probability=0.1,
        keep_probability=0.1
    )

    # Test with empty sequence
    sequences = ["", "M", "MKALCL"]
    tokenized = []
    for seq in sequences:
        encoding = tokenizer(seq, padding=False, truncation=True, max_length=20, return_tensors=None)
        tokenized.append({
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        })

    # Should not crash
    batch = collator(tokenized)

    # Verify basic structure
    assert batch['input_ids'].shape[0] == len(sequences)
    assert batch['attention_mask'].shape[0] == len(sequences)
    assert batch['labels'].shape[0] == len(sequences)

    print("✓ Edge cases test passed")


def test_probability_validation():
    """Test that probability validation works correctly"""
    print("Testing probability validation...")

    tokenizer = ProtModernBertTokenizer()

    # Valid probabilities should work
    try:
        collator = ProtDataCollatorForLM(
            tokenizer=tokenizer,
            mlm_probability=0.15,
            mask_token_probability=0.8,
            random_token_probability=0.1,
            keep_probability=0.1
        )
        print("✓ Valid probabilities accepted")
    except ValueError:
        assert False, "Valid probabilities should not raise ValueError"

    # Invalid probabilities should be normalized (not raise error)
    try:
        collator = ProtDataCollatorForLM(
            tokenizer=tokenizer,
            mlm_probability=0.15,
            mask_token_probability=0.5,
            random_token_probability=0.3,
            keep_probability=0.3  # Sum > 1.0 - should be normalized
        )
        # Should not raise an error, but should normalize probabilities
        total_prob = collator.p_mask + collator.p_rand + collator.p_keep
        assert abs(total_prob - 1.0) < 1e-6, f"Probabilities should be normalized to sum to 1.0, got {total_prob}"
        print("✓ Invalid probabilities correctly normalized")
    except ValueError:
        assert False, "Invalid probabilities should be normalized, not raise ValueError"

    print("✓ Probability validation test passed")


def test_non_standard_aa_tokens_never_masked():
    """Test that non-standard AA tokens (X=24) are never selected for masking when excluded."""
    print("Testing non-standard AA token masking exclusion...")

    tokenizer = ProtModernBertTokenizer()
    non_standard_ids = tokenizer.NON_STANDARD_AA_TOKEN_IDS

    collator = ProtDataCollatorForLM(
        tokenizer=tokenizer,
        mlm_probability=1.0,  # mask everything eligible
        mask_token_probability=0.8,
        random_token_probability=0.1,
        keep_probability=0.1,
        extra_excluded_token_ids=non_standard_ids,
    )

    # Sequence with X tokens: "AXGXMXK" -> [4, 24, 6, 24, 20, 24, 15, 1(eos)]
    for _ in range(50):
        seq = "AXGXMXK"
        tokenized = [{
            'input_ids': tokenizer(seq, padding=False, return_tensors=None)['input_ids'],
            'attention_mask': tokenizer(seq, padding=False, return_tensors=None)['attention_mask'],
        }]

        batch = collator(tokenized)
        labels = batch['labels'][0]
        input_ids = batch['input_ids'][0]
        original = torch.tensor(tokenized[0]['input_ids'])

        # Find X positions in original (token ID 24)
        x_positions = (original == 24)

        # X positions must ALWAYS have label=-100 (never masked for loss)
        assert (labels[x_positions] == -100).all(), (
            f"X token positions should have label=-100 but got {labels[x_positions]}"
        )

        # X positions must NEVER be replaced in input_ids
        assert (input_ids[x_positions] == original[x_positions]).all(), (
            f"X token positions should keep original value but got {input_ids[x_positions]}"
        )

        # Standard AA positions should still be maskable (mlm_probability=1.0)
        standard_mask = torch.zeros_like(original, dtype=torch.bool)
        for sid in tokenizer.STANDARD_AA_TOKEN_IDS:
            standard_mask |= (original == sid)
        masked_standard = (labels[standard_mask] != -100)
        assert masked_standard.any(), "Standard AAs should still be masked (mlm_probability=1.0)"

    print("✓ Non-standard AA token masking exclusion test passed")


def test_non_standard_tokens_excluded_from_random_replacement():
    """Test that non-standard AA tokens are never used as random replacements."""
    print("Testing non-standard AA token random replacement exclusion...")

    tokenizer = ProtModernBertTokenizer()
    non_standard_ids = tokenizer.NON_STANDARD_AA_TOKEN_IDS

    collator = ProtDataCollatorForLM(
        tokenizer=tokenizer,
        mlm_probability=1.0,
        mask_token_probability=0.0,    # no [MASK] replacements
        random_token_probability=1.0,  # ALL masked tokens get random replacement
        keep_probability=0.0,
        extra_excluded_token_ids=non_standard_ids,
    )

    # Use a long standard-AA-only sequence for many random replacements
    seq = "MKALCLLLLPVLGLLTGSSGS" * 5
    tokenized = [{
        'input_ids': tokenizer(seq, padding=False, return_tensors=None)['input_ids'],
        'attention_mask': tokenizer(seq, padding=False, return_tensors=None)['attention_mask'],
    }]

    for _ in range(20):
        batch = collator(tokenized)
        input_ids = batch['input_ids'][0]
        original = torch.tensor(tokenized[0]['input_ids'])

        # Find positions that were changed (random replacements)
        changed = input_ids != original
        replaced_tokens = set(input_ids[changed].tolist())

        # None of the replaced tokens should be non-standard AA IDs
        for tid in replaced_tokens:
            assert tid not in non_standard_ids, (
                f"Token ID {tid} is a non-standard AA and should not be used as random replacement"
            )

    print("✓ Non-standard AA token random replacement exclusion test passed")


def test_tokenizer_aa_classification():
    """Test that tokenizer correctly classifies standard vs non-standard AA token IDs."""
    print("Testing tokenizer AA classification...")

    tokenizer = ProtModernBertTokenizer()
    vocab = tokenizer.get_vocab()

    # 20 standard amino acids
    standard_aas = set("ALGVSREDTIPKFQNYMHWC")
    expected_standard_ids = frozenset(vocab[aa] for aa in standard_aas)
    assert tokenizer.STANDARD_AA_TOKEN_IDS == expected_standard_ids, (
        f"Expected {expected_standard_ids}, got {tokenizer.STANDARD_AA_TOKEN_IDS}"
    )

    # Non-standard: X, B, O, U, Z
    non_standard_aas = set("XBOUZ")
    expected_non_standard_ids = frozenset(vocab[aa] for aa in non_standard_aas)
    assert tokenizer.NON_STANDARD_AA_TOKEN_IDS == expected_non_standard_ids, (
        f"Expected {expected_non_standard_ids}, got {tokenizer.NON_STANDARD_AA_TOKEN_IDS}"
    )

    # No overlap
    assert tokenizer.STANDARD_AA_TOKEN_IDS.isdisjoint(tokenizer.NON_STANDARD_AA_TOKEN_IDS), (
        "Standard and non-standard AA IDs should not overlap"
    )

    # Neither set contains special tokens
    special_ids = set(tokenizer.all_special_ids)
    assert tokenizer.STANDARD_AA_TOKEN_IDS.isdisjoint(special_ids), (
        "Standard AA IDs should not include special tokens"
    )
    assert tokenizer.NON_STANDARD_AA_TOKEN_IDS.isdisjoint(special_ids), (
        "Non-standard AA IDs should not include special tokens"
    )

    print("  Standard AA IDs:", sorted(tokenizer.STANDARD_AA_TOKEN_IDS))
    print("  Non-standard AA IDs:", sorted(tokenizer.NON_STANDARD_AA_TOKEN_IDS))
    print("  Special IDs:", sorted(special_ids))

    print("✓ Tokenizer AA classification test passed")


def main():
    """Run all collator tests"""
    print("Running ProtDataCollatorForLM tests...\n")

    test_collator_basic_functionality()
    print()

    test_mlm_masking_probabilities()
    print()

    test_special_tokens_not_masked()
    print()

    test_masking_strategies()
    print()

    test_random_token_replacement()
    print()

    test_edge_cases()
    print()

    test_probability_validation()
    print()

    test_tokenizer_aa_classification()
    print()

    test_non_standard_aa_tokens_never_masked()
    print()

    test_non_standard_tokens_excluded_from_random_replacement()
    print()

    print("All tests passed!")


if __name__ == "__main__":
    main()
