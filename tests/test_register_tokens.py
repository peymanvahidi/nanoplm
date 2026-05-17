"""Tests for register token support in tokenizer, shard writing, and collator."""

import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# The nanoplm.pretraining package __init__ transitively imports triton
# (via canon_ops), which is Linux-only.  Guard against that so the test
# module can be collected on macOS / environments without triton.
_PRETRAINING_PKG = "nanoplm.pretraining"
if _PRETRAINING_PKG not in sys.modules:
    # Temporarily make the package importable without its __init__ side-effects.
    _pkg_spec = importlib.util.find_spec(_PRETRAINING_PKG)
    if _pkg_spec is not None:
        try:
            importlib.import_module(_PRETRAINING_PKG)
        except (ImportError, ModuleNotFoundError):
            # Create a bare package so submodule imports resolve.
            import types
            pkg = types.ModuleType(_PRETRAINING_PKG)
            pkg.__path__ = [str(Path(_pkg_spec.origin).parent)]
            pkg.__package__ = _PRETRAINING_PKG
            sys.modules[_PRETRAINING_PKG] = pkg

from nanoplm.pretraining.collator import (
    DataCollatorWithFlattening,
    ProtDataCollatorForLM,
)
from nanoplm.pretraining.dataset import ShardedDataset
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
from nanoplm.data.sharding.pretrain_shard_writer import (
    create_shards_native,
    is_native_shard_writer_available,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FASTA_SEQUENCES = ["ALGVSRED", "MKALCLLLL", "WWCC"]

FASTA_TEXT = ">s0\nALGVSRED\n>s1\nMKALCLLLL\n>s2\nWWCC\n"


def _expected_tokens(tokenizer, sequences, max_length):
    """Tokenize sequences using the Python tokenizer as ground truth."""
    out = []
    for seq in sequences:
        enc = tokenizer(
            seq,
            add_special_tokens=True,
            padding=False,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        out.append(enc["input_ids"].squeeze(0).numpy().astype(np.uint8))
    return out


# ---------------------------------------------------------------------------
# Tokenizer unit tests
# ---------------------------------------------------------------------------


class TestTokenizerRegisterTokens:
    """Verify ProtModernBertTokenizer produces correct output with register tokens."""

    def test_zero_registers_unchanged(self):
        """Default (0 registers) should match original behavior: seq + EOS."""
        tok = ProtModernBertTokenizer(num_register_tokens=0)
        ids = tok("ALGV", add_special_tokens=True)["input_ids"]
        assert ids == [4, 5, 6, 7, 1], f"Expected [A L G V EOS], got {ids}"

    def test_one_register_appended_after_eos(self):
        """1 register: seq + EOS + <unused0>."""
        tok = ProtModernBertTokenizer(num_register_tokens=1)
        ids = tok("ALGV", add_special_tokens=True)["input_ids"]
        assert ids == [4, 5, 6, 7, 1, 30], f"Expected [A L G V EOS unused0], got {ids}"

    def test_two_registers_appended_after_eos(self):
        """2 registers: seq + EOS + <unused0> + <unused1>."""
        tok = ProtModernBertTokenizer(num_register_tokens=2)
        ids = tok("ALGV", add_special_tokens=True)["input_ids"]
        assert ids == [4, 5, 6, 7, 1, 30, 31], (
            f"Expected [A L G V EOS unused0 unused1], got {ids}"
        )

    def test_invalid_num_register_tokens_raises(self):
        """Values outside 0-2 should raise ValueError."""
        with pytest.raises(ValueError):
            ProtModernBertTokenizer(num_register_tokens=3)
        with pytest.raises(ValueError):
            ProtModernBertTokenizer(num_register_tokens=-1)

    def test_register_tokens_with_bos(self):
        """Registers should come after EOS even when BOS is enabled."""
        tok = ProtModernBertTokenizer(use_bos_token=True, num_register_tokens=2)
        ids = tok("ALG", add_special_tokens=True)["input_ids"]
        # Expected: BOS A L G EOS unused0 unused1
        assert ids == [29, 4, 5, 6, 1, 30, 31], (
            f"Expected [BOS A L G EOS unused0 unused1], got {ids}"
        )

    def test_register_tokens_are_special(self):
        """Register tokens should be in all_special_ids so MLM skips them."""
        tok0 = ProtModernBertTokenizer(num_register_tokens=0)
        assert 30 not in tok0.all_special_ids
        assert 31 not in tok0.all_special_ids

        tok1 = ProtModernBertTokenizer(num_register_tokens=1)
        assert 30 in tok1.all_special_ids
        assert 31 not in tok1.all_special_ids

        tok2 = ProtModernBertTokenizer(num_register_tokens=2)
        assert 30 in tok2.all_special_ids
        assert 31 in tok2.all_special_ids

    def test_special_tokens_mask_protects_registers(self):
        """get_special_tokens_mask should mark register tokens as 1 (protected)."""
        tok = ProtModernBertTokenizer(num_register_tokens=2)
        ids = tok("ALGV", add_special_tokens=True)["input_ids"]
        # ids = [A=4, L=5, G=6, V=7, EOS=1, unused0=30, unused1=31]
        mask = tok.get_special_tokens_mask(ids, already_has_special_tokens=True)
        #        A  L  G  V  EOS unused0 unused1
        assert mask == [0, 0, 0, 0, 1, 1, 1]

    def test_truncation_preserves_register_tokens(self):
        """When sequence is truncated, register tokens should still be appended."""
        tok = ProtModernBertTokenizer(num_register_tokens=2)
        long_seq = "A" * 100
        ids = tok(
            long_seq,
            add_special_tokens=True,
            max_length=10,
            truncation=True,
        )["input_ids"]
        # 10 total: 7 amino acids + EOS + unused0 + unused1
        assert len(ids) == 10
        assert ids[-3:] == [1, 30, 31], f"Tail should be [EOS, unused0, unused1], got {ids[-3:]}"

    def test_num_register_tokens_attribute(self):
        """Tokenizer should expose num_register_tokens as an attribute."""
        tok = ProtModernBertTokenizer(num_register_tokens=2)
        assert tok.num_register_tokens == 2


# ---------------------------------------------------------------------------
# Collator tests
# ---------------------------------------------------------------------------


class TestCollatorWithRegisterTokens:
    """Verify the MLM collator correctly handles register tokens."""

    @pytest.fixture
    def tokenizer_with_registers(self):
        return ProtModernBertTokenizer(num_register_tokens=2)

    @pytest.fixture
    def collator(self, tokenizer_with_registers):
        return ProtDataCollatorForLM(
            tokenizer=tokenizer_with_registers,
            mlm_probability=1.0,  # mask everything maskable
            mask_token_probability=1.0,
            random_token_probability=0.0,
            keep_probability=0.0,
        )

    def _tokenize_batch(self, tokenizer, sequences):
        batch = []
        for seq in sequences:
            enc = tokenizer(seq, add_special_tokens=True, padding=False, return_tensors=None)
            batch.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
        return batch

    def test_register_tokens_never_masked(self, tokenizer_with_registers, collator):
        """With mlm_probability=1.0, register tokens and EOS must remain unmasked."""
        batch = self._tokenize_batch(tokenizer_with_registers, ["ALGVSREDTIPK"] * 4)
        result = collator(batch)

        input_ids = result["input_ids"]
        # Check every row: EOS (1) and register tokens (30, 31) should never become mask (3)
        for row in input_ids:
            tokens = row.tolist()
            # Find positions of EOS and register tokens in original
            # With mlm=1.0, all amino acids should be masked but specials should not
            # EOS and register tokens should remain as-is
            non_pad = [t for t in tokens if t != 0]  # exclude padding
            assert 1 in non_pad, "EOS token should be preserved"
            assert 30 in non_pad, "<unused0> should be preserved"
            assert 31 in non_pad, "<unused1> should be preserved"

    def test_register_tokens_excluded_from_random_pool(self, tokenizer_with_registers):
        """Register tokens should not appear in random replacements."""
        collator = ProtDataCollatorForLM(
            tokenizer=tokenizer_with_registers,
            mlm_probability=1.0,
            mask_token_probability=0.0,
            random_token_probability=1.0,  # all masked positions get random replacement
            keep_probability=0.0,
        )
        pool = collator.allowed_random_token_ids.tolist()
        assert 30 not in pool, "<unused0> should not be in random replacement pool"
        assert 31 not in pool, "<unused1> should not be in random replacement pool"
        # Regular amino acids should still be in the pool
        assert 4 in pool, "Amino acid A (4) should be in pool"

    def test_register_tokens_labels_ignored(self, tokenizer_with_registers, collator):
        """Labels for register token positions should be -100 (not part of loss)."""
        batch = self._tokenize_batch(tokenizer_with_registers, ["ALGVSRED"] * 4)
        result = collator(batch)

        labels = result["labels"]
        input_ids = result["input_ids"]
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                tok = input_ids[i, j].item()
                if tok in (0, 1, 30, 31):  # PAD, EOS, unused0, unused1
                    assert labels[i, j].item() == -100, (
                        f"Label at [{i},{j}] should be -100 for special token {tok}"
                    )


class TestFlatteningCollatorWithRegisterTokens:
    """Verify DataCollatorWithFlattening handles register tokens in cu_seqlens."""

    def test_register_tokens_in_sequence_boundary(self):
        """Register tokens must be included within cu_seqlens boundaries."""
        tok = ProtModernBertTokenizer(num_register_tokens=2)
        inner = ProtDataCollatorForLM(
            tokenizer=tok,
            mlm_probability=0.0,
        )
        collator = DataCollatorWithFlattening(
            collator=inner,
            return_position_ids=True,
        )

        sequences = ["ALGV", "MKAL"]
        batch = []
        for seq in sequences:
            enc = tok(seq, add_special_tokens=True, padding=False, return_tensors=None)
            batch.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})

        result = collator(batch)

        # Each sequence: 4 AA + EOS + unused0 + unused1 = 7 tokens
        cu = result["cu_seqlens"].tolist()
        assert cu == [0, 7, 14], f"Expected cu_seqlens [0, 7, 14], got {cu}"
        assert result["max_seqlen"] == 7

        # Position IDs should reset per sequence
        pos = result["position_ids"].tolist()
        assert pos == [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6], (
            f"Position IDs should reset per sequence, got {pos}"
        )

        # Register tokens should be present in flattened input_ids
        flat_ids = result["input_ids"].tolist()
        # Each seq: [A L G V EOS unused0 unused1]
        assert flat_ids[4:7] == [1, 30, 31], "First sequence should end with EOS + registers"
        assert flat_ids[11:14] == [1, 30, 31], "Second sequence should end with EOS + registers"


# ---------------------------------------------------------------------------
# Native shard writer tests
# ---------------------------------------------------------------------------


class TestNativeShardWriterRegisterTokens:
    """Verify native C shard writer produces correct output with register tokens."""

    @pytest.fixture(autouse=True)
    def _check_native(self):
        available, error = is_native_shard_writer_available()
        if not available:
            pytest.skip(f"Native shard writer unavailable: {error}")

    def test_native_matches_tokenizer_with_two_registers(self):
        """Native path output should exactly match Python tokenizer output."""
        tok = ProtModernBertTokenizer(num_register_tokens=2)
        max_length = 32
        expected = _expected_tokens(tok, FASTA_SEQUENCES, max_length)

        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = Path(tmpdir) / "test.fasta"
            output_dir = Path(tmpdir) / "shards"
            output_dir.mkdir()
            fasta_path.write_text(FASTA_TEXT, encoding="utf-8")

            result = create_shards_native(
                fasta_path=fasta_path,
                output_dir=output_dir,
                max_length=max_length,
                samples_per_shard=100,
                num_threads=1,
                use_bos_token=False,
                num_register_tokens=2,
            )
            assert result.sequence_count == len(FASTA_SEQUENCES)

            ds = ShardedDataset(str(output_dir))
            for idx in range(len(ds)):
                got = ds[idx]["input_ids"].numpy().astype(np.uint8)
                np.testing.assert_array_equal(
                    got, expected[idx],
                    err_msg=f"Native shard {idx} mismatch with tokenizer",
                )

    def test_native_matches_tokenizer_with_one_register(self):
        """Single register token via native path."""
        tok = ProtModernBertTokenizer(num_register_tokens=1)
        max_length = 32
        expected = _expected_tokens(tok, FASTA_SEQUENCES, max_length)

        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = Path(tmpdir) / "test.fasta"
            output_dir = Path(tmpdir) / "shards"
            output_dir.mkdir()
            fasta_path.write_text(FASTA_TEXT, encoding="utf-8")

            result = create_shards_native(
                fasta_path=fasta_path,
                output_dir=output_dir,
                max_length=max_length,
                samples_per_shard=100,
                num_threads=1,
                use_bos_token=False,
                num_register_tokens=1,
            )

            ds = ShardedDataset(str(output_dir))
            for idx in range(len(ds)):
                got = ds[idx]["input_ids"].numpy().astype(np.uint8)
                np.testing.assert_array_equal(got, expected[idx])

    def test_native_zero_registers_backward_compatible(self):
        """num_register_tokens=0 should produce identical output to omitting it."""
        tok = ProtModernBertTokenizer(num_register_tokens=0)
        max_length = 32
        expected = _expected_tokens(tok, FASTA_SEQUENCES, max_length)

        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = Path(tmpdir) / "test.fasta"
            output_dir = Path(tmpdir) / "shards"
            output_dir.mkdir()
            fasta_path.write_text(FASTA_TEXT, encoding="utf-8")

            result = create_shards_native(
                fasta_path=fasta_path,
                output_dir=output_dir,
                max_length=max_length,
                samples_per_shard=100,
                num_threads=1,
                use_bos_token=False,
                num_register_tokens=0,
            )

            ds = ShardedDataset(str(output_dir))
            for idx in range(len(ds)):
                got = ds[idx]["input_ids"].numpy().astype(np.uint8)
                np.testing.assert_array_equal(got, expected[idx])
                # Should NOT contain register token IDs
                assert 30 not in got.tolist()
                assert 31 not in got.tolist()

    def test_native_truncation_with_registers(self):
        """Truncation should account for register tokens in max_length budget."""
        tok = ProtModernBertTokenizer(num_register_tokens=2)
        max_length = 8  # tight: only 5 AA slots (8 - EOS - unused0 - unused1)
        long_seq = "A" * 100
        expected = _expected_tokens(tok, [long_seq], max_length)

        fasta_text = ">s0\n" + long_seq + "\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = Path(tmpdir) / "test.fasta"
            output_dir = Path(tmpdir) / "shards"
            output_dir.mkdir()
            fasta_path.write_text(fasta_text, encoding="utf-8")

            create_shards_native(
                fasta_path=fasta_path,
                output_dir=output_dir,
                max_length=max_length,
                samples_per_shard=100,
                num_threads=1,
                use_bos_token=False,
                num_register_tokens=2,
            )

            ds = ShardedDataset(str(output_dir))
            got = ds[0]["input_ids"].numpy().astype(np.uint8)
            np.testing.assert_array_equal(got, expected[0])
            assert len(got) == max_length
            assert got[-3:].tolist() == [1, 30, 31], "Last 3 should be EOS + registers"

    def test_native_multi_shard_with_registers(self):
        """Register tokens should be consistent across multiple shards."""
        tok = ProtModernBertTokenizer(num_register_tokens=2)
        max_length = 32
        expected = _expected_tokens(tok, FASTA_SEQUENCES, max_length)

        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = Path(tmpdir) / "test.fasta"
            output_dir = Path(tmpdir) / "shards"
            output_dir.mkdir()
            fasta_path.write_text(FASTA_TEXT, encoding="utf-8")

            result = create_shards_native(
                fasta_path=fasta_path,
                output_dir=output_dir,
                max_length=max_length,
                samples_per_shard=2,  # Force multiple shards
                num_threads=1,
                use_bos_token=False,
                num_register_tokens=2,
            )
            assert result.sequence_count == 3
            assert len(result.shard_paths) == 2  # 2 + 1

            ds = ShardedDataset(str(output_dir))
            for idx in range(len(ds)):
                got = ds[idx]["input_ids"].numpy().astype(np.uint8)
                np.testing.assert_array_equal(
                    got, expected[idx],
                    err_msg=f"Multi-shard mismatch at index {idx}",
                )

    def test_native_with_bos_and_registers(self):
        """BOS + register tokens should all be present in correct order."""
        tok = ProtModernBertTokenizer(use_bos_token=True, num_register_tokens=2)
        max_length = 32
        expected = _expected_tokens(tok, FASTA_SEQUENCES, max_length)

        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = Path(tmpdir) / "test.fasta"
            output_dir = Path(tmpdir) / "shards"
            output_dir.mkdir()
            fasta_path.write_text(FASTA_TEXT, encoding="utf-8")

            create_shards_native(
                fasta_path=fasta_path,
                output_dir=output_dir,
                max_length=max_length,
                samples_per_shard=100,
                num_threads=1,
                use_bos_token=True,
                num_register_tokens=2,
            )

            ds = ShardedDataset(str(output_dir))
            for idx in range(len(ds)):
                got = ds[idx]["input_ids"].numpy().astype(np.uint8)
                np.testing.assert_array_equal(got, expected[idx])
                # Verify structure: BOS ... EOS unused0 unused1
                assert got[0] == 29, "First token should be BOS"
                assert got[-3:].tolist() == [1, 30, 31]
