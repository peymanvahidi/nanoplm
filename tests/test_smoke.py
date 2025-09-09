"""
Smoke tests for nanoPLM - basic functionality verification.
These tests can run without pytest and verify core components work.
"""

import sys
import os
import tempfile
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLM
from nanoplm.pretraining.dataset import FastaMLMDataset
from transformers import DataCollatorForLanguageModeling


class SmokeTests:
    """Basic smoke tests for nanoPLM components."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def log(self, message, status="INFO"):
        print(f"[{status}] {message}")

    def test_tokenizer_creation(self):
        """Test that tokenizer can be created."""
        try:
            tokenizer = ProtModernBertTokenizer()
            assert tokenizer is not None
            assert hasattr(tokenizer, 'vocab_size')
            self.log("Tokenizer creation: PASSED")
            self.passed += 1
            return True
        except Exception as e:
            self.log(f"Tokenizer creation: FAILED - {e}", "ERROR")
            self.failed += 1
            self.errors.append(str(e))
            return False

    def test_model_creation(self):
        """Test that model can be created."""
        try:
            tokenizer = ProtModernBertTokenizer()
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
            assert model is not None
            self.log("Model creation: PASSED")
            self.passed += 1
            return True
        except Exception as e:
            self.log(f"Model creation: FAILED - {e}", "ERROR")
            self.failed += 1
            self.errors.append(str(e))
            return False

    def test_fasta_dataset_creation(self):
        """Test that FASTA dataset can be created."""
        try:
            # Create temporary FASTA file
            fasta_content = """>seq1
MKALCL
>seq2
MVLSPA
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
                f.write(fasta_content)
                temp_path = f.name

            try:
                tokenizer = ProtModernBertTokenizer()
                dataset = FastaMLMDataset(
                    fasta_path=temp_path,
                    tokenizer=tokenizer,
                    max_length=32
                )

                assert len(dataset) == 2
                assert dataset.max_length == 32

                # Test accessing a sample
                sample = dataset[0]
                assert 'input_ids' in sample
                assert 'attention_mask' in sample
                assert isinstance(sample['input_ids'], list)

                self.log("FASTA dataset creation: PASSED")
                self.passed += 1
                return True

            finally:
                os.unlink(temp_path)

        except Exception as e:
            self.log(f"FASTA dataset creation: FAILED - {e}", "ERROR")
            self.failed += 1
            self.errors.append(str(e))
            return False

    def test_data_collator(self):
        """Test that data collator works."""
        try:
            tokenizer = ProtModernBertTokenizer()

            # Create test sequences
            sequences = [
                {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]},
                {'input_ids': [1, 4, 5, 6], 'attention_mask': [1, 1, 1, 1]},
            ]

            collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=0.1
            )

            batch = collator(sequences)

            assert 'input_ids' in batch
            assert 'attention_mask' in batch
            assert 'labels' in batch
            assert batch['input_ids'].shape[0] == 2  # batch size
            assert batch['input_ids'].shape[1] == 4  # max length

            self.log("Data collator: PASSED")
            self.passed += 1
            return True

        except Exception as e:
            self.log(f"Data collator: FAILED - {e}", "ERROR")
            self.failed += 1
            self.errors.append(str(e))
            return False

    def test_model_forward_pass(self):
        """Test that model can do a forward pass."""
        try:
            tokenizer = ProtModernBertTokenizer()
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

            # Create a simple batch
            batch = {
                'input_ids': torch.tensor([[1, 2, 3, 0]]),
                'attention_mask': torch.tensor([[1, 1, 1, 0]]),
                'labels': torch.tensor([[1, 2, 3, -100]])
            }

            model.eval()
            with torch.no_grad():
                outputs = model(**batch)

            assert 'loss' in outputs
            assert 'logits' in outputs
            assert outputs['logits'].shape[0] == 1  # batch size
            assert outputs['logits'].shape[2] == tokenizer.vocab_size

            self.log("Model forward pass: PASSED")
            self.passed += 1
            return True

        except Exception as e:
            self.log(f"Model forward pass: FAILED - {e}", "ERROR")
            self.failed += 1
            self.errors.append(str(e))
            return False

    def test_full_pipeline(self):
        """Test the complete pipeline."""
        try:
            # Create temporary FASTA file
            fasta_content = """>seq1
MKALCL
>seq2
MVLSPA
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
                f.write(fasta_content)
                temp_path = f.name

            try:
                # Create components
                tokenizer = ProtModernBertTokenizer()
                dataset = FastaMLMDataset(
                    fasta_path=temp_path,
                    tokenizer=tokenizer,
                    max_length=16
                )

                # Get samples and create batch
                samples = [dataset[i] for i in range(len(dataset))]
                collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=True,
                    mlm_probability=0.1
                )
                batch = collator(samples)

                # Create model and run forward pass
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

                model.eval()
                with torch.no_grad():
                    outputs = model(**batch)

                assert 'loss' in outputs
                assert 'logits' in outputs

                self.log("Full pipeline: PASSED")
                self.passed += 1
                return True

            finally:
                os.unlink(temp_path)

        except Exception as e:
            self.log(f"Full pipeline: FAILED - {e}", "ERROR")
            self.failed += 1
            self.errors.append(str(e))
            return False

    def run_all_tests(self):
        """Run all smoke tests."""
        print("="*50)
        print("RUNNING NANO PLM SMOKE TESTS")
        print("="*50)

        # Run all tests
        test_methods = [
            self.test_tokenizer_creation,
            self.test_model_creation,
            self.test_fasta_dataset_creation,
            self.test_data_collator,
            self.test_model_forward_pass,
            self.test_full_pipeline
        ]

        for test_method in test_methods:
            test_method()

        # Print summary
        print("\n" + "="*50)
        print("SMOKE TEST SUMMARY")
        print("="*50)
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success rate: {(self.passed / (self.passed + self.failed)) * 100:.1f}%")

        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  - {error}")

        return self.failed == 0


def main():
    """Main entry point for smoke tests."""
    tester = SmokeTests()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
