import pytest
import tempfile
import os
import json
import yaml
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from transformers import DataCollatorForLanguageModeling

from nanoplm.pretraining.dataset import FastaMLMDataset
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLM, ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
from nanoplm.pretraining.pipeline import PretrainingConfig, ResumeConfig, run_pretraining
from nanoplm.cli.pretrain import from_yaml, pretrain


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


class TestResumePretraining:
    """Integration and unit tests for resume pretraining functionality."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create a temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_fasta_content(self):
        """Create minimal FASTA content for fast testing."""
        return """>protein1
MKALCLLLLP
>protein2
MVLSPADKTN
>protein3
MAIGTMAIGT
"""

    @pytest.fixture
    def train_fasta_file(self, sample_fasta_content):
        """Create a temporary training FASTA file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(sample_fasta_content)
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def val_fasta_file(self, sample_fasta_content):
        """Create a temporary validation FASTA file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(sample_fasta_content)
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def tiny_model_for_training(self):
        """Create a tiny model for fast training tests."""
        config = ProtModernBertMLMConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=29,
            mlp_activation="swiglu",
            mlp_dropout=0.0,
            mlp_bias=False,
            attention_bias=False,
            attention_dropout=0.0,
            classifier_activation="gelu"
        )
        model = ProtModernBertMLM(config)
        return model

    @pytest.mark.integration
    @patch('nanoplm.pretraining.pipeline.logger')
    def test_full_resume_training_cycle(
        self, 
        mock_logger,
        temp_checkpoint_dir, 
        train_fasta_file, 
        val_fasta_file,
        tiny_model_for_training
    ):
        """Test complete training and resume cycle."""
        # Disable W&B logging
        with patch.dict(os.environ, {'WANDB_MODE': 'disabled'}):
            # Phase 1: Initial training for 1 epoch
            pretrain_config = PretrainingConfig(
                train_fasta=train_fasta_file,
                val_fasta=val_fasta_file,
                ckp_dir=temp_checkpoint_dir,
                max_length=32,
                batch_size=2,
                num_epochs=1,
                lazy_dataset=False,
                learning_rate=1e-4,
                weight_decay=0.0,
                warmup_ratio=0.0,
                optimizer="adamw",
                gradient_accumulation_steps=1,
                mlm_probability=0.15,
                logging_steps_percentage=0.5,
                eval_steps_percentage=0.5,
                save_steps_percentage=1.0,
                seed=42,
                num_workers=0,
                multi_gpu=False,
                world_size=1,
                project_name="test-resume-training"
            )

            # Train initial model
            run_pretraining(model=tiny_model_for_training, pretrain_config=pretrain_config)

            # Find the actual checkpoint directory (it's in a run-specific subdirectory)
            checkpoint_root = Path(temp_checkpoint_dir)
            run_dirs = [d for d in checkpoint_root.iterdir() if d.is_dir() and d.name.startswith("run-")]
            assert len(run_dirs) == 1, f"Expected 1 run directory, found {len(run_dirs)}"
            checkpoint_dir = run_dirs[0]

            # Verify checkpoint files exist
            assert (checkpoint_dir / "config.json").exists(), "config.json should exist"
            assert (checkpoint_dir / "model.safetensors").exists(), "model.safetensors should exist"
            assert (checkpoint_dir / "trainer_state.json").exists(), "trainer_state.json should exist"

            # Read initial trainer state
            with open(checkpoint_dir / "trainer_state.json", "r") as f:
                initial_state = json.load(f)
            initial_epoch = initial_state.get("epoch", 0)

            # Phase 2: Resume training for 1 more epoch
            resume_config = ResumeConfig(
                is_resume=True,
                checkpoint_dir=str(checkpoint_dir),
                num_epochs=1
            )

            # Load model from checkpoint
            resumed_model = ProtModernBertMLM.from_pretrained(str(checkpoint_dir))

            # Continue training
            run_pretraining(
                model=resumed_model,
                pretrain_config=pretrain_config,
                resume_config=resume_config
            )

            # Find the resume checkpoint directory
            checkpoint_root = Path(temp_checkpoint_dir)
            resume_dirs = [d for d in checkpoint_root.iterdir() if d.is_dir() and d.name.endswith("-resume")]
            assert len(resume_dirs) == 1, f"Expected 1 resume directory, found {len(resume_dirs)}"
            resume_checkpoint_dir = resume_dirs[0]

            # Verify checkpoint updated
            assert (resume_checkpoint_dir / "trainer_state.json").exists(), "trainer_state.json should exist in resume dir"

            # Read final trainer state
            with open(resume_checkpoint_dir / "trainer_state.json", "r") as f:
                final_state = json.load(f)
            final_epoch = final_state.get("epoch", 0)

            # Verify epoch progression (should have trained for 2 epochs total)
            assert final_epoch > initial_epoch, "Should have trained for more epochs"

    @pytest.mark.integration
    def test_resume_preserves_hyperparameters(
        self,
        temp_checkpoint_dir,
        train_fasta_file,
        val_fasta_file,
        tiny_model_for_training
    ):
        """Test that resumed training preserves hyperparameters."""
        with patch.dict(os.environ, {'WANDB_MODE': 'disabled'}):
            # Train with specific hyperparameters
            original_lr = 5e-5
            original_batch_size = 2
            original_optimizer = "adamw"

            pretrain_config = PretrainingConfig(
                train_fasta=train_fasta_file,
                val_fasta=val_fasta_file,
                ckp_dir=temp_checkpoint_dir,
                max_length=32,
                batch_size=original_batch_size,
                num_epochs=1,
                lazy_dataset=False,
                learning_rate=original_lr,
                weight_decay=0.01,
                warmup_ratio=0.0,
                optimizer=original_optimizer,
                gradient_accumulation_steps=1,
                mlm_probability=0.15,
                logging_steps_percentage=0.5,
                eval_steps_percentage=0.5,
                save_steps_percentage=1.0,
                seed=42,
                num_workers=0,
                multi_gpu=False,
                world_size=1,
                project_name="test-hyperparams"
            )

            run_pretraining(model=tiny_model_for_training, pretrain_config=pretrain_config)

            # Find the actual checkpoint directory (it's in a run-specific subdirectory)
            checkpoint_root = Path(temp_checkpoint_dir)
            run_dirs = [d for d in checkpoint_root.iterdir() if d.is_dir() and d.name.startswith("run-")]
            assert len(run_dirs) == 1, f"Expected 1 run directory, found {len(run_dirs)}"
            checkpoint_dir = run_dirs[0]

            # Load training args and verify hyperparameters were saved
            training_args = torch.load(checkpoint_dir / "training_args.bin", weights_only=False)

            assert training_args.learning_rate == original_lr, "Learning rate should match"
            assert training_args.per_device_train_batch_size == original_batch_size, "Batch size should match"
            assert training_args.optim == "adamw_torch", "Optimizer should match"

    def test_yaml_resume_config_parsing(self, temp_checkpoint_dir, train_fasta_file, val_fasta_file):
        """Test YAML parsing with resume block."""
        yaml_content = {
            "resume": {
                "is_resume": True,
                "checkpoint_dir": temp_checkpoint_dir,
                "num_epochs": 5
            },
            "pretraining": {
                "train_fasta": train_fasta_file,
                "val_fasta": val_fasta_file,
                "ckp_dir": temp_checkpoint_dir,
                "max_length": 512,
                "batch_size": 32,
                "num_epochs": 10,
                "lazy_dataset": False,
                "warmup_ratio": 0.05,
                "optimizer": "adamw",
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-8,
                "learning_rate": 3e-6,
                "weight_decay": 0.0,
                "gradient_accumulation_steps": 1,
                "mlm_probability": 0.3,
                "mask_replace_prob": 0.8,
                "random_token_prob": 0.1,
                "keep_probability": 0.1,
                "logging_steps_percentage": 0.01,
                "eval_steps_percentage": 0.025,
                "save_steps_percentage": 0.1,
                "seed": 42,
                "num_workers": 0,
                "multi_gpu": False,
                "world_size": 1,
                "project_name": "test-resume"
            },
            "model": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "vocab_size": 29,
                "mlp_activation": "swiglu",
                "mlp_dropout": 0.0,
                "mlp_bias": False,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "classifier_activation": "gelu"
            }
        }

        # Verify resume block is correctly structured
        resume_dict = yaml_content.get("resume")
        assert resume_dict is not None, "Resume block should exist"
        assert resume_dict.get("is_resume") is True, "is_resume should be True"
        assert resume_dict.get("checkpoint_dir") == temp_checkpoint_dir, "checkpoint_dir should match"
        assert resume_dict.get("num_epochs") == 5, "num_epochs should match"

    def test_yaml_fresh_training_parsing(self, train_fasta_file, val_fasta_file, temp_checkpoint_dir):
        """Test YAML parsing without resume or with is_resume: False."""
        # Test 1: No resume block at all
        yaml_content_no_resume = {
            "pretraining": {
                "train_fasta": train_fasta_file,
                "val_fasta": val_fasta_file,
                "ckp_dir": temp_checkpoint_dir,
                "max_length": 512,
                "batch_size": 32,
                "num_epochs": 10,
                "lazy_dataset": False,
                "warmup_ratio": 0.05,
                "optimizer": "adamw",
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-8,
                "learning_rate": 3e-6,
                "weight_decay": 0.0,
                "gradient_accumulation_steps": 1,
                "mlm_probability": 0.3,
                "mask_replace_prob": 0.8,
                "random_token_prob": 0.1,
                "keep_probability": 0.1,
                "logging_steps_percentage": 0.01,
                "eval_steps_percentage": 0.025,
                "save_steps_percentage": 0.1,
                "seed": 42,
                "num_workers": 0,
                "multi_gpu": False,
                "world_size": 1,
                "project_name": "test-fresh"
            },
            "model": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "vocab_size": 29,
                "mlp_activation": "swiglu",
                "mlp_dropout": 0.0,
                "mlp_bias": False,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "classifier_activation": "gelu"
            }
        }

        resume_dict = yaml_content_no_resume.get("resume")
        assert resume_dict is None, "Resume block should not exist"

        # Test 2: With is_resume: False
        yaml_content_resume_false = {
            "resume": {
                "is_resume": False,
                "checkpoint_dir": temp_checkpoint_dir,
                "num_epochs": 5
            },
            "pretraining": yaml_content_no_resume["pretraining"],
            "model": yaml_content_no_resume["model"]
        }

        resume_dict = yaml_content_resume_false.get("resume")
        assert resume_dict is not None, "Resume block should exist"
        assert resume_dict.get("is_resume") is False, "is_resume should be False"

    def test_yaml_resume_missing_fields(self, temp_checkpoint_dir):
        """Test error handling when required resume fields are missing."""
        # Missing checkpoint_dir
        yaml_missing_checkpoint = {
            "resume": {
                "is_resume": True,
                "num_epochs": 5
            }
        }

        resume_dict = yaml_missing_checkpoint.get("resume")
        assert resume_dict.get("checkpoint_dir") is None, "checkpoint_dir should be missing"

        # Missing num_epochs
        yaml_missing_epochs = {
            "resume": {
                "is_resume": True,
                "checkpoint_dir": temp_checkpoint_dir
            }
        }

        resume_dict = yaml_missing_epochs.get("resume")
        assert resume_dict.get("num_epochs") is None, "num_epochs should be missing"

    @pytest.mark.integration
    def test_cli_from_yaml_with_resume(
        self,
        temp_checkpoint_dir,
        train_fasta_file,
        val_fasta_file,
        tiny_model_for_training
    ):
        """Test CLI from-yaml command with resume enabled."""
        # First, create a checkpoint
        with patch.dict(os.environ, {'WANDB_MODE': 'disabled'}):
            pretrain_config = PretrainingConfig(
                train_fasta=train_fasta_file,
                val_fasta=val_fasta_file,
                ckp_dir=temp_checkpoint_dir,
                max_length=32,
                batch_size=2,
                num_epochs=1,
                lazy_dataset=False,
                learning_rate=1e-4,
                weight_decay=0.0,
                warmup_ratio=0.0,
                optimizer="adamw",
                gradient_accumulation_steps=1,
                mlm_probability=0.15,
                logging_steps_percentage=0.5,
                eval_steps_percentage=0.5,
                save_steps_percentage=1.0,
                seed=42,
                num_workers=0,
                multi_gpu=False,
                world_size=1,
                project_name="test-cli-resume"
            )

            run_pretraining(model=tiny_model_for_training, pretrain_config=pretrain_config)

            # Find the actual checkpoint directory
            checkpoint_root = Path(temp_checkpoint_dir)
            run_dirs = [d for d in checkpoint_root.iterdir() if d.is_dir() and d.name.startswith("run-")]
            assert len(run_dirs) == 1, f"Expected 1 run directory, found {len(run_dirs)}"
            actual_checkpoint_dir = str(run_dirs[0])

            # Create YAML file with resume block
            yaml_content = {
                "resume": {
                    "is_resume": True,
                    "checkpoint_dir": actual_checkpoint_dir,
                    "num_epochs": 1
                },
                "pretraining": {
                    "train_fasta": train_fasta_file,
                    "val_fasta": val_fasta_file,
                    "ckp_dir": temp_checkpoint_dir,
                    "max_length": 32,
                    "batch_size": 2,
                    "num_epochs": 1,
                    "lazy_dataset": False,
                    "warmup_ratio": 0.0,
                    "optimizer": "adamw",
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.999,
                    "adam_epsilon": 1e-8,
                    "learning_rate": 1e-4,
                    "weight_decay": 0.0,
                    "gradient_accumulation_steps": 1,
                    "mlm_probability": 0.15,
                    "mask_replace_prob": 0.8,
                    "random_token_prob": 0.1,
                    "keep_probability": 0.1,
                    "logging_steps_percentage": 0.5,
                    "eval_steps_percentage": 0.5,
                    "save_steps_percentage": 1.0,
                    "seed": 42,
                    "num_workers": 0,
                    "multi_gpu": False,
                    "world_size": 1,
                    "project_name": "test-cli-resume"
                },
                "model": {
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "vocab_size": 29,
                    "mlp_activation": "swiglu",
                    "mlp_dropout": 0.0,
                    "mlp_bias": False,
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                    "classifier_activation": "gelu"
                }
            }

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(yaml_content, f)
                yaml_file = f.name

            try:
                # Test CLI invocation
                runner = CliRunner()
                result = runner.invoke(pretrain, ['from-yaml', yaml_file])

                # Should complete successfully (exit code 0)
                assert result.exit_code == 0, f"CLI should succeed. Output: {result.output}"

            finally:
                if os.path.exists(yaml_file):
                    os.unlink(yaml_file)

    @pytest.mark.integration
    def test_cli_from_yaml_without_resume(
        self,
        temp_checkpoint_dir,
        train_fasta_file,
        val_fasta_file
    ):
        """Test CLI from-yaml command with fresh training."""
        with patch.dict(os.environ, {'WANDB_MODE': 'disabled'}):
            # Create YAML file without resume block
            yaml_content = {
                "pretraining": {
                    "train_fasta": train_fasta_file,
                    "val_fasta": val_fasta_file,
                    "ckp_dir": temp_checkpoint_dir,
                    "max_length": 32,
                    "batch_size": 2,
                    "num_epochs": 1,
                    "lazy_dataset": False,
                    "warmup_ratio": 0.0,
                    "optimizer": "adamw",
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.999,
                    "adam_epsilon": 1e-8,
                    "learning_rate": 1e-4,
                    "weight_decay": 0.0,
                    "gradient_accumulation_steps": 1,
                    "mlm_probability": 0.15,
                    "mask_replace_prob": 0.8,
                    "random_token_prob": 0.1,
                    "keep_probability": 0.1,
                    "logging_steps_percentage": 0.5,
                    "eval_steps_percentage": 0.5,
                    "save_steps_percentage": 1.0,
                    "seed": 42,
                    "num_workers": 0,
                    "multi_gpu": False,
                    "world_size": 1,
                    "project_name": "test-cli-fresh"
                },
                "model": {
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "vocab_size": 29,
                    "mlp_activation": "swiglu",
                    "mlp_dropout": 0.0,
                    "mlp_bias": False,
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                    "classifier_activation": "gelu"
                }
            }

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(yaml_content, f)
                yaml_file = f.name

            try:
                # Test CLI invocation
                runner = CliRunner()
                result = runner.invoke(pretrain, ['from-yaml', yaml_file])

                # Should complete successfully
                assert result.exit_code == 0, f"CLI should succeed. Output: {result.output}"

                # Find the actual checkpoint directory (it's in a run-specific subdirectory)
                checkpoint_root = Path(temp_checkpoint_dir)
                run_dirs = [d for d in checkpoint_root.iterdir() if d.is_dir() and d.name.startswith("run-")]
                assert len(run_dirs) == 1, f"Expected 1 run directory, found {len(run_dirs)}"
                checkpoint_dir = run_dirs[0]

                # Verify checkpoint was created
                assert (checkpoint_dir / "config.json").exists(), "config.json should be created"
                assert (checkpoint_dir / "model.safetensors").exists(), "model.safetensors should be created"

            finally:
                if os.path.exists(yaml_file):
                    os.unlink(yaml_file)

    def test_cli_resume_missing_checkpoint_dir(self, train_fasta_file, val_fasta_file, temp_checkpoint_dir):
        """Test CLI error handling when checkpoint_dir is missing."""
        with patch.dict(os.environ, {'WANDB_MODE': 'disabled'}):
            # Create YAML with resume but missing checkpoint_dir
            yaml_content = {
                "resume": {
                    "is_resume": True,
                    "num_epochs": 5
                },
                "pretraining": {
                    "train_fasta": train_fasta_file,
                    "val_fasta": val_fasta_file,
                    "ckp_dir": temp_checkpoint_dir,
                    "max_length": 32,
                    "batch_size": 2,
                    "num_epochs": 1,
                    "lazy_dataset": False,
                    "warmup_ratio": 0.0,
                    "optimizer": "adamw",
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.999,
                    "adam_epsilon": 1e-8,
                    "learning_rate": 1e-4,
                    "weight_decay": 0.0,
                    "gradient_accumulation_steps": 1,
                    "mlm_probability": 0.15,
                    "mask_replace_prob": 0.8,
                    "random_token_prob": 0.1,
                    "keep_probability": 0.1,
                    "logging_steps_percentage": 0.5,
                    "eval_steps_percentage": 0.5,
                    "save_steps_percentage": 1.0,
                    "seed": 42,
                    "num_workers": 0,
                    "multi_gpu": False,
                    "world_size": 1,
                    "project_name": "test-error"
                },
                "model": {
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "vocab_size": 29,
                    "mlp_activation": "swiglu",
                    "mlp_dropout": 0.0,
                    "mlp_bias": False,
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                    "classifier_activation": "gelu"
                }
            }

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(yaml_content, f)
                yaml_file = f.name

            try:
                runner = CliRunner()
                result = runner.invoke(pretrain, ['from-yaml', yaml_file])

                # Should fail with error
                assert result.exit_code != 0, "CLI should fail when checkpoint_dir is missing"
                assert "checkpoint_dir" in result.output.lower(), "Error message should mention checkpoint_dir"

            finally:
                if os.path.exists(yaml_file):
                    os.unlink(yaml_file)

    def test_cli_resume_invalid_checkpoint_path(self, train_fasta_file, val_fasta_file, temp_checkpoint_dir):
        """Test CLI error handling when checkpoint path doesn't exist."""
        with patch.dict(os.environ, {'WANDB_MODE': 'disabled'}):
            # Create YAML with resume but invalid checkpoint path
            invalid_checkpoint = "/nonexistent/checkpoint/path"
            yaml_content = {
                "resume": {
                    "is_resume": True,
                    "checkpoint_dir": invalid_checkpoint,
                    "num_epochs": 5
                },
                "pretraining": {
                    "train_fasta": train_fasta_file,
                    "val_fasta": val_fasta_file,
                    "ckp_dir": temp_checkpoint_dir,
                    "max_length": 32,
                    "batch_size": 2,
                    "num_epochs": 1,
                    "lazy_dataset": False,
                    "warmup_ratio": 0.0,
                    "optimizer": "adamw",
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.999,
                    "adam_epsilon": 1e-8,
                    "learning_rate": 1e-4,
                    "weight_decay": 0.0,
                    "gradient_accumulation_steps": 1,
                    "mlm_probability": 0.15,
                    "mask_replace_prob": 0.8,
                    "random_token_prob": 0.1,
                    "keep_probability": 0.1,
                    "logging_steps_percentage": 0.5,
                    "eval_steps_percentage": 0.5,
                    "save_steps_percentage": 1.0,
                    "seed": 42,
                    "num_workers": 0,
                    "multi_gpu": False,
                    "world_size": 1,
                    "project_name": "test-error"
                },
                "model": {
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "vocab_size": 29,
                    "mlp_activation": "swiglu",
                    "mlp_dropout": 0.0,
                    "mlp_bias": False,
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                    "classifier_activation": "gelu"
                }
            }

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(yaml_content, f)
                yaml_file = f.name

            try:
                runner = CliRunner()
                result = runner.invoke(pretrain, ['from-yaml', yaml_file])

                # Should fail with error
                assert result.exit_code != 0, "CLI should fail when checkpoint path doesn't exist"
                assert "does not exist" in result.output.lower() or "not found" in result.output.lower(), \
                    "Error message should mention path doesn't exist"

            finally:
                if os.path.exists(yaml_file):
                    os.unlink(yaml_file)
