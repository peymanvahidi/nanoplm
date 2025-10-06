import torch
import wandb
from pathlib import Path
from safetensors.torch import load_file
from transformers import (
    TrainingArguments, 
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup
)
import time
from torch.optim import AdamW
import os

from nanoplm.distillation.collator import DistillDataCollator
from nanoplm.distillation.trainer import DistillationTrainer
from nanoplm.distillation.session_manager import TrainingSessionManager

from nanoplm.models.student import ProtX
from nanoplm.models.teacher import BaseTeacher, ProtT5

from nanoplm.data.dataset import KDDatasetOnTheFly, LoadKDDataset, LoadKDDatasetOptimized
from nanoplm.utils import get_device, logger

class DistillationPipeline():
    def __init__(
        self,
        train_file: str,
        val_file: str,
        protx_train_prefix: str,
        protx_val_prefix: str,
        student_embed_dim: int,
        student_num_layers: int,
        student_num_heads: int,
        on_the_fly: bool,
        multi_gpu: bool,
        num_epochs: int,
        batch_size: int,
        max_lr: float,
        max_grad_norm: float, # Gradient clipping threshold
        max_seqs_num: int,
        max_seq_len: int,
        val_ratio: float,
        num_workers: int,
        project_name: str,
        checkpoint_dir: str,  # To continue training from a checkpoint
        wandb_dir: str,
        device: str,
        lr_scheduler: str = "cosine",  # New parameter for scheduler type
        lr_scheduler_kwargs: dict = None,  # New parameter for scheduler kwargs
        sharded: bool = False,  # New parameter for sharded data loading
        use_optimized_loader: bool = True,  # NEW: Use optimized data loader
        max_open_files: int = 5,  # NEW: Max open H5 files in cache
        chunk_size: int = 32,  # NEW: Samples to read per chunk
        prefetch_batches: int = 2,  # NEW: Background prefetch batches
        use_threading: bool = True,  # NEW: Enable threading for I/O
        gradient_accumulation_steps: int = 2,  # NEW: Gradient accumulation steps
        projection_layer: bool = True,  # NEW: Whether to include projection layer
        _overrides: dict = None,
    ):
        self.train_file = train_file
        self.val_file = val_file
        self.protx_train_prefix = protx_train_prefix
        self.protx_val_prefix = protx_val_prefix
        self.student_embed_dim = student_embed_dim
        self.student_num_layers = student_num_layers
        self.student_num_heads = student_num_heads
        self.on_the_fly = on_the_fly
        self.multi_gpu = multi_gpu
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_lr = max_lr
        self.max_grad_norm = max_grad_norm
        self.max_seqs_num = max_seqs_num
        self.max_seq_len = max_seq_len
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.project_name = project_name
        self.checkpoint_dir = checkpoint_dir
        self.wandb_dir = wandb_dir
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}
        self.sharded = sharded
        self.use_optimized_loader = use_optimized_loader
        self.max_open_files = max_open_files
        self.chunk_size = chunk_size
        self.prefetch_batches = prefetch_batches
        self.use_threading = use_threading
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.projection_layer = projection_layer
        self._overrides = _overrides or {}
        
        # Store original values for proper resumption
        if not hasattr(self, '_original_lr'):
            self._original_lr = self.max_lr

        # Apply any overrides
        for key, value in self._overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.is_main_process = self.local_rank == 0

    def train(self):

        student = ProtX(
            embed_dim=self.student_embed_dim,
            num_layers=self.student_num_layers,
            num_heads=self.student_num_heads,
            projection_layer=self.projection_layer,
        )

        # Setup training session (new or resumed)
        session_manager = TrainingSessionManager(
            checkpoint_dir=self.checkpoint_dir,
            wandb_dir=self.wandb_dir,
            project_name=self.project_name
        )
        
        # Prepare configuration for saving (for new sessions)
        distill_pipeline_config = {
            "train_file": self.train_file,
            "val_file": self.val_file,
            "protx_train_prefix": self.protx_train_prefix,
            "protx_val_prefix": self.protx_val_prefix,
            "student_embed_dim": self.student_embed_dim,
            "student_num_layers": self.student_num_layers,
            "student_num_heads": self.student_num_heads,
            "on_the_fly": self.on_the_fly,
            "multi_gpu": self.multi_gpu,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "max_lr": self.max_lr,
            "max_seqs_num": self.max_seqs_num,
            "max_seq_len": self.max_seq_len,
            "val_ratio": self.val_ratio,
            "num_workers": self.num_workers,
            "project_name": self.project_name,
            "wandb_dir": self.wandb_dir,
            "device": self.device,
            "lr_scheduler": self.lr_scheduler,
            "lr_scheduler_kwargs": self.lr_scheduler_kwargs,
            # "max_grad_norm": self.max_grad_norm,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "projection_layer": self.projection_layer,
        }
        
        run_name, output_dir, is_resuming = session_manager.setup_session(distill_pipeline_config)

        if is_resuming:
            checkpoint_path = Path(self.checkpoint_dir)
            safetensors_path = checkpoint_path / "model.safetensors"

            model_loaded = False
            if safetensors_path.exists():
                logger.info(f"Loading model weights from {safetensors_path}")
                state_dict = load_file(safetensors_path, device=self.device)
                student.load_state_dict(state_dict)
                model_loaded = True
            if not model_loaded:
                logger.warning(f"Could not find model weights in {self.checkpoint_dir}. Training from scratch.")

        teacher_model_for_collator = None
        
        teacher = None
        if self.on_the_fly:
            teacher = ProtT5()
            teacher_model_for_collator = teacher.encoder_model
        
        train_dataset, val_dataset = self._load_dataset(
            teacher=teacher,
        )

        data_collator = DistillDataCollator(
            teacher_model=teacher_model_for_collator,
            on_the_fly=self.on_the_fly
        )

        # Setup GPU configuration
        world_size, effective_batch_size, gradient_accumulation_steps = self._batch_config()

        # Calculate num_training_steps for scheduler
        # With gradient accumulation, we need fewer training steps since each step processes more samples
        num_training_steps = ((self.max_seqs_num * (1 - self.val_ratio)) // effective_batch_size) * self.num_epochs

        logger.info(f"Training configuration:")
        logger.info(f"  Multi-GPU: {self.multi_gpu}")
        logger.info(f"  World size: {world_size}")
        logger.info(f"  Per-device batch size: {self.batch_size}")
        logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {effective_batch_size}")
        logger.info(f"  Total training steps: {num_training_steps}")
        logger.info(f"  Training samples: {int(self.max_seqs_num * (1 - self.val_ratio))}")

        # Reduce evaluation frequency for HDD to minimize I/O overhead
        eval_steps = max(1, int(num_training_steps*0.01))  # 1% of training steps
        save_steps = eval_steps * 5  # Save every 2 evaluations (~5% of training)

        trainer_dict = {
            "output_dir": str(output_dir),
            "num_train_epochs": self.num_epochs,
            "max_steps": int(num_training_steps),
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "warmup_steps": int(num_training_steps*0.05),
            "learning_rate": self.max_lr,
            "logging_dir": str(output_dir / "logs"),
            "logging_strategy": "steps",
            "logging_steps": eval_steps,
            "save_strategy": "steps",
            "save_steps": save_steps,
            "eval_strategy": "steps",
            "eval_steps": eval_steps,
            "report_to": "wandb",
            "run_name": run_name,
            "dataloader_num_workers": self.num_workers,
            "remove_unused_columns": False,
            "label_names": ["teacher_embeddings"],
            "gradient_accumulation_steps": gradient_accumulation_steps,
            # "max_grad_norm": self.max_grad_norm,
            # "dataloader_pin_memory": True,
            # "dataloader_prefetch_factor": 2,
        }

        if self.multi_gpu:
            trainer_dict["ddp_backend"] = "nccl" if torch.cuda.is_available() else "gloo"
            trainer_dict["ddp_find_unused_parameters"] = True

        training_args = TrainingArguments(**trainer_dict)

        if self.is_main_process:
            wandb_config = session_manager.setup_wandb_config(
                run_name=run_name,
                training_args=training_args,
                is_resuming=is_resuming
            )
            wandb.init(**wandb_config)
        
        optimizer = AdamW(
            student.parameters(),
            lr=self.max_lr
        )
        
        num_training_steps_for_scheduler = num_training_steps
        
        # When resuming, the number of epochs is the number of *additional* epochs.
        # We need to adjust the total steps for the scheduler accordingly.
        if is_resuming:
            # We assume the new `num_epochs` is the total desired number of epochs.
            # To calculate remaining steps, we'd need to know the completed steps.
            # For simplicity and correctness with a new scheduler, we base steps on new num_epochs.
            logger.info("Creating a new scheduler for the resumed training.")
        
        scheduler = self._get_scheduler(optimizer, num_training_steps_for_scheduler)
        
        # Always pass optimizers as a tuple (optimizer, scheduler)
        # Even for constant learning rate, pass (optimizer, None)
        trainer = DistillationTrainer(
            model=student,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            optimizers=(optimizer, scheduler),  # scheduler can be None for constant learning rate
            # callbacks=[onnx_export_callback]
        )
        
        logger.info(f"Starting training with Hugging Face Trainer. Output dir: {output_dir}")
        if self.is_main_process:
            wandb.config.update(training_args.to_dict())
        
        # When resuming with a new learning rate, we need to start with a fresh
        # optimizer and scheduler. The HF Trainer would otherwise load the old
        # state from `optimizer.pt`, overwriting our new LR.
        # The cleanest workaround is to temporarily remove the optimizer and
        # scheduler state files if they exist.
        if is_resuming and self._overrides.get("max_lr") is not None:
            logger.info("New learning rate provided for resumed training. Removing old optimizer/scheduler state.")
            optimizer_path = Path(self.checkpoint_dir) / "optimizer.pt"
            scheduler_path = Path(self.checkpoint_dir) / "scheduler.pt"
            
            if optimizer_path.exists():
                optimizer_path.unlink()
                logger.info(f"Removed {optimizer_path}")
            
            if scheduler_path.exists():
                scheduler_path.unlink()
                logger.info(f"Removed {scheduler_path}")

        # Train with or without checkpoint resumption
        if is_resuming:
            logger.info(f"Resuming training from checkpoint: {self.checkpoint_dir}")
            train_result = trainer.train(resume_from_checkpoint=self.checkpoint_dir)
        else:
            train_result = trainer.train()
        
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        if val_dataset:
            logger.info(f"Evaluating on {len(val_dataset)} samples")
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        if self.is_main_process:
            wandb.finish()
            
            logger.info(f"Training complete!")

    def _load_dataset(
        self,
        teacher: BaseTeacher = None,
        seed: int = None
    ):
        if self.on_the_fly:
            train_dataset = KDDatasetOnTheFly(
                input_fasta=self.train_file,
                teacher=teacher,
                max_seq_len=self.max_seq_len,
                device=str(self.device)
            )
            val_dataset = KDDatasetOnTheFly(
                input_fasta=self.val_file,
                teacher=teacher,
                max_seq_len=self.max_seq_len,
                device=str(self.device)
            )
        else:
            if self.use_optimized_loader:
                logger.info("Using LoadKDDatasetOptimized for better performance")
                train_dataset = LoadKDDatasetOptimized(
                    h5_path=self.protx_train_prefix,
                    device=str(self.device),
                    seed=seed if seed is not None else int(time.time()),
                    sharded=self.sharded,
                    max_open_files=self.max_open_files,
                    chunk_size=self.chunk_size,
                    prefetch_batches=self.prefetch_batches,
                    use_threading=self.use_threading
                )
                val_dataset = LoadKDDatasetOptimized(
                    h5_path=self.protx_val_prefix,
                    device=str(self.device),
                    seed=seed if seed is not None else int(time.time()) + 1,
                    sharded=self.sharded,
                    max_open_files=self.max_open_files,
                    chunk_size=self.chunk_size,
                    prefetch_batches=self.prefetch_batches,
                    use_threading=self.use_threading
                )
            else:
                logger.info("Using standard LoadKDDataset")
                train_dataset = LoadKDDataset(
                    h5_path=self.protx_train_prefix,
                    device=str(self.device),
                    seed=seed if seed is not None else int(time.time()),
                    sharded=self.sharded
                )
                val_dataset = LoadKDDataset(
                    h5_path=self.protx_val_prefix,
                    device=str(self.device),
                    seed=seed if seed is not None else int(time.time()) + 1,
                    sharded=self.sharded
                )
        
        return train_dataset, val_dataset

    def _batch_config(self):
        world_size = self.world_size if self.multi_gpu else 1
        effective_batch_size = self.batch_size * world_size * self.gradient_accumulation_steps
        os.environ["WANDB_LOG_MODEL"] = "end"

        return world_size, effective_batch_size, self.gradient_accumulation_steps

    def _get_scheduler(self, optimizer, num_training_steps):
        if num_training_steps <= 0:
            logger.warning("Number of training steps is 0 or less. No scheduler will be created.")
            return None

        logger.info(f"Creating {self.lr_scheduler} scheduler with {num_training_steps} training steps")
        
        # Set warmup steps to 0 if the user wants a truly constant LR from the start
        # unless they override it via lr_scheduler_kwargs
        warmup_steps = self.lr_scheduler_kwargs.get("num_warmup_steps", int(num_training_steps * 0.05))
        if self.lr_scheduler == "constant":
            warmup_steps = self.lr_scheduler_kwargs.get("num_warmup_steps", 0)

        if self.lr_scheduler == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=0.5,
                **self.lr_scheduler_kwargs
            )
        elif self.lr_scheduler == "linear":
            return get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
                **self.lr_scheduler_kwargs
            )
        elif self.lr_scheduler == "polynomial":
            return get_polynomial_decay_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
                **self.lr_scheduler_kwargs
            )
        elif self.lr_scheduler == "constant":
            logger.info("Using constant learning rate scheduler.")
            return get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps
            )
        else:
            raise ValueError(f"Unknown learning rate scheduler: {self.lr_scheduler}")
