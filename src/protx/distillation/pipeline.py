import torch
import wandb
from pathlib import Path
from transformers import PreTrainedTokenizer, TrainingArguments, get_cosine_schedule_with_warmup
import time
from torch.optim import AdamW
import os

from .collator import DistillDataCollator
# from .callbacks import OnnxExportCallback
from .trainer import DistillationTrainer
# from .scheduler import ProtXScheduler

from ..models.student import ProtX
from ..models.teacher import ProtT5

from ..data.dataset import ProtXDataGen, ProtXDataLoader
from ..utils import get_device, create_dirs, logger

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
        max_seqs_num: int,
        max_seq_len: int,
        val_ratio: float,
        num_workers: int,
        project_name: str,
        checkpoint_path: str,
        wandb_dir: str,
        device: str,
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
        self.max_seqs_num = max_seqs_num
        self.max_seq_len = max_seq_len
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.project_name = project_name
        self.checkpoint_path = checkpoint_path
        self.wandb_dir = wandb_dir
        self.device = device if device else get_device()


        # Initialize distributed training if multi_gpu is enabled
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main_process = self.rank == 0

    def train(self):

        student = ProtX(
            embed_dim=self.student_embed_dim,
            num_layers=self.student_num_layers,
            num_heads=self.student_num_heads,
        )

        teacher = ProtT5()

        timestamp = int(time.time())
        run_name = f"run-{timestamp}"

        output_dir = Path(self.wandb_dir) / run_name
        create_dirs(output_dir)

        teacher_model_for_collator = None
        teacher_tokenizer_for_dataset = None
        
        if self.on_the_fly:
            teacher_model_for_collator = teacher.encoder_model
            teacher_tokenizer_for_dataset = teacher.tokenizer
        
        train_dataset, val_dataset = self._load_dataset(
            teacher_tokenizer=teacher_tokenizer_for_dataset,
        )

        data_collator = DistillDataCollator(
            teacher_model=teacher_model_for_collator,
            on_the_fly=self.on_the_fly
        )

        # Setup GPU configuration
        world_size, effective_batch_size = self._gpu_config()

        # Calculate num_training_steps for scheduler
        num_training_steps = ((self.max_seqs_num * (1 - self.val_ratio)) // effective_batch_size) * self.num_epochs

        logger.info(f"Training configuration:")
        logger.info(f"  Multi-GPU: {self.multi_gpu}")
        logger.info(f"  World size: {world_size}")
        logger.info(f"  Per-device batch size: {self.batch_size}")
        logger.info(f"  Effective batch size: {effective_batch_size}")
        logger.info(f"  Total training steps: {num_training_steps}")
        logger.info(f"  Training samples: {int(self.max_seqs_num * (1 - self.val_ratio))}")

        # Reduce evaluation frequency for HDD to minimize I/O overhead
        eval_steps = max(1, int(num_training_steps*0.05))  # 5% of training steps (reduced from 1%)
        save_steps = eval_steps * 2  # Save every 2 evaluations (~10% of training)

        training_dict = {
            "output_dir": str(output_dir),
            "num_train_epochs": self.num_epochs,
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size * 2,
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
            # Performance optimizations for HDD + multi-GPU
            "dataloader_prefetch_factor": 25,  # Prefetch more batches to hide HDD latency
            "dataloader_persistent_workers": True,  # Keep workers alive
            "gradient_accumulation_steps": 1,  # Increase effective batch size, reduce I/O frequency
            "prediction_loss_only": True,  # Speed up evaluation by not computing extra outputs
            "load_best_model_at_end": False,  # Skip loading best model at end to save I/O
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        }

        if self.multi_gpu:
            training_dict["fp16"] = True
            training_dict["dataloader_pin_memory"] = True
            training_dict["ddp_backend"] = "nccl" if torch.cuda.is_available() else "gloo"
            # Additional multi-GPU optimizations
            training_dict["ddp_find_unused_parameters"] = False
            training_dict["ddp_bucket_cap_mb"] = 25  # Reduce bucket size for better memory efficiency
            training_dict["bf16"] = torch.cuda.is_bf16_supported()  # Use bf16 if available (better than fp16)
            if training_dict["bf16"]:
                training_dict["fp16"] = False  # Use bf16 instead of fp16 if supported

        training_args = TrainingArguments(**training_dict)

        if self.is_main_process:
            wandb.init(
                project=self.project_name,
                name=run_name,
                config=training_args.to_dict(),
                settings=wandb.Settings(start_method="fork")
            )

        
        # ONNX export disabled - will be handled by separate script after training
        # onnx_export_callback = OnnxExportCallback(
        #     onnx_export_path=str(output_dir / "onnx"),
        #     batch_size=self.distill_config.batch_size,
        #     seq_len=self.data_config.max_seq_len,
        #     device=str(self.device)
        # )
        
        optimizer = AdamW(
            student.parameters(),
            lr=self.max_lr
        )
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(num_training_steps * 0.05),
            num_training_steps=num_training_steps,
            num_cycles=0.5
        )
        
        trainer = DistillationTrainer(
            model=student,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            optimizers=(optimizer, scheduler),
            # callbacks=[onnx_export_callback]
        )
        
        if self.is_main_process:
            logger.info(f"Starting training with Hugging Face Trainer. Output dir: {output_dir}")
            wandb.config.update(training_args.to_dict())
        
        train_result = trainer.train(resume_from_checkpoint=self.checkpoint_path)
        
        if self.is_main_process:
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
            
            best_model_path = str(output_dir)
            wandb.run.log_artifact(
                artifact_or_path=best_model_path,
                name=f"best-model-{run_name}",
                type="model",
                aliases=["best", "latest", "production"]
            )

            logger.info(f"Best model saved as wandb artifact: best-model-{run_name}")

            wandb.finish()
            
            logger.info(f"Training complete!")

    def _load_dataset(
        self,
        teacher_tokenizer: PreTrainedTokenizer = None,
        seed: int = None
    ):
        if self.on_the_fly:
            train_dataset = ProtXDataGen(
                data_path=self.train_file,
                teacher_tokenizer=teacher_tokenizer,
                max_seq_len=self.max_seq_len,
                device=str(self.device)
            )
            val_dataset = ProtXDataGen(
                data_path=self.val_file,
                teacher_tokenizer=teacher_tokenizer,
                max_seq_len=self.max_seq_len,
                device=str(self.device)
            )
        else:
            train_dataset = ProtXDataLoader(
                h5_path=self.protx_train_prefix,
                device=str(self.device),
                seed=seed if seed is not None else int(time.time())
            )
            val_dataset = ProtXDataLoader(
                h5_path=self.protx_val_prefix,
                device=str(self.device),
                seed=seed if seed is not None else int(time.time()) + 1
            )
        
        return train_dataset, val_dataset

    def _gpu_config(self):
        """
        Configure GPU settings and environment variables for training.

        Returns:
            tuple: (world_size, effective_batch_size)
        """
        gradient_accumulation_steps = 1  # Updated to match training config

        # Determine the world size (number of GPUs/processes)
        world_size = self.world_size if self.multi_gpu else 1

        # Calculate effective batch size (per_device_batch_size * world_size * gradient_accumulation_steps)
        effective_batch_size = self.batch_size * world_size * gradient_accumulation_steps

        # Configure environment variables for performance
        os.environ["WANDB_LOG_MODEL"] = "end"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings in multi-processing
        # With 16 CPUs per GPU, use more threads for better performance
        os.environ["OMP_NUM_THREADS"] = str(min(12, self.num_workers * 2))  # Use 12 threads or 2x num_workers

        return world_size, effective_batch_size