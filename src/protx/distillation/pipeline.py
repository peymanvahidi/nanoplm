import torch
import wandb
from pathlib import Path
from transformers import PreTrainedTokenizer, TrainingArguments
import time

from .collator import DistillDataCollator
from .callbacks import OnnxExportCallback
from .trainer import DistillationTrainer

from ..models.student import ProtX
from ..models.teacher import ProtT5
from ..config import DataConfig
from ..config.distill_config import DistillConfig
from ..data.dataset import ProtXDataGen, ProtXDataLoader
from ..utils import get_device, create_dirs, logger

class DistillationPipeline():
    def __init__(
        self,
        data_config: DataConfig = DataConfig(),
        distill_config: DistillConfig = DistillConfig(),
        device: str = None
    ):
        self.data_config = data_config
        self.distill_config = distill_config
        self.device = device if device else get_device()

    def train(
        self,
        checkpoint_path: str = None,
        project_name: str = "protx-distillation"
    ):

        student = ProtX(
            embed_dim=self.distill_config.student_embed_dim,
            num_layers=self.distill_config.student_num_layers,
            num_heads=self.distill_config.student_num_heads,
        )

        teacher = ProtT5()

        timestamp = int(time.time())
        run_name = f"run-{timestamp}"

        output_dir = Path(self.distill_config.checkpoint_dir) / run_name
        create_dirs(output_dir)

        teacher_model_for_collator = None
        teacher_tokenizer_for_dataset = None
        
        if self.distill_config.on_the_fly:
            teacher_model_for_collator = teacher.encoder_model
            teacher_tokenizer_for_dataset = teacher.tokenizer
        
        train_dataset, val_dataset = self._load_dataset(
            teacher_tokenizer=teacher_tokenizer_for_dataset,
        )

        data_collator = DistillDataCollator(
            teacher_model=teacher_model_for_collator,
            on_the_fly=self.distill_config.on_the_fly
        )

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.distill_config.num_epochs,
            per_device_train_batch_size=self.distill_config.batch_size,
            per_device_eval_batch_size=self.distill_config.batch_size * 2,
            warmup_steps=self.distill_config.warmup_steps,
            learning_rate=self.distill_config.max_lr,
            logging_dir=str(output_dir / "logs"),
            logging_strategy="steps",
            logging_steps=30,
            save_strategy="steps",
            save_steps=100,
            eval_strategy="steps",
            eval_steps=30,
            report_to="wandb",
            run_name=run_name,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            lr_scheduler_type="cosine_with_min_lr",
            lr_scheduler_kwargs={"min_lr": self.distill_config.min_lr},
            ddp_find_unused_parameters=False,  # Faster distributed training
            ddp_backend="nccl" if torch.cuda.is_available() else "gloo",  # Optimal backend
            remove_unused_columns=False,  # Keep all columns for custom loss
        )
        
        wandb.init(project=project_name, name=run_name, config=training_args.to_dict(), reinit=True)

        onnx_export_callback = OnnxExportCallback(
            onnx_export_path=str(output_dir / "onnx"),
            batch_size=self.distill_config.batch_size,
            seq_len=self.data_config.max_seq_len,
            device=str(self.device)
        )
        
        trainer = DistillationTrainer(
            model=student,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[onnx_export_callback],
        )
        
        logger.info(f"Starting training with Hugging Face Trainer. Output dir: {output_dir}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint_path)
        
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        if val_dataset:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        wandb.finish()
        
        logger.info(f"Training complete!")
        return student

    def _load_dataset(
        self,
        teacher_tokenizer: PreTrainedTokenizer = None,
        seed: int = None
    ):
        if self.distill_config.on_the_fly:
            train_dataset = ProtXDataGen(
                data_path=self.data_config.train_file,
                teacher_tokenizer=teacher_tokenizer,
                max_seq_len=self.data_config.max_seq_len,
                device=str(self.device)
            )
            val_dataset = ProtXDataGen(
                data_path=self.data_config.val_file,
                teacher_tokenizer=teacher_tokenizer,
                max_seq_len=self.data_config.max_seq_len,
                device=str(self.device)
            )
        else:
            train_files = self._find_dataset_files(self.data_config.protx_train_prefix, "training")
            val_files = self._find_dataset_files(self.data_config.protx_val_prefix, "validation")
            
            train_dataset = ProtXDataLoader(
                h5_path=train_files,
                device=str(self.device),
                seed=seed if seed is not None else int(time.time())
            )
            val_dataset = ProtXDataLoader(
                h5_path=val_files,
                device=str(self.device),
                seed=seed if seed is not None else int(time.time()) + 1
            )
        
        return train_dataset, val_dataset
        
    def _find_dataset_files(
        self,
        file_prefix: str,
        dataset_type: str
    ) -> list:
        
        path = Path(file_prefix)
        directory = path.parent
        base_name = path.stem
        
        pattern = f"{base_name}*.h5"
        files = sorted(list(directory.glob(pattern)))
        
        if not files:
            raise FileNotFoundError(f"No {dataset_type} data files found matching {pattern} in {directory}")
        
        return files
