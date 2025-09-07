from nanoplm.distillation.pipeline import DistillationPipeline
from nanoplm.distillation.session_manager import TrainingSessionManager
from nanoplm.utils import logger
from typing import Optional

class DistillationPipelineBuilder:
    """
    Builder class for DistillationPipeline that supports easy resuming from checkpoints.
    
    When resuming, you only need to specify:
    - checkpoint_dir: Path to checkpoint to resume from
    - Any parameters you want to override (like num_epochs)
    
    Usage:
        # Fresh training
        builder = DistillationPipelineBuilder()
        pipeline = builder.with_training_files(
            train_file="path/to/train.fasta",
            val_file="path/to/val.fasta"
        ).with_model_config(
            student_embed_dim=512,
            student_num_layers=6
        ).build()
        
        # Resume from checkpoint (only specify what you want to change)
        builder = DistillationPipelineBuilder()
        pipeline = builder.resume_from_checkpoint(
            checkpoint_dir="path/to/checkpoint",
            num_epochs=20  # Continue training for 10 more epochs
        )
    """
    
    def __init__(self):
        self.config = {}
    
    def with_training_files(
        self,
        train_file: str,
        val_file: str,
        protx_train_prefix: str,
        protx_val_prefix: str
    ):
        """Set training file paths."""
        self.config.update({
            "train_file": train_file,
            "val_file": val_file,
            "protx_train_prefix": protx_train_prefix,
            "protx_val_prefix": protx_val_prefix,
            "checkpoint_dir": None,
        })
        return self
    
    def with_model_config(
        self,
        student_embed_dim: int,
        student_num_layers: int,
        student_num_heads: int,
        projection_layer: bool = True
    ):
        """Set student model architecture."""
        self.config.update({
            "student_embed_dim": student_embed_dim,
            "student_num_layers": student_num_layers,
            "student_num_heads": student_num_heads,
            "projection_layer": projection_layer,
        })
        return self
    
    def with_training_config(
        self,
        num_epochs: int,
        batch_size: int,
        max_lr: float,
        max_grad_norm: Optional[float],
        max_seqs_num: int,
        max_seq_len: int,
        val_ratio: float,
        num_workers: int,
        lr_scheduler: str = "cosine",
        lr_scheduler_kwargs: dict = None,
    ):
        """Set training hyperparameters."""
        self.config.update({
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "max_lr": max_lr,
            "max_grad_norm": max_grad_norm,
            "max_seqs_num": max_seqs_num,
            "max_seq_len": max_seq_len,
            "val_ratio": val_ratio,
            "num_workers": num_workers,
            "lr_scheduler": lr_scheduler,
            "lr_scheduler_kwargs": lr_scheduler_kwargs or {},
        })
        return self
    
    def with_experiment_config(
        self,
        project_name: str,
        wandb_dir: str,
        device: str,
        on_the_fly: bool,
        multi_gpu: bool,
        sharded: bool = False,
        use_optimized_loader: bool = True,
        max_open_files: int = 5,
        chunk_size: int = 32,
        prefetch_batches: int = 2,
        use_threading: bool = True
    ):
        """Set experiment and infrastructure configuration."""
        self.config.update({
            "project_name": project_name,
            "wandb_dir": wandb_dir,
            "device": device,
            "on_the_fly": on_the_fly,
            "multi_gpu": multi_gpu,
            "sharded": sharded,
            "use_optimized_loader": use_optimized_loader,
            "max_open_files": max_open_files,
            "chunk_size": chunk_size,
            "prefetch_batches": prefetch_batches,
            "use_threading": use_threading,
        })
        return self
    
    def resume_from_checkpoint(
        self,
        checkpoint_dir: str,
        **overrides
    ) -> DistillationPipeline:
        """
        Resume training from a checkpoint with optional parameter overrides.
        
        Args:
            checkpoint_dir: Path to the checkpoint directory
            **overrides: Any parameters you want to override (e.g., num_epochs=20)
            
        Returns:
            DistillationPipeline instance ready for training
        """
        # Load original configuration
        saved_config = TrainingSessionManager.load_training_config(checkpoint_dir)
        if saved_config is None:
            raise ValueError(
                f"Could not load training configuration from checkpoint: {checkpoint_dir}. "
                "This checkpoint may be from an older version that doesn't save configuration."
            )
        
        # Apply overrides
        final_config = {**saved_config, **overrides}
        final_config["checkpoint_dir"] = checkpoint_dir
        final_config["_overrides"] = overrides
        
        logger.info(f"Resuming training from {checkpoint_dir}")
        if overrides:
            logger.info(f"Overriding parameters: {list(overrides.keys())}")
        
        return DistillationPipeline(**final_config)
    
    def build(self) -> DistillationPipeline:
        """
        Build a DistillationPipeline with the configured parameters.
        
        Returns:
            DistillationPipeline instance ready for training
        """
        # Validate required parameters
        required_params = ["train_file", "val_file"]
        missing_params = [p for p in required_params if p not in self.config or self.config[p] is None]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        return DistillationPipeline(**self.config) 