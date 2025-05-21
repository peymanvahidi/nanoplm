import torch
import wandb
from tqdm import tqdm
from pathlib import Path
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader
import math
import time

from ..models.student import ProtX
from ..models.teacher import ProtT5
from ..config import DataConfig
from ..config.distill_config import DistillConfig
from ..data.dataset import ProtXDataGen, ProtXDataLoader
from ..utils import get_device, create_dirs
from .scheduler import ProtXScheduler

class DistillPipeline():
    def __init__(
        self,
        data_config: DataConfig = DataConfig(),
        distill_config: DistillConfig = DistillConfig(),
        device: str = None
    ):
        self.data_config = data_config
        self.distill_config = distill_config
        self.device = get_device() if not device else device

        val_seqs_num = int(self.data_config.max_seqs_num * self.data_config.val_ratio)
        train_seqs_num = self.data_config.max_seqs_num - val_seqs_num
        self.train_batches_num = math.ceil(train_seqs_num / self.distill_config.batch_size)
            

    def train(
        self,
        student: torch.nn.Module = ProtX(),
        teacher = ProtT5(),
        checkpoint_path: str = None,
        start_epoch: int = 0,
        project_name: str = "protx-distillation"
    ):
        timestamp = int(time.time())
        unique_id = f"{timestamp}"
        
        wandb.init(
            project=project_name,
            config={
                "max_seqs_num": self.data_config.max_seqs_num,
                "max_seq_len": self.data_config.max_seq_len,
                "min_seq_len": self.data_config.min_seq_len,
                "val_ratio": self.data_config.val_ratio,
                
                "num_epochs": self.distill_config.num_epochs,
                "batch_size": self.distill_config.batch_size,
                "warmup_steps": self.distill_config.warmup_steps,
                "start_lr": self.distill_config.start_lr,
                "max_lr": self.distill_config.max_lr,
                "min_lr": self.distill_config.min_lr,
                "T_0": self.distill_config.T_0,
                "T_mult": self.distill_config.T_mult,
                "gamma": self.distill_config.gamma,
                "on_the_fly": self.distill_config.on_the_fly,

                "student_embed_dim": self.distill_config.student_embed_dim,
                "student_num_layers": self.distill_config.student_num_layers,
                "student_num_heads": self.distill_config.student_num_heads,
            },
            name=f"run-{unique_id}"
        )
        
        # Update unique_id to include wandb run ID
        unique_id = f"{timestamp}-{wandb.run.id[:8]}"
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            student.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']

        optimizer = torch.optim.AdamW(student.parameters(), lr=self.distill_config.start_lr)
        
        scheduler = ProtXScheduler(
            optimizer,
            warmup_steps=self.distill_config.warmup_steps,
            initial_lr=self.distill_config.start_lr,
            max_lr=self.distill_config.max_lr,
            min_lr=self.distill_config.min_lr,
            T_0=self.distill_config.T_0,
            T_mult=self.distill_config.T_mult,
            gamma=self.distill_config.gamma,
            max_cycle_length=self.distill_config.max_cycle_length
        )
        
        student = student.to(self.device)
        
        # Only load teacher model to device if computing embeddings on-the-fly
        teacher_model = None
        if self.distill_config.on_the_fly:
            teacher_model = teacher.encoder_model.to(self.device)
        
        train_data, val_data = self._load_dataset(
            teacher_tokenizer=teacher.tokenizer if self.distill_config.on_the_fly else None,
            batch_size=self.distill_config.batch_size,
            seed=timestamp
        )
        
        train_batches_total = len(train_data)
        val_batches_total = len(val_data)
        
        checkpoint_dir = self.distill_config.checkpoint_dir
        plots_dir = self.distill_config.plots_dir
        create_dirs(checkpoint_dir)
        create_dirs(plots_dir)
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        global_step = 0
        
        print(f"\n{'=' * 60}")
        print(f"Starting training: {self.distill_config.num_epochs} epochs")
        print(f"{'=' * 60}\n")

        for epoch in range(start_epoch, self.distill_config.num_epochs):
            if not self.distill_config.on_the_fly and epoch > start_epoch:
                train_data, val_data = self._load_dataset(
                    batch_size=self.distill_config.batch_size,
                    seed=epoch
                )
                # Update batch counts for progress bars
                train_batches_total = len(train_data)
                val_batches_total = len(val_data)
            
            student.train()
            epoch_loss = 0.0
            train_batches = 0
            
            print(f"\n{'-' * 20} Epoch {epoch+1}/{self.distill_config.num_epochs} {'-' * 20}")
            
            train_iter = tqdm(
                train_data, 
                desc=f"Training",
                leave=True,
                ncols=100,
                total=train_batches_total,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )

            for batch in train_iter:
                current_lr = optimizer.param_groups[0]['lr']
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                if self.distill_config.on_the_fly:
                    with torch.no_grad():
                        teacher_embeds = teacher_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        ).last_hidden_state
                else:
                    teacher_embeds = batch["teacher_embeddings"].to(self.device)
                
                loss = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_repr=teacher_embeds
                )

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                train_batches += 1
                train_iter.set_postfix(loss=f"{loss.item():.2f}", lr=f"{current_lr:.6f}")
                
                global_step += 1
                
                wandb.log({
                    "learning_rate": current_lr,
                    "train_loss": loss.item()
                }, step=global_step)
            
            avg_train_loss = epoch_loss / train_batches if train_batches > 0 else 0
            train_losses.append(avg_train_loss)
            
            student.eval()
            val_loss = 0.0
            val_batches = 0
            
            val_iter = tqdm(
                val_data, 
                desc=f"Validation",
                leave=True,
                ncols=100,
                total=val_batches_total,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
            
            with torch.no_grad():
                for batch in val_iter:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    
                    if self.distill_config.on_the_fly:
                        teacher_embeds = teacher_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        ).last_hidden_state
                    else:
                        teacher_embeds = batch["teacher_embeddings"].to(self.device)
                    
                    loss = student(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        target_repr=teacher_embeds
                    )
                    
                    val_loss += loss.item()
                    val_batches += 1
                    val_iter.set_postfix(loss=f"{loss.item():.2f}")
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            val_losses.append(avg_val_loss)
            
            print(f"\nEpoch {epoch+1}/{self.distill_config.num_epochs} Summary:")
            print(f"  Training Loss: {avg_train_loss:.4f}")
            print(f"  Validation Loss: {avg_val_loss:.4f}")
            
            wandb.log({
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss
            }, step=global_step)
            
            model_path = f"{checkpoint_dir}/{unique_id}_e{epoch+1}_tl-{avg_train_loss:.2f}_vl-{avg_val_loss:.2f}.pt"
            torch.save(student.state_dict(), model_path)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = f"{checkpoint_dir}/{unique_id}_best.pt"
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'model_state_dict': student.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    },
                    best_model_path
                )
                print(f"  ✓ New best model saved (val_loss: {best_val_loss:.4f})")
            
            print(f"{'-' * 60}")
        
        print(f"\n{'=' * 60}")
        print(f"Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"{'=' * 60}\n")
        
        wandb.log({"best_val_loss": best_val_loss}, step=global_step)
        wandb.finish()

        return student

    def _load_dataset(
        self,
        teacher_tokenizer: PreTrainedTokenizer = None,
        batch_size: int = 32,
        seed: int = None
    ):
        if self.distill_config.on_the_fly:
            # Use original dataset with on-the-fly embedding computation
            train_dataset = ProtXDataGen(
                data_path=self.data_config.train_file,
                teacher_tokenizer=teacher_tokenizer,
                max_seq_len=self.data_config.max_seq_len,
                device=self.device
            )
            val_dataset = ProtXDataGen(
                data_path=self.data_config.val_file,
                teacher_tokenizer=teacher_tokenizer,
                max_seq_len=self.data_config.max_seq_len,
                device=self.device
            )
        else:

            train_files = self._find_dataset_files(self.data_config.protx_train_prefix, "training")
            val_files = self._find_dataset_files(self.data_config.protx_val_prefix, "validation")
            
            train_dataset = ProtXDataLoader(
                h5_path=train_files,
                device=self.device,
                seed=seed
            )
            val_dataset = ProtXDataLoader(
                h5_path=val_files,
                device=self.device,
                seed=seed
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=self._collate_fn
        )

        return train_loader, val_loader
    
    def _collate_fn(self, batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        if "teacher_embeddings" in batch[0]:
            teacher_embeddings = torch.stack([item["teacher_embeddings"] for item in batch])
            result["teacher_embeddings"] = teacher_embeddings
            
        return result
        
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

if __name__ == "__main__":
    pipeline = DistillPipeline()
    pipeline.train()