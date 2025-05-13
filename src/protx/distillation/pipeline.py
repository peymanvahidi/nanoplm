import math
import time
import torch
import mlflow
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader

from ..models.student import ProtX
from ..models.teacher import ProtT5
from ..data.dataset import ProtXDataGen
from ..config import DataConfig, DistillConfig
from ..utils import create_dirs, get_device
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
        start_epoch: int = 0
    ):
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            timestamp = int(time.time())
            unique_id = f"{timestamp}-{run_id[:8]}"
            
            mlflow.log_params({
                "max_seqs_num": self.data_config.max_seqs_num,
                "max_seq_len": self.data_config.max_seq_len,
                "min_seq_len": self.data_config.min_seq_len,
                "val_ratio": self.data_config.val_ratio,
                
                "num_epochs": self.distill_config.num_epochs,
                "batch_size": self.distill_config.batch_size,
                "warmup_steps": self.distill_config.warmup_steps,
                "max_lr": self.distill_config.max_lr,
                "min_lr": self.distill_config.min_lr,
                "coinse_cycle": self.distill_config.coinse_cycle,
                "cosine_factor": self.distill_config.cosine_factor,
                "gamma": self.distill_config.gamma,

                "student_embed_dim": self.distill_config.student_embed_dim,
                "student_num_layers": self.distill_config.student_num_layers,
                "student_num_heads": self.distill_config.student_num_heads,
            })
            
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                student.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']

            # Start with very low LR for warmup
            optimizer = torch.optim.AdamW(student.parameters(), lr=1e-8)
            
            # Use the new scheduler
            scheduler = ProtXScheduler(
                optimizer,
                warmup_steps=self.distill_config.warmup_steps,  # 1000 steps of warmup
                max_lr=self.distill_config.max_lr,  # Target max LR (1e-3)
                min_lr=self.distill_config.min_lr,  # Min LR during cosine cycles
                T_0=self.distill_config.coinse_cycle,  # Length of first cosine cycle
                T_mult=self.distill_config.cosine_factor,  # Multiplicative factor for cycle length
                gamma=self.distill_config.gamma  # Weight decay factor per cycle
            )
            
            student = student.to(self.device)
            teacher_model = teacher.encoder_model.to(self.device)
            
            train_data, val_data = self._load_dataset(
                teacher_tokenizer=teacher.tokenizer,
                batch_size=self.distill_config.batch_size
            )
            
            checkpoint_dir = self.distill_config.checkpoint_dir
            plots_dir = self.distill_config.plots_dir
            create_dirs(checkpoint_dir)
            create_dirs(plots_dir)
            
            best_val_loss = float('inf')
            train_losses = []
            val_losses = []

            for epoch in range(start_epoch, self.distill_config.num_epochs):
                student.train()
                epoch_loss = 0.0
                train_batches = 0
                
                train_iter = tqdm(train_data, desc=f"Epoch {epoch+1}/{self.distill_config.num_epochs} [Train]")

                current_lr = optimizer.param_groups[0]['lr']
                mlflow.log_metric("learning_rate", current_lr, step=epoch)
                print(f"Learning rate: {current_lr:.6f}")

                for batch in train_iter:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    
                    with torch.no_grad():
                        teacher_embeds = teacher_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        ).last_hidden_state
                    
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
                    train_iter.set_postfix(loss=loss.item())
                    
                    mlflow.log_metric("step_train_loss", loss.item(), step=epoch * self.train_batches_num + train_batches)
                    
                avg_train_loss = epoch_loss / train_batches if train_batches > 0 else 0
                train_losses.append(avg_train_loss)
                print(f"Epoch {epoch + 1}/{self.distill_config.num_epochs}, Average Training Loss: {avg_train_loss:.4f}")
                
                mlflow.log_metric("avg_train_loss", avg_train_loss, step=epoch)
                
                student.eval()
                val_loss = 0.0
                val_batches = 0
                
                val_iter = tqdm(val_data, desc=f"Epoch {epoch+1}/{self.distill_config.num_epochs} [Valid]")
                
                with torch.no_grad():
                    for batch in val_iter:
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        
                        teacher_embeds = teacher_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        ).last_hidden_state
                        
                        loss = student(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            target_repr=teacher_embeds
                        )
                        
                        val_loss += loss.item()
                        val_batches += 1
                        val_iter.set_postfix(loss=loss.item())
                        
                        mlflow.log_metric("step_val_loss", loss.item(), step=epoch * self.train_batches_num + val_batches)
                        
                avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
                val_losses.append(avg_val_loss)
                print(f"Epoch {epoch + 1}/{self.distill_config.num_epochs}, Average Validation Loss: {avg_val_loss:.4f}")
                
                mlflow.log_metric("avg_val_loss", avg_val_loss, step=epoch)
                
                model_path = f"{checkpoint_dir}/{unique_id}_e{epoch+1}_tl-{avg_train_loss:.2f}_vl-{avg_val_loss:.2f}.pt"
                torch.save(student.state_dict(), model_path)
                mlflow.log_artifact(model_path)
                
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
                    mlflow.log_artifact(best_model_path)
                    print(f"New best model saved with validation loss: {best_val_loss:.4f}")
                
            plt.figure(figsize=(10, 6))
            epochs = list(range(start_epoch + 1, start_epoch + len(train_losses) + 1))
            plt.plot(epochs, train_losses, 'b-', label='Training Loss')
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
            plt.title(f'Training and Validation Loss (Run {unique_id})')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plot_path = f"{plots_dir}/loss_curves_run-{unique_id}.png"
            plt.savefig(plot_path)
            plt.show()
            
            print(f"Loss curves saved to {plot_path}")
            mlflow.log_artifact(plot_path)
            mlflow.log_metric("best_val_loss", best_val_loss)

        return student

    def _collate_fn(self, batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    def _load_dataset(
        self,
        teacher_tokenizer: PreTrainedTokenizer,
        batch_size:int
    ):
        train_dataset = ProtXDataGen(
            data_path=self.data_config.train_file,
            teacher_tokenizer=teacher_tokenizer,
            max_seq_len=self.data_config.max_seq_len
        )
        val_dataset = ProtXDataGen(
            data_path=self.data_config.val_file,
            teacher_tokenizer=teacher_tokenizer,
            max_seq_len=self.data_config.max_seq_len
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

if __name__ == "__main__":
    pipeline = DistillPipeline()
    pipeline.train()