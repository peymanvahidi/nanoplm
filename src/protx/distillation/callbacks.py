import torch
import os
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from ..utils import create_dirs
from ..models.student import ProtX

class OnnxExportCallback(TrainerCallback):
    def __init__(
        self,
        onnx_export_path: str,
        batch_size: int,
        seq_len: int,
        device: str = 'cpu'
    ):
        self.onnx_export_path = onnx_export_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        
        # Track best models
        self.best_train_loss = float('inf')
        self.best_train_loss_step = 0
        self.best_train_loss_checkpoint = None
        
        self.best_eval_loss = float('inf')
        self.best_eval_loss_step = 0
        self.best_eval_loss_checkpoint = None
        
        self.final_checkpoint = None
        self.final_step = 0
        self.final_train_loss = None
        self.final_eval_loss = None
        
        create_dirs(self.onnx_export_path)

    def _save_model_as_onnx(
        self,
        model: torch.nn.Module,
        filepath: str,
        batch_size: int,
        seq_len: int,
        device: str
    ):
        if filepath.endswith('.onnx'):
            create_dirs(filepath)
        else:
            create_dirs(filepath)
            filepath = os.path.join(filepath, 'model.onnx')

        dummy_input_ids = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)
        dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)
        
        # Ensure model is on the correct device for export
        model_device = next(model.parameters()).device
        model.to(device)
        model.eval()
        
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            filepath,
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=14
        )
        model.to(model_device)
        print(f"Model saved to {filepath}")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs
    ):
        """Track training metrics to identify best models"""
        if logs is None:
            return
            
        current_step = state.global_step
        
        if 'train_loss' in logs:
            train_loss = logs['train_loss']
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                self.best_train_loss_step = current_step
                self.best_train_loss_checkpoint = os.path.join(args.output_dir, f"checkpoint-{current_step}")
        
        if 'eval_loss' in logs:
            eval_loss = logs['eval_loss']
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                self.best_eval_loss_step = current_step
                self.best_eval_loss_checkpoint = os.path.join(args.output_dir, f"checkpoint-{current_step}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """Save the 3 best models as ONNX at the end of training"""
        if not state.is_world_process_zero:  # Only save on the main process
            return
            
        # Update final model info
        self.final_step = state.global_step
        self.final_checkpoint = args.output_dir
        
        # Get final losses from the last log
        if state.log_history:
            last_log = state.log_history[-1]
            self.final_train_loss = last_log.get('train_loss', 0.0)
            self.final_eval_loss = last_log.get('eval_loss', 0.0)
        
        print(f"\n=== Exporting Best Models to ONNX ===")
        print(f"Best training loss: {self.best_train_loss:.4f} at step {self.best_train_loss_step}")
        print(f"Best validation loss: {self.best_eval_loss:.4f} at step {self.best_eval_loss_step}")
        print(f"Final model at step {self.final_step}")
        
        # Export best training loss model
        if self.best_train_loss_checkpoint and os.path.exists(self.best_train_loss_checkpoint):
            best_train_filename = f"best_train_loss_tl{self.best_train_loss:.4f}_step{self.best_train_loss_step}.onnx"
            best_train_path = os.path.join(self.onnx_export_path, best_train_filename)
            self._load_and_export_checkpoint(self.best_train_loss_checkpoint, best_train_path)
        
        # Export best validation loss model
        if self.best_eval_loss_checkpoint and os.path.exists(self.best_eval_loss_checkpoint):
            best_eval_filename = f"best_val_loss_vl{self.best_eval_loss:.4f}_step{self.best_eval_loss_step}.onnx"
            best_eval_path = os.path.join(self.onnx_export_path, best_eval_filename)
            self._load_and_export_checkpoint(self.best_eval_loss_checkpoint, best_eval_path)
        
        # Export final model
        final_filename = f"final_model_tl{self.final_train_loss:.4f}_vl{self.final_eval_loss:.4f}_step{self.final_step}.onnx"
        final_path = os.path.join(self.onnx_export_path, final_filename)
        self._save_model_as_onnx(
            model.module if hasattr(model, 'module') else model,
            final_path,
            self.batch_size,
            self.seq_len,
            self.device
        )
        
        print(f"=== ONNX Export Complete ===\n")

    def _load_and_export_checkpoint(self, checkpoint_path: str, output_path: str):
        """Load a checkpoint and export it to ONNX"""
        try:
            model = ProtX.from_pretrained(checkpoint_path)
            
            self._save_model_as_onnx(
                model,
                output_path,
                self.batch_size,
                self.seq_len,
                self.device
            )
        except Exception as e:
            print(f"Warning: Could not export checkpoint {checkpoint_path}: {e}")
            print("This checkpoint might not be compatible with the current model structure.")
