import torch
import os
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from ..utils import create_dirs

class OnnxExportCallback(TrainerCallback):
    def __init__(self, onnx_export_path: str, batch_size: int, seq_len: int, device: str = 'cpu'):
        self.onnx_export_path = onnx_export_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        create_dirs(self.onnx_export_path) # Create directory if it doesn't exist

    def _save_model_as_onnx(
        self,
        model: torch.nn.Module,
        filepath: str,
        batch_size: int,
        seq_len: int,
        device: str
    ):
        # Always ensure directory exists
        if filepath.endswith('.onnx'):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        else:
            # Create the directory and append model.onnx to the path
            os.makedirs(filepath, exist_ok=True)
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
            output_names=['last_hidden_state'], # Changed from 'output' to match BaseModelOutput
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'} # Changed from 'output'
            },
            opset_version=14
        )
        # Restore model to original device
        model.to(model_device)
        print(f"Model saved to {filepath}")

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Event called after a checkpoint save."""
        if state.is_world_process_zero: # Only save on the main process
            # Construct unique path for this save based on global step
            save_dir = os.path.join(self.onnx_export_path, f"checkpoint-{state.global_step}")
            self._save_model_as_onnx(
                model.module if hasattr(model, 'module') else model, # Handle DDP
                save_dir,
                self.batch_size, # Or use args.per_device_train_batch_size
                self.seq_len,    # This needs to be available, e.g. from model config or data_config
                self.device
            )