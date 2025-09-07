import torch
from transformers import Trainer
from typing import Dict, Union, Any, Optional, Tuple
import logging
logger = logging.getLogger(__name__)

class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            training_mode=True
        )
        student_repr = outputs.last_hidden_state
        
        if "teacher_embeddings" in inputs and inputs["teacher_embeddings"] is not None:
            teacher_embeddings = inputs["teacher_embeddings"]
            attention_mask = inputs["attention_mask"]
            loss = self._compute_distillation_loss(
                student_repr, 
                teacher_embeddings, 
                attention_mask
            )
        else:
            logger.warning("Teacher embeddings not found in inputs during compute_loss")
            loss = torch.tensor(0.0, device=student_repr.device, requires_grad=True)
        
        return (loss, outputs) if return_outputs else loss
    
    def _compute_distillation_loss(
        self, 
        student_repr: torch.Tensor, 
        teacher_embeddings: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        teacher_embeddings = teacher_embeddings.to(student_repr.device)
        mask = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        diff = ((student_repr - teacher_embeddings) ** 2) * mask
        loss = diff.sum() / mask.sum().clamp(min=1)
        return loss
    
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
