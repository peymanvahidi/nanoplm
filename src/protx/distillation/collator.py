import torch
from typing import List, Dict, Any
from transformers import PreTrainedModel

class DistillDataCollator:
    def __init__(
        self,
        teacher_model: PreTrainedModel = None,
        on_the_fly: bool = False
    ):
        self.teacher_model = teacher_model
        self.on_the_fly = on_the_fly

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([feature["input_ids"] for feature in features])
        attention_mask = torch.stack([feature["attention_mask"] for feature in features])
        
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.on_the_fly:
            if self.teacher_model is None:
                raise ValueError("teacher_model must be provided when on_the_fly is True.")
            
            # Ensure teacher_model is on the same device as input_ids if using CUDA
            # This might need adjustment based on how device placement is handled with Trainer
            if input_ids.is_cuda and not next(self.teacher_model.parameters()).is_cuda:
                 self.teacher_model.to(input_ids.device)
            
            with torch.no_grad():
                teacher_embeddings = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state

            batch["teacher_embeddings"] = teacher_embeddings

        else:
            if "teacher_embeddings" not in features[0]:
                raise ValueError("teacher_embeddings must be in features when on_the_fly is False.")
            
            teacher_embeddings = torch.stack([feature["teacher_embeddings"] for feature in features])
            batch["teacher_embeddings"] = teacher_embeddings
            
        return batch