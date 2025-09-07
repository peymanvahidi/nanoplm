import re
import torch
from typing import Dict, Optional
from transformers import (
    T5ForConditionalGeneration, 
    T5EncoderModel, 
    T5Tokenizer
)

from nanoplm.models.teacher.base import BaseTeacher

class ProtT5(BaseTeacher):
    def __init__(
        self, 
        model_name: str = "Rostlab/prot_t5_xl_uniref50",
        device: Optional[str] = None
    ):
        super().__init__(device=device)
        self.model_name = model_name

    @property
    def full_model(self) -> T5ForConditionalGeneration:
        full_model = T5ForConditionalGeneration.from_pretrained(
            self.model_name,
        ).to(self.device).eval()
        return full_model
    
    @property
    def encoder_model(self) -> T5EncoderModel:
        encoder_model = T5EncoderModel.from_pretrained(
            self.model_name
        ).to(self.device).eval()
        return encoder_model
    
    @property
    def tokenizer(self) -> T5Tokenizer:
        return T5Tokenizer.from_pretrained(self.model_name, legacy=False)
    
    def get_layer_weights(self) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            model = self.full_model
            return model.state_dict()
    
    def get_layer_by_name(self, layer_name: str) -> torch.Tensor:
        state_dict = self.get_layer_weights()
        if layer_name in state_dict:
            return state_dict[layer_name]
        raise ValueError(f"Layer {layer_name} not found in model.")
    
    def preprocess(self, sequence: str) -> str:
        seq = (sequence or "").strip().upper()
        seq = re.sub(r"[UZOB]", "X", seq)
        return " ".join(list(seq))
