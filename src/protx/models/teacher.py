import torch
from pathlib import Path
from typing import Dict, Union, Tuple, List
from transformers import T5ForConditionalGeneration, T5EncoderModel, T5Config

from ..utils.common import get_device

class TeacherModel:
    """
    To get the full model or the encoder model.
    """
    
    def __init__(
        self, 
        model_name: str = "Rostlab/prot_t5_xl_uniref50",
        use_cache: bool = True
    ):
        self.device = get_device()
        self.model_name = model_name
        self.use_cache = use_cache
    
    @property
    def full_model(self) -> T5ForConditionalGeneration:
        """Lazy-load the full T5 model when needed."""
        full_model = T5ForConditionalGeneration.from_pretrained(
            self.model_name,
            use_cache=self.use_cache
        ).to(self.device)
        full_model.eval()
        return full_model
    
    @property
    def encoder_model(self) -> T5EncoderModel:
        """Lazy-load the T5 encoder-only model when needed."""
        encoder_model = T5EncoderModel.from_pretrained(
            self.model_name
        ).to(self.device)
        encoder_model.eval()
        return encoder_model
    
    def get_encoder_embeddings(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        only_last_hidden_state: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        """
        Generate embeddings using only the encoder part of the model.

        If only_last_hidden_state is False, includes all hidden states,
        usually we only want the last hidden state.
        """
        with torch.no_grad():

            # Send input_ids and attention_mask to the correct device
            if input_ids.device.type != self.device:
                input_ids = input_ids.to(self.device)
            
            if attention_mask.device.type != self.device:
                attention_mask = attention_mask.to(self.device)
                
            # Get the embeddings
            outputs = self.encoder_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=only_last_hidden_state
            )
            
            # Return the embeddings
            if only_last_hidden_state:
                return outputs.last_hidden_state
            return outputs.last_hidden_state, outputs.hidden_states
    
    def tokenize(self, sequences: List[str]) -> torch.Tensor:
        """
        Tokenize a list of sequences.
        """
        return self.tokenizer.encode(sequences, return_tensors="pt")
    
    def get_layer_weights(self) -> Dict[str, torch.Tensor]:
        """
        Extract layer weights from the full model.
        Sometimes can be used for initializing a student model.
        """
        with torch.no_grad():
            model = self.full_model
            return model.state_dict()
    
    def get_layer_by_name(self, layer_name: str) -> torch.Tensor:
        """
        Get specific layer weights by name.
        """
        state_dict = self.get_layer_weights()
        if layer_name in state_dict:
            return state_dict[layer_name]
        raise ValueError(f"Layer {layer_name} not found in model.")
    
    # def save_encoder_only(self, output_dir: Union[str, Path]) -> str:
    #     """
    #     Save the encoder-only model to disk.
    #     """
    #     output_path = Path(output_dir)
    #     output_path.mkdir(exist_ok=True, parents=True)
        
    #     self.encoder_model.save_pretrained(output_path)
    #     return str(output_path)
    
    # def save_full_model(self, output_dir: Union[str, Path]) -> str:
    #     """
    #     Save the full model to disk.
    #     """
    #     output_path = Path(output_dir)
    #     output_path.mkdir(exist_ok=True, parents=True)
        
    #     self.full_model.save_pretrained(output_path)
    #     return str(output_path) 