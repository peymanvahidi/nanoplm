import torch
from pathlib import Path
from typing import Dict, Union, Tuple
from transformers import T5ForConditionalGeneration, T5EncoderModel, T5Config


class TeacherModel:
    """
    Wrapper for loading and using a pre-trained T5 model as a teacher model
    for knowledge distillation. Supports using only the encoder or the full model.
    """
    
    def __init__(
        self, 
        model_name_or_path: str = "Rostlab/prot_t5_xl_uniref50",
        device: str = "auto",
        use_cache: bool = True
    ):
        """
        Initialize the teacher model.
        
        Args:
            model_name_or_path: Name of the model from HuggingFace or path to saved model
            device: Device to load the model on ("cpu", "cuda", "cuda:0", "mps", etc.)
            use_cache: Whether to use model caching for faster inference
        """
        self.model_name_or_path = model_name_or_path

        if device == "auto":
            # Prioritize MPS on Mac over CPU, but CUDA over MPS
            if torch.backends.mps.is_available():
                self.device = "mps"
                print(f"Using MPS (Apple Metal) for GPU acceleration")
            elif torch.cuda.is_available():
                self.device = "cuda"
                print(f"Using CUDA for GPU acceleration")
            else:
                self.device = "cpu"
                print(f"Using CPU for computation (no GPU acceleration available)")
        else:
            self.device = device
            print(f"Using specified device: {device}")
        
        self.use_cache = use_cache
        
        # Full model and encoder-only model are loaded on demand to save memory
        self._full_model = None
        self._encoder_model = None
        self.config = T5Config.from_pretrained(model_name_or_path)
    
    @property
    def full_model(self) -> T5ForConditionalGeneration:
        """Lazy-load the full T5 model when needed."""
        if self._full_model is None:
            self._full_model = T5ForConditionalGeneration.from_pretrained(
                self.model_name_or_path,
                use_cache=self.use_cache
            ).to(self.device)
            self._full_model.eval()
        return self._full_model
    
    @property
    def encoder_model(self) -> T5EncoderModel:
        """Lazy-load the T5 encoder-only model when needed."""
        if self._encoder_model is None:
            print(f"Loading encoder model to {self.device} device")
            self._encoder_model = T5EncoderModel.from_pretrained(
                self.model_name_or_path
            ).to(self.device)
            self._encoder_model.eval()
        return self._encoder_model
    
    def get_encoder_embeddings(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        """
        Generate embeddings using only the encoder part of the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            output_hidden_states: If True, return all hidden states
            
        Returns:
            If output_hidden_states=False: Last hidden state [batch_size, seq_len, hidden_size]
            If output_hidden_states=True: (last_hidden_state, all_hidden_states)
        """
        with torch.no_grad():
            # Ensure tensors are on the correct device
            if input_ids.device.type != self.device:
                input_ids = input_ids.to(self.device)
            
            if attention_mask.device.type != self.device:
                attention_mask = attention_mask.to(self.device)
                
            outputs = self.encoder_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states
            )
            
            if output_hidden_states:
                return outputs.last_hidden_state, outputs.hidden_states
            return outputs.last_hidden_state
    
    def get_layer_weights(self) -> Dict[str, torch.Tensor]:
        """
        Extract layer weights from the full model for initializing a student model.
        
        Returns:
            Dictionary of layer weights keyed by layer name.
        """
        with torch.no_grad():
            model = self.full_model
            # Get state dict and return it
            return model.state_dict()
    
    def get_layer_by_name(self, layer_name: str) -> torch.Tensor:
        """
        Get specific layer weights by name.
        
        Args:
            layer_name: Name of the layer in the model's state dict
        
        Returns:
            Tensor containing the weights for the specified layer
        """
        state_dict = self.get_layer_weights()
        if layer_name in state_dict:
            return state_dict[layer_name]
        raise ValueError(f"Layer {layer_name} not found in model.")
    
    def save_encoder_only(self, output_dir: Union[str, Path]) -> str:
        """
        Save the encoder-only model to disk.
        
        Args:
            output_dir: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        self.encoder_model.save_pretrained(output_path)
        return str(output_path)
    
    def save_full_model(self, output_dir: Union[str, Path]) -> str:
        """
        Save the full model to disk.
        
        Args:
            output_dir: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        self.full_model.save_pretrained(output_path)
        return str(output_path) 