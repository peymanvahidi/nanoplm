import abc
import torch.nn as nn
from typing import Optional

from transformers import PreTrainedTokenizerBase
from nanoplm.utils.common import get_device


class BaseTeacher(abc.ABC):
    def __init__(self, device: Optional[str] = None) -> None:
        # Store as string to be compatible with torch .to(device)
        self.device: str = str(device) if device is not None else get_device()

    @property
    @abc.abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Tokenizer to use for teacher tokenization."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def encoder_model(self) -> nn.Module:
        """Encoder model to use for teacher embedding calculation."""
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess(self, sequence: str) -> str:
        """Preprocess raw input sequence before tokenization."""
        raise NotImplementedError
