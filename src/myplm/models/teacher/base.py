import abc
import torch.nn as nn

from transformers import PreTrainedTokenizerBase


class BaseTeacher(abc.ABC):

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
