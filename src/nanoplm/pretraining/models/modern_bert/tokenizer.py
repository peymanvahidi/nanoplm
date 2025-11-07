from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.normalizers import Replace
from tokenizers.pre_tokenizers import Split
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing


class ProtModernBertTokenizer(PreTrainedTokenizerFast):
    def __init__(
        self,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="</s>",
        mask_token="<mask>",
        **kwargs
    ):
        # Default vocabulary mapping: amino acids + special tokens (pad/eos/unk/mask)
        vocab = {
            "A": 4, "L": 5, "G": 6, "V": 7, "S": 8, "R": 9, "E": 10, "D": 11,
            "T": 12, "I": 13, "P": 14, "K": 15, "F": 16, "Q": 17, "N": 18,
            "Y": 19, "M": 20, "H": 21, "W": 22, "C": 23, "X": 24, "B": 25,
            "O": 26, "U": 27, "Z": 28,
            pad_token: 0, eos_token: 1, unk_token: 2, mask_token: 3,
        }

        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=unk_token))

        tokenizer.normalizer = Replace(r"[UZOB]", "X")

        tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")

        tokenizer.post_processor = TemplateProcessing(
            single=f"$A {eos_token}",
            pair=f"$A {eos_token} $B:1 {eos_token}:1",
            special_tokens=[
                (eos_token, vocab[eos_token]),
            ],
        )

        super().__init__(
            tokenizer_object=tokenizer,
            unk_token=unk_token,
            pad_token=pad_token,
            eos_token=eos_token,
            mask_token=mask_token,
            model_input_names=["input_ids", "attention_mask"],
            **kwargs
        )

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        """Preprocess text before tokenization: strip and uppercase."""
        text = (text or "").strip().upper()
        return (text, kwargs)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # Append EOS to each sequence; no BOS token used
        if token_ids_1 is None:
            if token_ids_0 and token_ids_0[-1] == self.eos_token_id:
                return token_ids_0
            return token_ids_0 + [self.eos_token_id]

        if token_ids_0 and token_ids_0[-1] != self.eos_token_id:
            token_ids_0 = token_ids_0 + [self.eos_token_id]
        if token_ids_1 and token_ids_1[-1] != self.eos_token_id:
            token_ids_1 = token_ids_1 + [self.eos_token_id]
        return token_ids_0 + token_ids_1

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        if token_ids_1 is None:
            # Mark EOS at the end as special
            return [0] * len(token_ids_0) + [1]

        return [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [0] * (len(token_ids_0) + 1)  # +1 for EOS
        return [0] * (len(token_ids_0) + 1) + [1] * (len(token_ids_1) + 1)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        import os
        import json

        if filename_prefix is not None:
            vocab_file = os.path.join(save_directory, f"{filename_prefix}-vocab.json")
        else:
            vocab_file = os.path.join(save_directory, "vocab.json")

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        return (vocab_file,)
