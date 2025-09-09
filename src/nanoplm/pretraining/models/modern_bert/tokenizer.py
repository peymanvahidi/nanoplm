import re
from transformers import PreTrainedTokenizer


class ProtModernBertTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="</s>",
        mask_token="<mask>",
    ):
        # Default vocabulary mapping: amino acids + special tokens (pad/eos/unk/mask)
        self.vocab = {
            "A": 4, "L": 5, "G": 6, "V": 7, "S": 8, "R": 9, "E": 10, "D": 11,
            "T": 12, "I": 13, "P": 14, "K": 15, "F": 16, "Q": 17, "N": 18,
            "Y": 19, "M": 20, "H": 21, "W": 22, "C": 23, "X": 24, "B": 25,
            "O": 26, "U": 27, "Z": 28,
            pad_token: 0, eos_token: 1, unk_token: 2, mask_token: 3,
        }

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            eos_token=eos_token,
            mask_token=mask_token,
        )

        self.unk_token_id = self.vocab.get(unk_token)
        self.pad_token_id = self.vocab.get(pad_token)
        self.eos_token_id = self.vocab.get(eos_token)
        self.mask_token_id = self.vocab.get(mask_token)

        self.model_input_names = ["input_ids", "attention_mask"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def _tokenize(self, text):
        # Simple character-level tokenization over protein sequences
        return list(text)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        if index in self.added_tokens_decoder:
            return self.added_tokens_decoder[index].content
        return {v: k for k, v in self.vocab.items()}.get(index, self.unk_token)
    
    def preprocess(self, sequence: str) -> str:
        seq = (sequence or "").strip().upper()
        seq = re.sub(r"[UZOB]", "X", seq)
        return seq

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
