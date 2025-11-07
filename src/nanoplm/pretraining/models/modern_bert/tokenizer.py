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
        bos_token="<s>",
        use_bos_token=False,
        **kwargs,
    ):
        # Default vocabulary mapping: amino acids + special tokens (pad/eos/unk/mask/bos)
        vocab = {
            pad_token: 0, eos_token: 1, unk_token: 2, mask_token: 3,

            "A": 4, "L": 5, "G": 6, "V": 7, "S": 8, "R": 9, "E": 10, "D": 11,
            "T": 12, "I": 13, "P": 14, "K": 15, "F": 16, "Q": 17, "N": 18,
            "Y": 19, "M": 20, "H": 21, "W": 22, "C": 23, "X": 24, "B": 25,
            "O": 26, "U": 27, "Z": 28,

            bos_token: 29, "<unused0>": 30, "<unused1>": 31,
        }

        self.use_bos_token = use_bos_token

        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=unk_token))

        tokenizer.normalizer = Replace(r"[UZOB]", "X")

        tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")

        if use_bos_token:
            # Format: <bos>seq<eos> for single, <bos>seq<eos><bos>seq<eos> for pair
            tokenizer.post_processor = TemplateProcessing(
                single=f"{bos_token} $A {eos_token}",
                pair=f"{bos_token} $A {eos_token} {bos_token}:1 $B:1 {eos_token}:1",
                special_tokens=[
                    (bos_token, vocab[bos_token]),
                    (eos_token, vocab[eos_token]),
                ],
            )
        else:
            # Format: seq<eos> for single, seq<eos>seq<eos> for pair
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
            bos_token=bos_token if use_bos_token else None,
            model_input_names=["input_ids", "attention_mask"],
            **kwargs,
        )

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        """Preprocess text before tokenization: strip and uppercase."""
        text = (text or "").strip().upper()
        return (text, kwargs)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs by adding special tokens.

        Without BOS: seq<eos> for single, seq<eos>seq<eos> for pair
        With BOS: <bos>seq<eos> for single, <bos>seq<eos><bos>seq<eos> for pair
        """
        if self.use_bos_token:
            if token_ids_1 is None:
                return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
            return (
                [self.bos_token_id]
                + token_ids_0
                + [self.eos_token_id]
                + [self.bos_token_id]
                + token_ids_1
                + [self.eos_token_id]
            )
        else:
            if token_ids_1 is None:
                return token_ids_0 + [self.eos_token_id]
            return token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def get_special_tokens_mask(
        self, token_ids_0, token_ids_1=None, already_has_special_tokens=False
    ):
        """
        Get mask identifying which tokens are special tokens.

        Returns a list of 0s and 1s, with 1 indicating a special token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        if self.use_bos_token:
            if token_ids_1 is None:
                return [1] + [0] * len(token_ids_0) + [1]
            return (
                [1] + [0] * len(token_ids_0) + [1] + [1] + [0] * len(token_ids_1) + [1]
            )
        else:
            if token_ids_1 is None:
                return [0] * len(token_ids_0) + [1]
            return [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Create token type IDs for sequence pairs.

        Token type IDs are 0 for the first sequence and 1 for the second sequence.
        """
        if self.use_bos_token:
            if token_ids_1 is None:
                return [0] * (len(token_ids_0) + 2)  # +2 for BOS and EOS
            return [0] * (len(token_ids_0) + 2) + [1] * (
                len(token_ids_1) + 2
            )  # +2 for BOS and EOS each
        else:
            if token_ids_1 is None:
                return [0] * (len(token_ids_0) + 1)  # +1 for EOS
            return [0] * (len(token_ids_0) + 1) + [1] * (
                len(token_ids_1) + 1
            )  # +1 for EOS each

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
