from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, normalizers, Regex
from tokenizers.pre_tokenizers import Split
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing


class ProtModernBertTokenizer(PreTrainedTokenizerFast):
    # 20 standard amino acid token IDs (A=4 .. C=23)
    STANDARD_AA_TOKEN_IDS: frozenset = frozenset(range(4, 24))
    # Non-standard / ambiguous amino acid token IDs (X=24 .. Z=28).
    # Note: the tokenizer normalizer converts B, O, U, Z -> X before
    # tokenization, so in practice only X (24) appears in data. IDs 25-28
    # are included defensively for correctness if input bypasses the
    # normalizer.
    NON_STANDARD_AA_TOKEN_IDS: frozenset = frozenset(range(24, 29))

    REGISTER_TOKENS = ["<unused0>", "<unused1>"]
    REGISTER_TOKEN_IDS = [30, 31]

    def __init__(
        self,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="</s>",
        mask_token="<mask>",
        bos_token="<s>",
        use_bos_token=False,
        num_register_tokens=0,
        **kwargs,
    ):
        if num_register_tokens < 0 or num_register_tokens > 2:
            raise ValueError(
                f"num_register_tokens must be 0, 1, or 2, got {num_register_tokens}"
            )

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
        self.num_register_tokens = num_register_tokens

        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=unk_token))

        upper_case = [
            normalizers.Replace(chr(ord("a") + i), chr(ord("A") + i)) for i in range(26)
        ]
        unknown_aa = [normalizers.Replace(Regex(r"[UZOB]"), "X")]

        tokenizer.normalizer = normalizers.Sequence(upper_case + unknown_aa)

        tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")

        # Build register token suffix for TemplateProcessing
        reg_suffix = ""
        reg_special = []
        for i in range(num_register_tokens):
            reg_token = self.REGISTER_TOKENS[i]
            reg_suffix += f" {reg_token}"
            reg_special.append((reg_token, vocab[reg_token]))

        if use_bos_token:
            tokenizer.post_processor = TemplateProcessing(
                single=f"{bos_token} $A {eos_token}{reg_suffix}",
                pair=f"{bos_token} $A {eos_token}{reg_suffix} {bos_token}:1 $B:1 {eos_token}:1{reg_suffix}",
                special_tokens=[
                    (bos_token, vocab[bos_token]),
                    (eos_token, vocab[eos_token]),
                    *reg_special,
                ],
            )
        else:
            tokenizer.post_processor = TemplateProcessing(
                single=f"$A {eos_token}{reg_suffix}",
                pair=f"$A {eos_token}{reg_suffix} $B:1 {eos_token}:1{reg_suffix}",
                special_tokens=[
                    (eos_token, vocab[eos_token]),
                    *reg_special,
                ],
            )

        # Register tokens marked as additional_special_tokens so they are
        # excluded from MLM masking via get_special_tokens_mask.
        additional_special = self.REGISTER_TOKENS[:num_register_tokens] if num_register_tokens > 0 else None

        super().__init__(
            tokenizer_object=tokenizer,
            unk_token=unk_token,
            pad_token=pad_token,
            eos_token=eos_token,
            mask_token=mask_token,
            bos_token=bos_token if use_bos_token else None,
            additional_special_tokens=additional_special,
            **kwargs,
        )
