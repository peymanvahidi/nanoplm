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

        upper_case = [
            normalizers.Replace(chr(ord("a") + i), chr(ord("A") + i)) for i in range(26)
        ]
        unknown_aa = [normalizers.Replace(Regex(r"[UZOB]"), "X")]

        tokenizer.normalizer = normalizers.Sequence(upper_case + unknown_aa)

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
            **kwargs,
        )
