from transformers import PreTrainedTokenizer

class ProtXTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab=None, unk_token="<unk>", pad_token="<pad>", eos_token="</s>"):
        
        # Define vocabulary mapping amino acids & special tokens
        self.vocab = {
            "A": 3, "L": 4, "G": 5, "V": 6, "S": 7, "R": 8, "E": 9, "D": 10,
            "T": 11, "I": 12, "P": 13, "K": 14, "F": 15, "Q": 16, "N": 17,
            "Y": 18, "M": 19, "H": 20, "W": 21, "C": 22, "X": 23, "B": 24,
            "O": 25, "U": 26, "Z": 27, pad_token: 0, eos_token: 1, unk_token: 2
        }

        # Initialize parent class properly
        super().__init__(
            unk_token=unk_token, 
            pad_token=pad_token, 
            eos_token=eos_token
        )

        # Set up token ID attributes
        self.unk_token_id = self.vocab.get(unk_token)
        self.pad_token_id = self.vocab.get(pad_token)
        self.eos_token_id = self.vocab.get(eos_token)
        
        self.model_input_names = ["input_ids", "attention_mask"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab 

    def _tokenize(self, text):
        return list(text)  
    
    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.unk_token_id)  
    
    def _convert_id_to_token(self, index):
        if index in self.added_tokens_decoder:
            return self.added_tokens_decoder[index].content
        return {v: k for k, v in self.vocab.items()}.get(index, self.unk_token)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequences by adding eos tokens.
        """
        if token_ids_1 is None:
            if token_ids_0 and token_ids_0[-1] == self.eos_token_id:
                return token_ids_0
            return token_ids_0 + [self.eos_token_id]
        
        # For sequence pairs, add EOS to each sequence and concatenate
        if token_ids_0 and token_ids_0[-1] != self.eos_token_id:
            token_ids_0 = token_ids_0 + [self.eos_token_id]
        if token_ids_1 and token_ids_1[-1] != self.eos_token_id:
            token_ids_1 = token_ids_1 + [self.eos_token_id]
        return token_ids_0 + token_ids_1

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True
            )
            
        if token_ids_1 is None:
            # Mark only the EOS token as special
            return [0] * len(token_ids_0) + [1]
        
        # For sequence pairs
        return [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Create the token type IDs corresponding to sequences in a sequence or a pair.
        """
        if token_ids_1 is None:
            # For single sequences, all tokens have type 0
            return [0] * (len(token_ids_0) + 1)  # +1 for the EOS token
        
        # For sequence pairs, first sequence is type 0, second is type 1
        return [0] * (len(token_ids_0) + 1) + [1] * (len(token_ids_1) + 1)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the tokenizer's vocabulary files. Required for checkpointing.
        """
        import os
        import json
        
        if filename_prefix is not None:
            vocab_file = os.path.join(save_directory, f"{filename_prefix}-vocab.json")
        else:
            vocab_file = os.path.join(save_directory, "vocab.json")
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)
