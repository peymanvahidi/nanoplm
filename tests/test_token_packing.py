import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from nanoplm.pretraining.dataset import TokenPackingDataset
from nanoplm.pretraining.collator import DataCollatorWithFlattening, ProtDataCollatorForLM

class MockDataset(Dataset):
    def __init__(self, lengths):
        self.lengths = lengths
        
    def __len__(self):
        return len(self.lengths)
        
    def __getitem__(self, idx):
        length = self.lengths[idx]
        return {
            "input_ids": torch.full((length,), 1, dtype=torch.long),
            "attention_mask": torch.ones(length, dtype=torch.long),
            "labels": torch.full((length,), -100, dtype=torch.long)
        }

def test_token_packing_dataset():
    # 10 samples of length 10
    lengths = [10] * 10
    ds = MockDataset(lengths)
    
    # Pack with max 30 tokens -> should get batches of 3 samples (10 -> 3 batches of 3, 1 batch of 1)
    packed_ds = TokenPackingDataset(ds, max_tokens_per_batch=30, drop_last=False, split_samples=False)
    
    batches = list(packed_ds)
    # Check assertions
    assert len(batches) == 4, f"Expected 4 batches, got {len(batches)}"
    assert len(batches[0]) == 3, f"Expected batch size 3, got {len(batches[0])}"
    assert len(batches[3]) == 1, f"Expected batch size 1 (remainder), got {len(batches[3])}"
    print("test_token_packing_dataset passed")

def test_data_collator_flattening():
    # Mock tokenizer
    class DummyTokenizer:
        mask_token = "[MASK]"
        mask_token_id = 99
        pad_token_id = 0
        vocab_size = 100
        cls_token_id = 1
        sep_token_id = 2
        unk_token_id = 3
        eos_token_id = 2
        all_special_ids = [0, 1, 2, 3, 99]

        def get_vocab(self):
            return {f"tok_{i}": i for i in range(100)}
        
        def get_special_tokens_mask(self, val, already_has_special_tokens=True):
             return [0] * len(val)
        def __len__(self): return 100
        
        def pad(self, examples, padding=True, return_tensors='pt', pad_to_multiple_of=None):
            # rudimentary padding logic for test
            input_ids = [e["input_ids"] for e in examples]
            labels = [e["labels"] for e in examples]
            attention_mask = [e["attention_mask"] for e in examples]
            
            max_len = max(len(x) for x in input_ids)
            if pad_to_multiple_of:
                 max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
            
            padded_input_ids = []
            padded_labels = []
            padded_attention_mask = []
            
            for i in range(len(input_ids)):
                 l = len(input_ids[i])
                 diff = max_len - l
                 if diff > 0:
                    padded_input_ids.append(torch.cat([input_ids[i], torch.full((diff,), self.pad_token_id, dtype=torch.long)])) 
                    padded_labels.append(torch.cat([labels[i], torch.full((diff,), -100, dtype=torch.long)]))
                    padded_attention_mask.append(torch.cat([attention_mask[i], torch.zeros(diff, dtype=torch.long)]))
                 else:
                    padded_input_ids.append(input_ids[i])
                    padded_labels.append(labels[i])
                    padded_attention_mask.append(attention_mask[i])

            return {
                 "input_ids": torch.stack(padded_input_ids),
                 "labels": torch.stack(padded_labels),
                 "attention_mask": torch.stack(padded_attention_mask)
            }
            

    tokenizer = DummyTokenizer()
    
    # We create a dummy collator for wrapping? No, we use ProtDataCollatorForLM which uses tokenizer.pad
    inner_collator = ProtDataCollatorForLM(tokenizer, mlm_probability=0.0) # no masking
    
    collator = DataCollatorWithFlattening(collator=inner_collator, pad_to_multiple_of=8)
    
    batch = [
        {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.ones(3), "labels": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([4, 5]), "attention_mask": torch.ones(2), "labels": torch.tensor([4, 5])}
    ]
    
    out = collator(batch)
    
    assert "input_ids" in out
    assert "cu_seqlens" in out
    assert "max_seqlen" in out
    
    # Input total length: 3+2=5. Pad to 8.
    assert out["input_ids"].shape[0] == 8, f"Expected length 8 (pad multiple), got {out['input_ids'].shape[0]}"
    
    expected_cu = torch.tensor([0, 3, 5], dtype=torch.int32)
    # Note: cu_seqlens should match the valid tokens. 
    # The padding is appended at the end of flattened tensor in collator logic, but not added to cu_seqlens list of sequences?
    # Let's check collator logic.
    # If collator appends padding to input_ids, does it update cu_seqlens?
    # Yes:
    # if "cu_seqlens" in batch:
    #    current_cu = batch["cu_seqlens"]
    #    new_end = current_cu[-1] + remainder
    #    batch["cu_seqlens"] = torch.cat([current_cu, new_end.unsqueeze(0)])
    
    expected_cu_padded = torch.tensor([0, 3, 5, 8], dtype=torch.int32)
    
    assert torch.equal(out["cu_seqlens"].cpu(), expected_cu_padded), f"Expected cu_seqlens {expected_cu_padded}, got {out['cu_seqlens']}"
    
    assert out["max_seqlen"] == 3
    print("test_data_collator_flattening passed")

if __name__ == "__main__":
    test_token_packing_dataset()
    test_data_collator_flattening()
    print("Tests passed!")
