import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from nanoplm.data.datasets.pretrain import TokenPackingDataset
from nanoplm.pretraining.collator import DataCollatorWithFlattening, ProtDataCollatorForLM

class MockDataset(Dataset):
    def __init__(self, lengths):
        self.lengths = lengths
        self.total_tokens = sum(lengths)
        
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
    assert len(packed_ds) == 4, f"Expected __len__ to report 4 batches, got {len(packed_ds)}"
    assert len(batches[0]) == 3, f"Expected batch size 3, got {len(batches[0])}"
    assert len(batches[3]) == 1, f"Expected batch size 1 (remainder), got {len(batches[3])}"
    print("test_token_packing_dataset passed")


def test_token_packing_len_matches_iterator_without_split():
    lengths = [9, 9, 9, 9, 9]
    ds = MockDataset(lengths)
    packed_ds = TokenPackingDataset(ds, max_tokens_per_batch=20, drop_last=False, split_samples=False)

    batches = list(packed_ds)

    assert len(packed_ds) == len(batches) == 3


def test_token_packing_len_matches_iterator_with_split():
    lengths = [7, 7, 7]
    ds = MockDataset(lengths)
    packed_ds = TokenPackingDataset(ds, max_tokens_per_batch=10, drop_last=False, split_samples=True)

    batches = list(packed_ds)

    assert len(packed_ds) == len(batches) == 3


def test_token_packing_fast_forward_applies_only_to_target_epoch():
    lengths = [10] * 10
    ds = MockDataset(lengths)
    packed_ds = TokenPackingDataset(
        ds,
        max_tokens_per_batch=30,
        drop_last=False,
        split_samples=False,
        sampler=SequentialSampler(ds),
    )

    consumed = packed_ds.fast_forward(1, epoch=5)

    packed_ds.set_epoch(5)
    worker_epoch_5 = pickle.loads(pickle.dumps(packed_ds))
    resumed_batches = list(worker_epoch_5)

    packed_ds.set_epoch(6)
    worker_epoch_6 = pickle.loads(pickle.dumps(packed_ds))
    next_epoch_batches = list(worker_epoch_6)

    assert consumed == 3
    assert len(resumed_batches) == 3
    assert len(next_epoch_batches) == 4

class EpochLengthDataset(MockDataset):
    """Lengths depend on the epoch, like a cluster-sampled dataset's members."""

    def __init__(self, per_epoch):
        self.per_epoch = per_epoch
        self.epoch = 0
        super().__init__(per_epoch[0])

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.lengths = self.per_epoch[epoch]

    def get_all_sequence_lengths(self):
        return np.asarray(self.lengths, dtype=np.int32)


def test_token_packing_len_is_recomputed_when_the_epoch_changes():
    ds = EpochLengthDataset({0: [10] * 6, 1: [20] * 6})
    packed = TokenPackingDataset(ds, max_tokens_per_batch=30, drop_last=False,
                                 sampler=SequentialSampler(ds))
    assert len(packed) == len(list(packed)) == 2
    packed.set_epoch(1)
    assert len(packed) == len(list(packed)) == 6
    packed.set_epoch(1)  # same epoch: cached value is kept
    assert packed._cached_len == 6
    packed.set_epoch(0)
    assert len(packed) == 2


def test_fast_forward_replays_the_target_epochs_lengths():
    ds = EpochLengthDataset({0: [10] * 6, 1: [15] * 6})
    packed = TokenPackingDataset(ds, max_tokens_per_batch=30, drop_last=False,
                                 sampler=SequentialSampler(ds))
    # epoch 1 packs two 15-token samples per batch (epoch 0 would pack three)
    assert packed.fast_forward(2, epoch=1) == 4
    assert ds.epoch == 1
    assert len(list(packed)) == 1


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


class IdDataset(Dataset):
    def __len__(self):
        return 1000

    def __getitem__(self, i):
        return {"input_ids": torch.full((10,), i, dtype=torch.long)}

    def get_all_sequence_lengths(self):
        return np.full(1000, 10, dtype=np.int32)


def _first_ids(batch):
    return [int(s["input_ids"][0]) for s in batch]


def test_single_process_packing_visits_each_item_once_with_four_workers():
    from collections import Counter
    from nanoplm.data.datasets.pretrain import packing_sampler
    ds = IdDataset()
    packed = TokenPackingDataset(ds, max_tokens_per_batch=100, drop_last=True,
                                 sampler=packing_sampler(ds, seed=7, distributed=False))
    passes = []
    for epoch in (0, 1):
        packed.set_epoch(epoch)
        dl = DataLoader(packed, batch_size=None, num_workers=4, collate_fn=_first_ids,
                        multiprocessing_context="fork")
        batches = list(dl)
        seen = Counter(i for b in batches for i in b)
        assert len(seen) == 1000 and max(seen.values()) == 1
        assert len(batches) == len(packed) == 100
        passes.append([i for b in batches for i in b])
    assert passes[0] != passes[1]                 # set_epoch reshuffles


def test_single_process_resume_replays_the_same_permutation():
    from nanoplm.data.datasets.pretrain import packing_sampler
    ds = IdDataset()

    def packed():
        return TokenPackingDataset(ds, max_tokens_per_batch=100, drop_last=True,
                                   sampler=packing_sampler(ds, seed=7, distributed=False))

    full = packed()
    full.set_epoch(3)
    ids = [i for b in full for i in _first_ids(b)]
    resumed = packed()
    assert resumed.fast_forward(40, epoch=3) == 400
    assert [i for b in resumed for i in _first_ids(b)] == ids[400:]
