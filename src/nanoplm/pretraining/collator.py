import torch
from typing import Iterable, Optional, List
from transformers import DataCollatorForLanguageModeling


class ProtDataCollatorForLM(DataCollatorForLanguageModeling):
    """
    Protein-aware MLM collator:
      - custom (mask, random, keep) proportions
      - random replacements drawn only from non-special tokens
      - never masks at padding (attention_mask == 0)
    """

    def __init__(
        self,
        tokenizer,
        mlm_probability: float = 0.03,
        mask_token_probability: float = 0.80,
        random_token_probability: float = 0.10,
        keep_probability: float = 0.10,
        *,
        extra_excluded_token_ids: Optional[Iterable[int]] = None,
        **kwargs,
    ):
        # parent handles dynamic padding, tensor type, etc.
        super().__init__(
            tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability, **kwargs
        )

        # normalize split (robust to slight mis-specified sums)
        total = mask_token_probability + random_token_probability + keep_probability
        self.p_mask = mask_token_probability / total
        self.p_rand = random_token_probability / total
        self.p_keep = keep_probability / total

        if getattr(self.tokenizer, "mask_token_id", None) is None:
            raise ValueError("Tokenizer must define a mask_token_id for MLM.")

        # Build the pool for random replacements: all vocab ids minus specials (and optional extras)
        vocab_ids = list(self.tokenizer.get_vocab().values())
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        special_ids.add(self.tokenizer.mask_token_id)
        if extra_excluded_token_ids:
            special_ids.update(extra_excluded_token_ids)

        allowed = [tid for tid in vocab_ids if tid not in special_ids]
        if not allowed:
            raise ValueError(
                "No allowable token ids for random replacement after exclusions."
            )
        self.allowed_random_token_ids = torch.tensor(allowed, dtype=torch.long)

    def torch_mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ):
        """
        Mirrors HF logic but uses custom p_mask/p_rand/p_keep and a restricted random pool.
        """
        labels = inputs.clone()

        # base Bernoulli for whether a position is subject to any corruption
        probability_matrix = torch.full(
            labels.shape, self.mlm_probability, device=inputs.device
        )

        if special_tokens_mask is None:
            # compute via tokenizer
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val.tolist(), already_has_special_tokens=True
                )
                for val in labels
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool, device=inputs.device
            )

        # never corrupt special positions
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # sample which positions to corrupt at all
        masked_indices = torch.bernoulli(probability_matrix).to(torch.bool)
        labels[~masked_indices] = -100  # ignore in loss

        if masked_indices.any():
            # Within the selected positions, decide mask / random / keep
            dice = torch.rand(size=inputs.shape, device=inputs.device)

            mask_choice = (dice < self.p_mask) & masked_indices
            rand_choice = (
                (dice >= self.p_mask)
                & (dice < self.p_mask + self.p_rand)
                & masked_indices
            )
            # keep_choice is implicit

            # 1) replace with [MASK]
            inputs[mask_choice] = self.tokenizer.mask_token_id

            # 2) replace with random non-special token
            if rand_choice.any():
                pool = self.allowed_random_token_ids.to(inputs.device)
                n = rand_choice.sum().item()
                idxs = torch.randint(
                    low=0, high=pool.numel(), size=(n,), device=inputs.device
                )
                inputs[rand_choice] = pool[idxs]

            # 3) keep_choice -> unchanged
        return inputs, labels

    def __call__(self, examples: List[dict]):
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Ensure tensors are in long dtype (what HF models expect)
        batch["input_ids"] = batch["input_ids"].long()
        batch["attention_mask"] = batch["attention_mask"].long()

        input_ids = batch["input_ids"]

        # Build/augment special mask
        if "special_tokens_mask" in batch:
            special = batch["special_tokens_mask"].bool()
        else:
            special = [
                self.tokenizer.get_special_tokens_mask(
                    v.tolist(), already_has_special_tokens=True
                )
                for v in input_ids
            ]
            special = torch.tensor(special, dtype=torch.bool, device=input_ids.device)

        # Forbid masking where attention_mask == 0 (padding, packed slack, etc.)
        if "attention_mask" in batch:
            special |= ~batch["attention_mask"].bool()

        inputs, labels = self.torch_mask_tokens(input_ids, special_tokens_mask=special)
        batch["input_ids"] = inputs
        batch["labels"] = labels

        return batch
