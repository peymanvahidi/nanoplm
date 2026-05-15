"""Embedding closures consumed by biotrainer's autoeval pipeline."""

from __future__ import annotations

from typing import Generator, Iterable, Tuple

import torch
from tqdm import tqdm

from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer


@torch.no_grad()
def embed_per_residue(
    model: PureProtModernBertMLM,
    tokenizer: ProtModernBertTokenizer,
    sequences: Iterable[str],
    device: torch.device,
    batch_size: int = 16,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Yield ``(sequence, per_residue_embedding)`` for each input sequence.

    Per-residue embedding shape: ``(seq_len, hidden_size)``. Special tokens
    (EOS) are stripped; only amino-acid positions are returned.
    """
    seq_list = list(sequences)

    for i in tqdm(range(0, len(seq_list), batch_size), desc="Embedding per-residue"):
        batch_seqs = seq_list[i : i + batch_size]
        encoded = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = model.model(input_ids=input_ids, attention_mask=attention_mask)

        for j, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            emb = hidden[j, :seq_len, :].float().cpu()
            yield seq, emb


@torch.no_grad()
def embed_per_sequence(
    model: PureProtModernBertMLM,
    tokenizer: ProtModernBertTokenizer,
    sequences: Iterable[str],
    device: torch.device,
    batch_size: int = 16,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Yield ``(sequence, per_sequence_embedding)`` for each input sequence.

    Per-sequence embedding shape: ``(hidden_size,)`` — mean pool over residue
    token positions, excluding special tokens.
    """
    seq_list = list(sequences)

    for i in tqdm(range(0, len(seq_list), batch_size), desc="Embedding per-sequence"):
        batch_seqs = seq_list[i : i + batch_size]
        encoded = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = model.model(input_ids=input_ids, attention_mask=attention_mask)

        for j, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            emb = hidden[j, :seq_len, :].mean(dim=0).float().cpu()
            yield seq, emb
