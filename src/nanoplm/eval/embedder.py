"""Wraps PureProtModernBertMLM for biotrainer's supervised CustomEmbedder interface."""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional, Tuple

import torch

from nanoplm.eval.zero_shot import TOKEN_BUDGET, special_token_offsets
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer


class NanoPLMCustomEmbedder:
    """Factory for a ``CustomEmbedder`` wrapping the nanoPLM model.

    Instantiated via ``NanoPLMCustomEmbedder.build(...)`` so that biotrainer
    symbols are already loaded before the inner class body is evaluated.
    """

    @staticmethod
    def build(
        model: PureProtModernBertMLM,
        tokenizer: ProtModernBertTokenizer,
        device: torch.device,
        biotrainer: Dict[str, Any],
        batch_size: int = 60,
        loop_count: Optional[int] = None,
    ):
        """Return a ``CustomEmbedder`` producing frozen-encoder embeddings.

        Args:
            model: Pure-torch nanoPLM MLM model.
            tokenizer: Matching tokenizer.
            device: torch device the model lives on.
            biotrainer: dict of biotrainer symbols (see ``_load_biotrainer_symbols``
                in ``eval/pipeline.py``).
            batch_size: Upper bound on sequences per forward pass.
            loop_count: Refinement depth override; None runs the checkpoint at
                the depth it was trained with.
        """
        CustomEmbedder = biotrainer["CustomEmbedder"]

        n_leading, _ = special_token_offsets(tokenizer)
        max_rows = max(1, int(batch_size))

        def _batches(sequences: List[str]) -> Generator[List[str], None, None]:
            """Group sequences into padded batches bounded by rows and tokens.

            Supervised benchmarks embed sequences up to 2000 residues, so a fixed
            row count alone would build batches far past what fits in memory.
            """
            batch: List[str] = []
            longest = 0
            for sequence in sequences:
                padded_len = max(longest, len(sequence))
                if batch and (
                    len(batch) + 1 > max_rows
                    or (len(batch) + 1) * padded_len > TOKEN_BUDGET
                ):
                    yield batch
                    batch, padded_len = [], len(sequence)
                batch.append(sequence)
                longest = padded_len
            if batch:
                yield batch

        class _Impl(CustomEmbedder):
            def __init__(self):
                self._model = model
                self._tokenizer = tokenizer
                self._device = device

            def per_residue(
                self, sequences: List[str]
            ) -> Generator[Tuple[str, torch.Tensor], None, None]:
                for batch in _batches(list(sequences)):
                    hidden = self._encode(batch)
                    for row, sequence in enumerate(batch):
                        yield sequence, hidden[
                            row, n_leading : n_leading + len(sequence)
                        ]

            def per_sequence(
                self, sequences: List[str]
            ) -> Generator[Tuple[str, torch.Tensor], None, None]:
                for sequence, embedding in self.per_residue(sequences):
                    yield sequence, embedding.mean(dim=0)

            def compute_attention_map(self, sequence: str):
                raise NotImplementedError(
                    "nanoPLM does not expose attention maps, so the "
                    "pbc_supervised_contact framework is not wired up. Use "
                    "pbc_zeroshot_contact for contact prediction."
                )

            def _encode(self, batch: List[str]) -> torch.Tensor:
                """Hidden states for ``batch``, as ``[rows, seq, hidden]``.

                Calls the encoder directly because the MLM wrapper returns
                logits. ``ModernBertModel.forward`` applies ``final_norm`` before
                returning, and without ``_cu_seqlens`` it stays on the padded
                SDPA path rather than the flat varlen one.
                """
                encoded = self._tokenizer(
                    batch, padding="longest", return_tensors="pt"
                )
                input_ids = encoded["input_ids"].to(self._device)
                attention_mask = encoded["attention_mask"].to(self._device)
                with torch.no_grad(), torch.amp.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=self._device.type == "cuda",
                ):
                    hidden = self._model.model(
                        input_ids=input_ids, attention_mask=attention_mask,
                        loop_count=loop_count,
                    )
                return hidden.detach().float().cpu()

        return _Impl()
