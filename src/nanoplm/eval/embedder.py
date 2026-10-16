"""Wraps PureProtModernBertMLM for biotrainer's supervised CustomEmbedder interface."""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional, Tuple

import torch

from nanoplm.eval.zero_shot import TOKEN_BUDGET, special_token_offsets
from nanoplm.pretraining.models.modern_bert.modeling import collect_attention_probs
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

        n_leading, n_trailing = special_token_offsets(tokenizer)
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

            def compute_attention_map(self, sequence: str) -> torch.Tensor:
                """Attention features for one sequence, as ``[L, L, maps]``.

                ``maps`` is layers x heads (the layer loop counts once per
                application, so a Loopie checkpoint contributes one block per
                pass). The recipe follows biotrainer's Huggingface embedder so
                the probe sees the same kind of feature it does: post-softmax
                attention, symmetrized, then average-product corrected. Its own
                ``_apc``/``_symmetrize`` are reused rather than reimplemented,
                which keeps the two paths from drifting apart.

                Special tokens are dropped by the tokenizer's actual offsets,
                not by ``[1:-1]``: this tokenizer emits register tokens after
                the EOS, and a fixed single-token trim would leave them in the
                map as spurious residues.

                Cost: the returned map is L x L x maps, 512 maps for the 32x16
                base model, so ~0.5 GB fp16 at L=500 and ~4 GB at L=2000. The
                per-layer loop below keeps the *peak* at one layer's worth
                rather than the whole stack's.
                """
                from biotrainer.embedding.huggingface.huggingface_transformer_embedder import (  # noqa: E501
                    HuggingfaceTransformerEmbedder as _Reference,
                )

                encoded = self._tokenizer([sequence], return_tensors="pt")
                input_ids = encoded["input_ids"].to(self._device)
                attention_mask = encoded["attention_mask"].to(self._device)

                # `self._model.model` is the uncompiled inner encoder (attribute
                # access falls through OptimizedModule), which is what the
                # capture needs: it hangs off the eager SDPA path, and the
                # varlen flash path never materializes a softmax to capture.
                with torch.no_grad(), torch.amp.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=self._device.type == "cuda",
                ), collect_attention_probs(self._model.model) as per_layer:
                    self._model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        loop_count=loop_count,
                    )

                if not per_layer:
                    raise RuntimeError(
                        "no attention was captured; the forward pass did not "
                        "take the padded SDPA path"
                    )

                total = input_ids.shape[-1]
                end = total - n_trailing
                maps = []
                for probs in per_layer:
                    # (1, heads, T, T) -> (heads, L, L), then correct per map.
                    layer = probs[0, :, n_leading:end, n_leading:end].float()
                    layer = _Reference._apc(_Reference._symmetrize(layer))
                    maps.append(layer.to(torch.float16))
                attention_map = torch.cat(maps, dim=0)
                return attention_map.permute(1, 2, 0)  # (residues, residues, map)

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
