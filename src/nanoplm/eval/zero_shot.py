"""Wraps PureProtModernBertMLM for biotrainer's zero-shot BioEngineer interface."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm

from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
from nanoplm.utils.logger import logger

# Upper bound on tokens per batched forward pass. The effective batch is
# ``min(zero_shot_batch_size, TOKEN_BUDGET // window_len)``, so long windows
# shrink the batch instead of running out of memory.
TOKEN_BUDGET = 65_536


def special_token_offsets(tokenizer: ProtModernBertTokenizer) -> Tuple[int, int]:
    """Return ``(leading, trailing)`` special-token counts for ``tokenizer``.

    ``ProtModernBertTokenizer`` emits ``[BOS?] + residues + EOS + register tokens``,
    so the trailing count grows with ``num_register_tokens``. Slicing off a fixed
    single trailing token (as earlier revisions did) silently kept register-token
    logits in the scored positions.
    """
    leading = 1 if getattr(tokenizer, "use_bos_token", False) else 0
    trailing = 1 + int(getattr(tokenizer, "num_register_tokens", 0) or 0)
    return leading, trailing


class NanoPLMBioEngineer:
    """Factory for a ``BioEngineer`` wrapping the nanoPLM model.

    Instantiated via ``NanoPLMBioEngineer.build(...)`` so that biotrainer
    symbols are already loaded before the inner class bodies are evaluated.
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
        """Return a ``BioEngineer`` wrapping ``model`` for zero-shot scoring.

        Args:
            model: Pure-torch nanoPLM MLM model.
            tokenizer: Matching tokenizer.
            device: torch device the model lives on.
            biotrainer: dict of biotrainer symbols (see ``_load_biotrainer_symbols``
                in ``eval/pipeline.py``).
            batch_size: Sequences per forward pass. Masked-marginals batches this
                many masked positions together; the categorical Jacobian batches
                this many mutated sequences.
            loop_count: Refinement depth override; None runs the checkpoint at
                the depth it was trained with.
        """
        CustomBioEngineerModel = biotrainer["CustomBioEngineerModel"]
        CustomBioEngineerModelWrapper = biotrainer["CustomBioEngineerModelWrapper"]
        BioEngineer = biotrainer["BioEngineer"]
        ZeroShotMethod = biotrainer["ZeroShotMethod"]
        STANDARD_AAS = biotrainer["STANDARD_AAS"]
        get_optimal_window = biotrainer["get_optimal_window"]

        n_leading, n_trailing = special_token_offsets(tokenizer)

        class _Impl(CustomBioEngineerModel):
            def __init__(self):
                super().__init__()
                self._model = model
                self._tokenizer = tokenizer
                self._device = device

            def get_name(self) -> str:
                return "nanoPLM"

            def supported_methods(self):
                return [
                    ZeroShotMethod.MASKED_MARGINALS,
                    ZeroShotMethod.WT_MARGINALS,
                    ZeroShotMethod.PSEUDOPERPLEXITY,
                    # Required by pbc_zeroshot_contact; BioEngineer rejects the
                    # call outright when it is missing.
                    ZeroShotMethod.JACOBIAN_CONTACT,
                ]

            def tokenize(self, batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
                enc = self._tokenizer(batch, padding="longest", return_tensors="pt")
                return (
                    enc["input_ids"].to(self._device),
                    enc["attention_mask"].to(self._device),
                )

            def _forward(self, input_ids, attention_mask):
                with torch.no_grad(), torch.amp.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=self._device.type == "cuda",
                ):
                    outputs = self._model(
                        input_ids=input_ids, attention_mask=attention_mask,
                        loop_count=loop_count,
                    )
                return outputs["logits"]

            def run_model(self, input_ids, attention_mask=None):
                # Contract: single sequence in, [seq_len, vocab_size] out.
                return self._forward(input_ids, attention_mask)[0]

            def run_model_batched(self, input_ids, attention_mask=None):
                # Native batched forward. The inherited default loops run_model
                # once per row, which would undo the batching entirely.
                return self._forward(input_ids, attention_mask)

            def get_mask_token_id(self):
                return self._tokenizer.mask_token_id

            def preprocess_sequences(self, sequences: Iterable[str]) -> List[str]:
                return [re.sub(r"[UZOB]", "X", s) for s in sequences]

            def aa_to_idx(self) -> Dict[str, int]:
                vocab = self._tokenizer.get_vocab()
                return {aa: vocab[aa] for aa in STANDARD_AAS}

            def strip_special_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
                return tensor[n_leading : tensor.shape[0] - n_trailing]

        class _Wrapper(CustomBioEngineerModelWrapper):
            """Adds batched masked-marginals and a tokenizer-correct Jacobian."""

            def __init__(self, custom_bioengineer, device: torch.device, batch_size: int):
                super().__init__(custom_bioengineer=custom_bioengineer, device=device)
                self._batch_size = max(1, int(batch_size))

            def _chunk_size(self, row_len: int) -> int:
                return max(1, min(self._batch_size, TOKEN_BUDGET // max(1, row_len)))

            def _get_masked_log_probabilities(self, sequence: str) -> torch.Tensor:
                """Batched masked-marginals.

                biotrainer masks one position at a time and runs one forward per
                position. This computes the same quantity, but groups positions
                that share a scoring window into a single forward pass.

                Positions are bucketed by window *length* so every row in a batch
                has exactly the same length: no padding is introduced, and each
                row sees the identical token slice and attention mask the
                per-position path would have passed. Results are written back by
                absolute position, so the stacking order does not depend on the
                order batches happen to be built in.
                """
                tokenized, attention_mask = self._tokenize([sequence], preprocess=True)
                mask_token_id = self.get_mask_token_id()
                seq_len = tokenized.size(1)
                device = tokenized.device
                base_tokens = tokenized[0]

                windows = {
                    i: get_optimal_window(
                        masked_position=i, seq_len_with_special=seq_len
                    )
                    for i in range(seq_len)
                }
                by_window_len: Dict[int, List[int]] = defaultdict(list)
                for position, (start, end) in windows.items():
                    by_window_len[end - start].append(position)

                per_position: List[Optional[torch.Tensor]] = [None] * seq_len
                with tqdm(
                    total=seq_len,
                    desc="Computing masked probabilities",
                    unit="pos",
                    ncols=100,
                    leave=False,
                ) as progress:
                    for window_len, positions in by_window_len.items():
                        chunk = self._chunk_size(window_len)
                        for offset in range(0, len(positions), chunk):
                            batch_positions = positions[offset : offset + chunk]
                            rows = torch.empty(
                                (len(batch_positions), window_len),
                                dtype=base_tokens.dtype,
                                device=device,
                            )
                            masks = (
                                torch.empty(
                                    (len(batch_positions), window_len),
                                    dtype=attention_mask.dtype,
                                    device=device,
                                )
                                if attention_mask is not None
                                else None
                            )
                            local_positions = torch.empty(
                                len(batch_positions), dtype=torch.long
                            )
                            for row_idx, position in enumerate(batch_positions):
                                start, end = windows[position]
                                row = base_tokens[start:end].clone()
                                row[position - start] = mask_token_id
                                rows[row_idx] = row
                                if masks is not None:
                                    masks[row_idx] = attention_mask[0, start:end]
                                local_positions[row_idx] = position - start

                            logits = self._model_batched_forward_fn(
                                input_ids=rows, attention_mask=masks
                            )
                            selected = logits[
                                torch.arange(len(batch_positions), device=logits.device),
                                local_positions.to(logits.device),
                            ].cpu()
                            for row_idx, position in enumerate(batch_positions):
                                per_position[position] = selected[row_idx]
                            progress.update(len(batch_positions))

                logits = torch.stack(per_position, dim=0)  # [seq_len, vocab_size]
                # Full vocabulary for probabilities (ProteinGym approach).
                log_probs = torch.log_softmax(logits, dim=-1)
                return self._strip_special_tokens(log_probs)

            def _compute_categorical_jacobian(
                self, sequence: str, batch_size: int = 32
            ) -> torch.Tensor:
                """Categorical Jacobian, corrected for nanoPLM's tokenizer.

                biotrainer's implementation hardcodes a ``[BOS] + residues + [EOS]``
                layout: it derives ``L = input_ids.shape[1] - 2`` and slices logits
                ``[1:-1]``. nanoPLM's tokenizer has no BOS and may append register
                tokens, so that arithmetic is wrong here (it raises on the final
                reshape). This uses the tokenizer's real offsets instead.

                ``batch_size`` is biotrainer's hardcoded 32 — the zero-shot contact
                pipeline offers no way to configure it — so the configured
                ``zero_shot_batch_size`` wins.
                """
                aa_token_ids = torch.tensor(
                    list(self.aa_to_idx().values()), device=self._device
                )
                input_ids, attention_mask = self._tokenize([sequence], preprocess=True)

                length = len(sequence)
                expected_len = n_leading + length + n_trailing
                if input_ids.shape[1] != expected_len:
                    raise ValueError(
                        f"Tokenized length {input_ids.shape[1]} does not match the "
                        f"expected {expected_len} = {n_leading} leading + {length} "
                        f"residues + {n_trailing} trailing special tokens. The "
                        f"categorical Jacobian cannot be aligned to residues."
                    )

                n_rows = length * 20
                chunk = self._chunk_size(expected_len)
                logger.info(
                    f"Categorical Jacobian: L={length} -> {n_rows} mutated sequences "
                    f"in {-(-n_rows // chunk)} forward passes (batch {chunk})."
                )

                mutated_ids = torch.tile(input_ids, (n_rows, 1))
                mutated_mask = torch.tile(attention_mask, (n_rows, 1))
                for position in range(length):
                    # Order of aa_token_ids matters: it defines the AA axes.
                    mutated_ids[
                        position * 20 : (position + 1) * 20, n_leading + position
                    ] = aa_token_ids

                ref_logits = self._model_forward_fn(input_ids, attention_mask)
                ref_logits = ref_logits[n_leading : n_leading + length][
                    :, aa_token_ids
                ].cpu()

                mutated_logits = []
                for batch_ids, batch_mask in zip(
                    torch.split(mutated_ids, chunk), torch.split(mutated_mask, chunk)
                ):
                    logits = self._model_batched_forward_fn(batch_ids, batch_mask)
                    mutated_logits.append(
                        logits[:, n_leading : n_leading + length, :][
                            :, :, aa_token_ids
                        ].cpu()
                    )

                # [L*20, L, 20] -> [L, 20, L, 20] in order of aa_token_ids/STANDARD_AAS
                mutated_logits = torch.cat(mutated_logits, dim=0).reshape(
                    length, 20, length, 20
                )
                return mutated_logits - ref_logits

        return BioEngineer.from_custom_model(
            _Wrapper(custom_bioengineer=_Impl(), device=device, batch_size=batch_size),
            device=device,
        )
