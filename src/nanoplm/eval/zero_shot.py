"""Wraps PureProtModernBertMLM for biotrainer's zero-shot BioEngineer interface."""

from __future__ import annotations

import re
from typing import Any, List, Tuple

import torch

from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer


class NanoPLMBioEngineer:
    """Factory for a ``BioEngineer`` wrapping the nanoPLM model.

    Instantiated via ``NanoPLMBioEngineer.build(...)`` so that biotrainer
    symbols (CustomBioEngineerModel, ZeroShotMethod, STANDARD_AAS) are
    already loaded before the inner class body is evaluated.
    """

    @staticmethod
    def build(
        model: PureProtModernBertMLM,
        tokenizer: ProtModernBertTokenizer,
        device: torch.device,
        biotrainer: dict[str, Any],
    ):
        """Return a ``BioEngineer`` wrapping ``model`` for ProteinGym zero-shot scoring.

        Args:
            model: Pure-torch nanoPLM MLM model.
            tokenizer: Matching tokenizer.
            device: torch device the model lives on.
            biotrainer: dict of biotrainer symbols (see ``_load_biotrainer_symbols``
                in ``eval/pipeline.py``).
        """
        CustomBioEngineerModel = biotrainer["CustomBioEngineerModel"]
        BioEngineer = biotrainer["BioEngineer"]
        ZeroShotMethod = biotrainer["ZeroShotMethod"]
        STANDARD_AAS = biotrainer["STANDARD_AAS"]

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
                ]

            def tokenize(self, batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
                enc = self._tokenizer(batch, padding="longest", return_tensors="pt")
                return (
                    enc["input_ids"].to(self._device),
                    enc["attention_mask"].to(self._device),
                )

            def run_model(self, input_ids, attention_mask=None):
                with torch.no_grad(), torch.amp.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=self._device.type == "cuda",
                ):
                    outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                return outputs["logits"].squeeze(0)

            def get_mask_token_id(self):
                return self._tokenizer.mask_token_id

            def preprocess_sequences(self, sequences: List[str]) -> List[str]:
                return [re.sub(r"[UZOB]", "X", s) for s in sequences]

            def aa_to_idx(self):
                vocab = self._tokenizer.get_vocab()
                return {aa: vocab[aa] for aa in STANDARD_AAS}

            def strip_special_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
                # No leading CLS in this tokenizer; strip trailing EOS only.
                return tensor[0:-1]

        return BioEngineer.from_custom_model(_Impl(), device=device)
