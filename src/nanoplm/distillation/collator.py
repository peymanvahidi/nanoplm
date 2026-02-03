import torch
from typing import List, Dict, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer


class DistillDataCollator:
    """
    Data collator for knowledge distillation.

    Handles two modes:
    1. on_the_fly=True: Receives raw sequences, tokenizes with both student and teacher
       tokenizers, computes teacher embeddings on the fly.
    2. on_the_fly=False: Receives pre-tokenized data with pre-computed teacher embeddings.
    """
    def __init__(
        self,
        teacher_model: Optional[PreTrainedModel] = None,
        teacher_tokenizer: Optional[PreTrainedTokenizer] = None,
        student_tokenizer: Optional[PreTrainedTokenizer] = None,
        on_the_fly: bool = False,
        max_seq_len: int = 1024,
    ):
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.on_the_fly = on_the_fly
        self.max_seq_len = max_seq_len

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if self.on_the_fly:
            return self._process_on_the_fly(features)
        else:
            return self._process_precomputed(features)

    def _process_on_the_fly(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process raw sequences: tokenize with both tokenizers, compute teacher embeddings."""
        if self.teacher_model is None:
            raise ValueError("teacher_model must be provided when on_the_fly is True.")
        if self.teacher_tokenizer is None:
            raise ValueError("teacher_tokenizer must be provided when on_the_fly is True.")
        if self.student_tokenizer is None:
            raise ValueError("student_tokenizer must be provided when on_the_fly is True.")

        # Extract raw sequences
        raw_sequences = [f["raw_sequence"] for f in features]

        # Student tokenization (for student model input)
        student_encoding = self.student_tokenizer.batch_encode_plus(
            raw_sequences,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        # Teacher tokenization (for computing teacher embeddings)
        # Preprocess for teacher: add spaces between amino acids
        preprocessed_sequences = [" ".join(list(seq)) for seq in raw_sequences]
        teacher_encoding = self.teacher_tokenizer.batch_encode_plus(
            preprocessed_sequences,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        # Compute teacher embeddings
        teacher_device = next(self.teacher_model.parameters()).device
        teacher_input_ids = teacher_encoding["input_ids"].to(teacher_device)
        teacher_attention_mask = teacher_encoding["attention_mask"].to(teacher_device)

        with torch.no_grad():
            # Enable fp16 inference for 30-50% speedup on CUDA
            if teacher_device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    teacher_embeddings = self.teacher_model(
                        input_ids=teacher_input_ids,
                        attention_mask=teacher_attention_mask
                    ).last_hidden_state
            else:
                # MPS and CPU: run without autocast (MPS fp16 support is limited)
                teacher_embeddings = self.teacher_model(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask
                ).last_hidden_state

            # Convert to fp32 for loss computation consistency
            teacher_embeddings = teacher_embeddings.cpu().float()

        return {
            "input_ids": student_encoding["input_ids"],
            "attention_mask": student_encoding["attention_mask"],
            "teacher_embeddings": teacher_embeddings,
        }

    def _process_precomputed(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process precomputed data: already has student-tokenized input_ids and teacher embeddings."""
        if "teacher_embeddings" not in features[0]:
            raise ValueError("teacher_embeddings must be in features when on_the_fly is False.")

        input_ids = torch.stack([feature["input_ids"] for feature in features])
        attention_mask = torch.stack([feature["attention_mask"] for feature in features])
        teacher_embeddings = torch.stack([feature["teacher_embeddings"] for feature in features])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "teacher_embeddings": teacher_embeddings,
        }