from dataclasses import dataclass
from typing import Generator, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.t5.modeling_t5 import T5LayerNorm

from nanoplm.distillation.models.student.tokenizer import ProtXTokenizer
from nanoplm.pretraining.models.modern_bert.modeling import (
    ModernBertConfig,
    ModernBertForMaskedLM,
)
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig


@dataclass
class ProtXConfig(ProtModernBertMLMConfig):
    """Student model config.

    Inherits all architecture fields from ProtModernBertMLMConfig (Canon,
    mHC-lite, GQA, DiffAttn, activation checkpointing, etc.) and adds
    distillation-specific fields.
    """
    projection_layer: bool = True
    # Output dimension of the projection head — must match the teacher encoder's
    # hidden size (480 for ESM-2 35M, 768 for ProtT5-Base, 1024 for ProtT5-XL).
    teacher_hidden_size: int = 1024


class ProtX(ModernBertForMaskedLM):
    """Student model for knowledge distillation.

    Backbone: pure-torch ModernBERT encoder with Triton-fused kernels (SReLU
    MLP, Canon conv, mHC-lite). Inherits from ModernBertForMaskedLM so the
    full optimised forward path (Flash Attention varlen, SDPA fallback,
    torch.compile-friendly custom ops) is available.

    The MLM head (self.head + self.decoder) from the parent is removed.  A
    linear projection + T5LayerNorm is added to project student hidden states
    to the teacher's embedding dimension (config.teacher_hidden_size).

    Tokenizer note: uses ProtXTokenizer (vocab_size=28, amino acids at IDs
    3-27) to keep compatibility with existing H5 datasets. This is different
    from ProtModernBertTokenizer (vocab_size=32), so existing H5 files can be
    reused without regeneration.
    """

    def __init__(self, config: ProtXConfig):
        self.tokenizer = ProtXTokenizer()
        self._use_projection_layer = config.projection_layer

        mb_config = ModernBertConfig(
            vocab_size=self.tokenizer.vocab_size,  # 28 — ProtXTokenizer vocab
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            mlp_activation=config.mlp_activation,
            max_position_embeddings=8192,
            mlp_dropout=config.mlp_dropout,
            mlp_bias=config.mlp_bias,
            no_mlp_on_first_layer=config.no_mlp_on_first_layer,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            classifier_activation=config.classifier_activation,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=None,
            unk_token_id=self.tokenizer.unk_token_id,
            mask_token_id=None,  # ProtXTokenizer has no mask token
            use_resid_lambdas=config.use_resid_lambdas,
            use_x0_lambdas=config.use_x0_lambdas,
            use_qk_norm=config.use_qk_norm,
            use_canon_layers=config.use_canon_layers,
            canon_layers_mode=config.canon_layers_mode,
            canon_layers_kernel_size=config.canon_layers_kernel_size,
            resid_lambda_init=config.resid_lambda_init,
            x0_lambda_init=config.x0_lambda_init,
            use_repo=config.use_repo,
            repo_after_n_layers=config.repo_after_n_layers,
            use_prores=config.use_prores,
            prores_T=config.prores_T,
            activation_checkpointing=config.activation_checkpointing,
            activation_checkpointing_mode=str(
                getattr(config, "activation_checkpointing_mode", "layer")
            ).strip().lower(),
            use_mhc_lite=config.use_mhc_lite,
            mhc_n_streams=config.mhc_n_streams,
            mhc_triton_fused=config.mhc_triton_fused,
            mhc_lite_wrapping_level=str(config.mhc_lite_wrapping_level).strip().lower(),
            use_diff_attn_v2=config.use_diff_attn_v2,
            attn_layer_pattern=config.attn_layer_pattern,
        )

        super().__init__(mb_config)

        # Remove MLM head and decoder — not needed for distillation.
        # super().__init__ already ran init_weights() on them, so we just drop.
        del self.head
        del self.decoder

        # Store for serialisation
        self.model_config = config

        # Projection head: student hidden_size → teacher embedding dim.
        # Only added when projection_layer=True (i.e. student hidden_size ≠ teacher_hidden_size).
        if self._use_projection_layer:
            self.proj = nn.Linear(config.hidden_size, config.teacher_hidden_size, bias=False)
            self.proj_norm = T5LayerNorm(config.teacher_hidden_size)

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        params = (p for p in self.parameters() if not only_trainable or p.requires_grad)
        total = sum(p.numel() for p in params)
        if exclude_embeddings:
            emb_params = sum(p.numel() for p in self.model.embeddings.parameters())
            total -= emb_params
        return total

    def forward(self, input_ids, attention_mask, training_mode=False):
        # Call the pure-torch encoder (self.model) directly via the SDPA path.
        # self.model returns (batch, seq_len, hidden_size) after final_norm.
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if training_mode:
            if self._use_projection_layer:
                projected = self.proj_norm(self.proj(hidden))
            else:
                projected = hidden
            return BaseModelOutput(
                last_hidden_state=projected,
                hidden_states=None,
                attentions=None,
            )

        # Inference: raw encoder embeddings, no projection
        return BaseModelOutput(
            last_hidden_state=hidden,
            hidden_states=None,
            attentions=None,
        )

    @staticmethod
    def load_and_generate_embeddings(
        checkpoint_path: str,
        sequences: Union[List[str], Iterator[str]],
        batch_size: int = 32,
        max_length: int = 512,
        device: str = "cuda",
        per_seq_embeddings: bool = True,
        mlp_activation: str = "swiglu",
        projection_layer: bool = True,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Load model from checkpoint and generate embeddings for sequences.

        Args:
            checkpoint_path: Path to the model.safetensors file.
            sequences: Iterator or list of protein sequences.
            batch_size: Number of sequences to process at once.
            max_length: Maximum sequence length for tokenisation.
            device: Device to run inference on.
            per_seq_embeddings: If True, return mean-pooled sequence-level
                embeddings [embed_dim]. If False, return per-token embeddings
                [sequence_length, embed_dim].
            mlp_activation: MLP activation function ("swiglu" or others).
            projection_layer: Whether to include the projection head.

        Yields:
            (sequence, embedding_tensor) for each input sequence.
        """
        try:
            embed_dim, num_layers, num_heads = ProtX.inspect_checkpoint_architecture(
                checkpoint_path
            )
            print(
                f"Detected architecture: embed_dim={embed_dim}, "
                f"num_layers={num_layers}, num_heads={num_heads}"
            )
        except Exception as e:
            print(f"Error detecting architecture: {e}")
            return

        config = ProtXConfig(
            hidden_size=embed_dim,
            intermediate_size=embed_dim * 2,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            mlp_activation=mlp_activation,
            projection_layer=projection_layer,
        )
        model = ProtX(config)

        try:
            state_dict = load_file(checkpoint_path)

            model_vocab_size = model.tokenizer.vocab_size
            checkpoint_vocab_size = None
            for key, tensor in state_dict.items():
                if "embeddings.tok_embeddings.weight" in key:
                    checkpoint_vocab_size = tensor.shape[0]
                    break

            if checkpoint_vocab_size and checkpoint_vocab_size != model_vocab_size:
                print(
                    f"Vocabulary size mismatch: checkpoint has {checkpoint_vocab_size}, "
                    f"model expects {model_vocab_size}. Adjusting..."
                )
                for key in list(state_dict.keys()):
                    if "embeddings.tok_embeddings.weight" in key:
                        state_dict[key] = state_dict[key][:model_vocab_size]

            model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return

        model.to(device)
        model.eval()

        if not isinstance(sequences, list):
            sequences = list(sequences)

        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i : i + batch_size]

                tokenized = model.tokenizer.batch_encode_plus(
                    batch_sequences,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )

                input_ids = tokenized["input_ids"].to(device)
                attention_mask = tokenized["attention_mask"].to(device)

                outputs = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    training_mode=False,
                )

                embeddings = outputs.last_hidden_state  # (batch, seq_len, embed_dim)

                if per_seq_embeddings:
                    embeddings_no_eos = embeddings[:, :-1, :]
                    mask_no_eos = attention_mask[:, :-1]
                    mask_expanded = mask_no_eos.unsqueeze(-1).expand(
                        embeddings_no_eos.size()
                    ).float()
                    masked_embeddings = embeddings_no_eos * mask_expanded
                    summed = torch.sum(masked_embeddings, dim=1)
                    summed_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    mean_pooled = summed / summed_mask
                    for seq, embedding in zip(batch_sequences, mean_pooled):
                        yield seq, embedding.cpu()
                else:
                    for seq, seq_embeddings, seq_mask in zip(
                        batch_sequences, embeddings, attention_mask
                    ):
                        actual_length = seq_mask.sum().item() - 1
                        yield seq, seq_embeddings[:actual_length].cpu()

    @staticmethod
    def inspect_checkpoint_architecture(
        checkpoint_path: str,
        head_dim: Optional[int] = None,
    ) -> Tuple[int, int, int]:
        """Inspect model architecture from a safetensors checkpoint.

        Args:
            checkpoint_path: path to model.safetensors
            head_dim: required for GQA checkpoints (qkv_dim != 3*embed_dim);
                ignored otherwise. If omitted on a GQA checkpoint the function
                raises a clear ValueError naming the parameter.

        Returns:
            (embed_dim, num_layers, num_heads)
        """
        try:
            state_dict = load_file(checkpoint_path)
        except Exception as e:
            raise FileNotFoundError(
                f"Error loading checkpoint from {checkpoint_path}: {e}"
            )

        total_params = sum(tensor.numel() for tensor in state_dict.values())
        print(f"Total number of parameters: {total_params:,}")

        embed_dim = None
        num_layers = 0
        num_heads = None

        for key, tensor in state_dict.items():
            if "embeddings.tok_embeddings.weight" in key:
                _, embed_dim = tensor.shape
                break
            elif "layers.0.attn.Wo.weight" in key:
                embed_dim, _ = tensor.shape
                break
            elif "layers.0.mlp.Wi.weight" in key:
                _, embed_dim = tensor.shape
                break

        if embed_dim is None:
            raise ValueError("Could not determine embed_dim from checkpoint")

        layer_indices = set()
        for key in state_dict.keys():
            if "layers." in key:
                parts = key.split(".")
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        try:
                            layer_indices.add(int(parts[i + 1]))
                        except ValueError:
                            continue

        num_layers = len(layer_indices)
        if num_layers == 0:
            raise ValueError("Could not determine num_layers from checkpoint")

        for key, tensor in state_dict.items():
            if "layers.0.attn.Wqkv.weight" in key:
                qkv_dim, model_dim = tensor.shape
                if qkv_dim == 3 * embed_dim:
                    # MHA: try common head_dim values, then divisor fallback.
                    for hd in [64, 32, 128]:
                        if embed_dim % hd == 0:
                            num_heads = embed_dim // hd
                            break
                    if num_heads is None:
                        for possible_heads in [8, 12, 16, 20, 24, 32]:
                            if embed_dim % possible_heads == 0:
                                num_heads = possible_heads
                                break
                else:
                    # GQA: qkv_dim = (num_q_heads + 2*num_kv_heads) * head_dim.
                    # Need head_dim to solve.
                    if head_dim is None:
                        raise ValueError(
                            f"GQA checkpoint detected (qkv_dim={qkv_dim} != "
                            f"3 * embed_dim={3 * embed_dim}). Pass head_dim "
                            f"explicitly to inspect_checkpoint_architecture()."
                        )
                    if embed_dim % head_dim != 0:
                        raise ValueError(
                            f"head_dim={head_dim} does not divide embed_dim={embed_dim}"
                        )
                    num_heads = embed_dim // head_dim
                break

        if num_heads is None:
            raise ValueError("Could not determine num_heads from checkpoint")

        return embed_dim, num_layers, num_heads
