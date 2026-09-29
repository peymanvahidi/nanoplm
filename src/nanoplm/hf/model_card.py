"""Render a Hugging Face model card from an export's checkpoint provenance.

`render_model_card` is pure: it takes the config that will ship with the
export plus the two provenance dicts the orchestrator (Task 14) already reads
(`pretrain_cfg`, the pretraining YAML; `manifest`, the data manifest) and
returns markdown. It never touches disk or imports nanoplm's training stack
except the tokenizer, used only to read off special-token ids.
"""

from __future__ import annotations

from typing import Any, Optional

from nanoplm.hf.configuration_nanoplm import NanoPLMConfig
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer

# The snippet the Quickstart section embeds verbatim, formatted with the repo
# id. A module-level constant so the notebook (Task 16) can reuse and test
# against the exact same code the card ships.
#
# R13-1: both from_pretrained calls carry trust_remote_code=True -- the
# exported config.json carries auto_map, so AutoTokenizer.from_pretrained
# without it hits transformers' resolve_trust_remote_code, which prompts on
# stdin and hangs/fails outside a TTY.
#
# R13-2: residues are sliced with the special-tokens mask, not `[:len(seq)]`.
# ProtModernBertTokenizer appends EOS then num_register_tokens register
# tokens AFTER the sequence, and can prepend BOS -- `[:len(seq)]` is only
# right for one layout. `special_tokens_mask` is popped before model(**batch):
# the model does not accept it as a keyword.
QUICKSTART = """from transformers import AutoModel, AutoTokenizer
import torch

REPO = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModel.from_pretrained(REPO, trust_remote_code=True).eval()

sequences = ["MKALCLLLLPVLG", "MVLSPADKTNVKAAWGKV"]
batch = tokenizer(
    sequences, return_tensors="pt", padding=True, return_special_tokens_mask=True
)
special = batch.pop("special_tokens_mask").bool()
residue_mask = batch["attention_mask"].bool() & ~special

with torch.no_grad():
    out = model(**batch)

# per-residue embeddings, special tokens (BOS/EOS/register/padding) excluded
for i, seq in enumerate(sequences):
    per_residue = out.last_hidden_state[i][residue_mask[i]]
    per_sequence = per_residue.mean(dim=0)
    print(seq, per_residue.shape, per_sequence.shape)
"""

_UNKNOWN = "unknown"


def _fmt(value: Any) -> str:
    """`f"{value:,}"` for a real count, the literal "unknown" when absent."""
    if value is None or value == "":
        return _UNKNOWN
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _table(rows: list[tuple[str, str]]) -> str:
    lines = ["| | |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return "\n".join(lines)


def _token_budget(pretrain_cfg: dict) -> str:
    """The training duration pretrain_config.yaml actually specifies, in
    whichever unit it names -- never converted between tokens and steps.

    PretrainingConfig (pretraining/config.py): WSD's budget is
    stable_tokens/decay_tokens, with stable_steps/decay_steps as a step
    override; every other schedule (warmup_stable | cosine | linear) uses a
    single train_tokens budget, with max_steps as its step override. "unknown"
    when the schedule's own budget field(s) are absent (a step override for a
    schedule that isn't set is not read as that schedule's budget).
    """
    if pretrain_cfg.get("lr_schedule") == "wsd":
        stable_tokens = pretrain_cfg.get("stable_tokens")
        decay_tokens = pretrain_cfg.get("decay_tokens")
        if stable_tokens is not None or decay_tokens is not None:
            parts = [
                f"{_fmt(v)} {name}"
                for name, v in (("stable", stable_tokens), ("decay", decay_tokens))
                if v is not None
            ]
            return " + ".join(parts) + " tokens"
        stable_steps = pretrain_cfg.get("stable_steps")
        decay_steps = pretrain_cfg.get("decay_steps")
        if stable_steps is not None or decay_steps is not None:
            parts = [
                f"{_fmt(v)} {name}"
                for name, v in (("stable", stable_steps), ("decay", decay_steps))
                if v is not None
            ]
            return " + ".join(parts) + " steps"
        return _UNKNOWN
    train_tokens = pretrain_cfg.get("train_tokens")
    if train_tokens is not None:
        return f"{_fmt(train_tokens)} tokens"
    max_steps = pretrain_cfg.get("max_steps")
    if max_steps is not None:
        return f"{_fmt(max_steps)} steps"
    return _UNKNOWN


def _tokenizer_section(config: NanoPLMConfig) -> str:
    tokenizer = ProtModernBertTokenizer(num_register_tokens=config.num_register_tokens)
    rows = [
        ("pad", tokenizer.pad_token_id),
        ("mask", tokenizer.mask_token_id),
        ("eos", tokenizer.eos_token_id),
    ]
    bos_or_cls = (
        tokenizer.cls_token_id
        if tokenizer.cls_token_id is not None
        else tokenizer.bos_token_id
    )
    if bos_or_cls is not None:
        rows.append(("bos/cls", bos_or_cls))
    ids_table = _table([(name, str(value)) for name, value in rows])
    return (
        "## Tokenizer\n\n"
        "Loads as transformers' generic fast tokenizer -- no custom tokenizer "
        "class ships with this export.\n\n"
        f"{ids_table}\n\n"
        f"Register tokens: {config.num_register_tokens}"
    )


def _notes_section(config: NanoPLMConfig) -> Optional[str]:
    notes: list[str] = []
    if config.use_huginn_looping:
        note = (
            "- **Recycling** (`use_huginn_looping=True`): `output_hidden_states=True` "
            "returns one entry per recurrent iteration -- the embeddings, then each "
            "prelude layer, then one entry per loop iteration, then each coda layer "
            "-- not one entry per physical layer firing."
        )
        if config.recycling_s0_noise_sigma > 0:
            note += (
                " Outputs vary between calls unless you seed torch "
                "(`torch.manual_seed(...)`): the recurrence's first iteration "
                "starts from Gaussian noise."
            )
        notes.append(note)
    if config.use_mhc_lite:
        notes.append(
            "- **mHC-lite** (`use_mhc_lite=True`): layer entries in "
            "`hidden_states` have shape `(batch, seq, "
            f"{config.mhc_n_streams}, hidden)`; the embeddings entry and "
            "`last_hidden_state` are `(batch, seq, hidden)`."
        )
    if config.use_moe:
        notes.append(
            "- **MoE** (`use_moe=True`): the export runs the portable reference "
            "kernels (no CUTLASS grouped GEMM); expect it to be slower than the "
            "native training path."
        )
    if not notes:
        return None
    return "## Notes\n\n" + "\n".join(notes)


def render_model_card(
    *,
    config: NanoPLMConfig,
    pretrain_cfg: dict,
    manifest: dict,
    param_count: int,
    repo_id: Optional[str],
) -> str:
    front_matter = (
        "---\n"
        "library_name: transformers\n"
        "pipeline_tag: fill-mask\n"
        "license: mit\n"
        "tags:\n"
        "  - protein-language-model\n"
        "  - biology\n"
        "  - protein\n"
        "  - nanoplm\n"
        "---"
    )

    quickstart = QUICKSTART.format(repo_id=repo_id if repo_id else _UNKNOWN)

    kv_heads = (
        config.num_kv_heads
        if config.num_kv_heads is not None
        else config.num_attention_heads
    )
    # Hyperparameters are small integers written plainly (1024, not 1,024);
    # `_fmt`'s comma grouping is reserved for genuinely large counts below
    # (parameters, tokens, sequences), where it aids readability.
    architecture = _table(
        [
            ("Parameters", _fmt(param_count)),
            ("Hidden size", str(config.hidden_size)),
            ("Layers", str(config.num_hidden_layers)),
            ("Attention heads", str(config.num_attention_heads)),
            ("KV heads", str(kv_heads)),
            ("Intermediate size", str(config.intermediate_size)),
            ("Vocabulary size", str(config.vocab_size)),
            ("Normalization", config.norm_type),
            ("MLP activation", config.mlp_activation),
        ]
    )

    # source_global_step's dataclass default is 0 -- indistinguishable from a
    # real fresh-init checkpoint, but this config was never derived from one
    # unless checkpoint_config.nanoplm_config_from_checkpoint set it, so 0 is
    # treated as absent (R13-5).
    steps = _fmt(config.source_global_step) if config.source_global_step else _UNKNOWN
    # "Token budget (config)" is pretrain_config.yaml's planned duration;
    # "Train tokens (dataset)" is the manifest's actual tokenized-corpus size
    # -- a different quantity the spec's Training table keeps distinct.
    training = _table(
        [
            ("Optimizer", pretrain_cfg.get("optimizer", _UNKNOWN)),
            ("LR schedule", pretrain_cfg.get("lr_schedule", _UNKNOWN)),
            ("Token budget (config)", _token_budget(pretrain_cfg)),
            (
                "Global batch (tokens)",
                _fmt(pretrain_cfg.get("global_batch_tokens")),
            ),
            (
                "Micro-batch (sequences)",
                _fmt(pretrain_cfg.get("micro_batch_seqs")),
            ),
            ("Max sequence length", _fmt(manifest.get("max_seq_len"))),
            ("Training sequences", _fmt(manifest.get("train_sequences"))),
            ("Train tokens (dataset)", _fmt(manifest.get("total_train_tokens"))),
            ("Training steps", steps),
            ("nanoPLM version", config.nanoplm_version or _UNKNOWN),
        ]
    )

    device_support = _table(
        [
            ("CPU", "Verified -- export load and forward tested in this project."),
            (
                "CUDA, Hopper (sm90 / H100)",
                "Verified -- export load and forward tested in this project.",
            ),
            (
                "CUDA, sm70+ (other architectures)",
                "Expected to work (portable pure-PyTorch path, no custom "
                "kernels) -- not verified.",
            ),
            (
                "MPS (Apple Silicon)",
                "Expected to work (portable pure-PyTorch path, no custom "
                "kernels) -- not verified.",
            ),
        ]
    )

    sections = [
        front_matter,
        "# nanoPLM\n\n"
        "A protein language model trained with [nanoPLM]"
        "(https://github.com/peymanvahidi/nanoplm), a ModernBERT-style "
        "masked-language-model encoder for protein sequences.",
        "## Quickstart\n\n"
        "Requires `torch` and `transformers>=5` (transformers 4.x cannot load "
        "this model); no `nanoplm` install is needed.\n\n"
        f"```python\n{quickstart}```",
        f"## Architecture\n\n{architecture}",
        _tokenizer_section(config),
        f"## Training\n\n{training}",
        f"## Device support\n\n{device_support}",
    ]
    notes = _notes_section(config)
    if notes is not None:
        sections.append(notes)
    sections.append(
        "## Citation\n\n"
        "```bibtex\n"
        "@software{nanoplm,\n"
        "  title = {nanoPLM},\n"
        "  url = {https://github.com/peymanvahidi/nanoplm}\n"
        "}\n"
        "```"
    )

    return "\n\n".join(sections) + "\n"
