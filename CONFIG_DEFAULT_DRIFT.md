# Config default drift: get-yaml template vs. dataclasses

**Status:** open, not fixed. Written 2026-07-19 from an audit of `pretrain`.
**Scope:** `src/nanoplm/cli/pretrain.py` (the `get_yaml()` template), `src/nanoplm/pretraining/config.py`, `src/nanoplm/pretraining/models/modern_bert/model.py`.

## The problem in one sentence

Commit `ee5b81b` ("docs: match cli and yaml defaults") rewrote the defaults in the CLI options and
in the `get-yaml` template, but did not touch the dataclasses those values eventually land in — so
**a key you delete from your YAML does not fall back to the value the template showed you.**

That makes the template's numbers a lie of omission. They are not defaults; they are one
particular pre-filled config. Every optional key in the template silently has two values: the one
you can see, and the one you get if you remove the line.

## Why this is worse than a cosmetic mismatch

Three of these change behaviour, not just numbers:

| Key | Template shows | Actual fallback | What deleting the line does |
| --- | --- | --- | --- |
| `model.wo_init` | `"kaiming"` | `"zero"` | `optimizer: muonh` **fails at startup**. Zero-init `Wo` gets a zero hyperball radius. |
| `model.attn_layer_pattern` | `'F'` (all-full-attention) | `None` → stride logic, `global_attn_every_n_layers=3` | You get a **different attention architecture** — sliding-window on 2 of every 3 layers. |
| `pretraining.max_grad_norm` | `1` (clipping on) | `float("inf")` (clipping off) | Gradient clipping **silently turns off**. |

The rest are quieter but still real: `micro_batch_seqs` 128 → 64 changes VRAM and throughput,
`global_batch_tokens` 1024000 → 256000 changes the effective batch by 4×.

## Full inventory

### Group 1 — introduced by `ee5b81b` (template moved, dataclass did not)

| Key | Template | Dataclass | Source |
| --- | --- | --- | --- |
| `model.hidden_size` | `1024` | *(required, no default)* | `model.py:41` |
| `model.intermediate_size` | `2048` | *(required, no default)* | `model.py:42` |
| `model.num_hidden_layers` | `16` | *(required, no default)* | `model.py:43` |
| `model.num_attention_heads` | `16` | *(required, no default)* | `model.py:44` |
| `model.norm_type` | `"rmsnorm"` | `"layernorm"` | `model.py:50` |
| `model.no_mlp_on_first_layer` | `false` | `True` | `model.py:54` |
| `model.global_rope_theta` | `40000.0` | `160_000.0` | `model.py:62` |
| `model.tie_word_embeddings` | `false` | `True` | `model.py:80` |
| `model.attn_layer_pattern` | `'F'` | `None` | `model.py` |
| `model.wo_init` | `"kaiming"` | `"zero"` | `model.py:86` |
| `pretraining.ckp_dir` | `"output/pretraining_checkpoints"` | `"output/pretraining"` | `config.py:56` |
| `pretraining.micro_batch_seqs` | `128` | `64` | `config.py:59` |
| `pretraining.adam_beta2` | `0.95` | `0.999` | `config.py:78` |
| `pretraining.max_grad_norm` | `1` | `float("inf")` | `config.py:83` |
| `pretraining.global_batch_tokens` | `1024000` | `256000` | `config.py:103` |
| `pretraining.eval_steps` | `1000` | `250` | `config.py:152` |
| `pretraining.save_steps` | `2500` | `5000` | `config.py:153` |
| `pretraining.debug_non_finite_params` | `false` | `True` | `config.py:155` |
| `pretraining.debug_layerwise_log_every` | `100` | `20` | `config.py:157` |
| `pretraining.profiler_start_step` | `18` | `10` | `config.py:189` |
| `pretraining.profiler_end_step` | `22` | `15` | `config.py:190` |

The four required `model` size keys are a distinct sub-case: the dataclass is right to require them.
The problem is only that the generated docs render the template value as if it were a default.

### Group 2 — long-standing, predates `ee5b81b`

Mostly inert, because each is gated behind a feature flag that defaults off.

| Key | Template | Dataclass | Gated by |
| --- | --- | --- | --- |
| `model.activation_checkpointing_mode` | `"attn"` | `"layer"` | not gated — **live drift** |
| `model.canon_layers_mode` | `"ac"` | `"abcd"` | `use_canon_layers` (off) |
| `model.mhc_triton_fused` | `true` | `False` | `use_mhc_lite` (off) |
| `model.moe_leading_dense_layers` | `1` | `0` | `use_moe` (off) |
| `resume.mode` | `"decay"` | `"continue"` | `resume.is_resume` (off) |

### Group 3 — not drift, but worth knowing

- `model.canon_layers_kernel_size`: template `5`, dataclass `None`. `__post_init__` resolves
  `None → 5` (`config.py:289-291`), so behaviour is identical.
- `pretraining.repo_rope_warmup_steps`: template `1000`, dataclass `None` → falls back to
  `warmup_steps`, itself `1000`. Identical.
- The mutually-exclusive budget keys (`stable_tokens`/`decay_tokens`/`train_tokens`/`max_steps`/
  `stable_steps`/`decay_steps`) are deliberately `None` in the dataclass. The template supplies one
  concrete example. Working as intended.

## Two keys the docs cannot see at all

`pure_torch` and `pure_te` — the backend selector — never reach the generated schema.
`tools/schema_gen/templates.py`'s `_TOP` regex treats any column-0 `key:` line as a **section
header** (that is how it finds `model:`/`pretraining:`/`resume:`). `pure_torch: true` at the bottom
of the template is read as opening a section named `pure_torch`, its value is discarded, and no
fields follow. `overlay.yaml`'s `x-extra-fields` cannot backfill it because it only writes into
existing sections.

Currently papered over by a hand-written sentence at `website/src/content/docs/reference/pretrain.mdx:49`
— which is itself slightly wrong: it says both booleans ship commented out, but `pure_torch: true`
is active; only `pure_te` is commented.

## Why nothing caught this

- `tests/docs_schema/test_parity.py` compares field **names** only, never default **values**.
- `tools/schema_gen/sources.py`'s `PARITY_SOURCES` covers `pretraining`, `resume`, `eval`, and
  `distillation` — but **not `model`**, which is where most of the drift lives.
- The CI "schema parity gate" in `.github/workflows/docs.yml` runs `git diff --exit-code` after
  regenerating. That catches a stale checked-in schema (it would have caught this one), but it says
  nothing about whether the template agrees with the dataclasses.

## Suggested fix order

1. **Decide which side is authoritative.** Most likely the template — `ee5b81b`'s values look
   deliberate (rmsnorm, kaiming, clipping on, GQA at 8 KV heads read like considered upgrades).
   If so, port them into `config.py` and `model.py` and the drift closes in one direction.
2. **Add default-value parity** to `tests/docs_schema/test_parity.py`, not just name parity.
3. **Add `model` to `PARITY_SOURCES`** so `ModernBertConfig` gets any coverage at all.
4. **Fix the `_TOP` regex** in `templates.py` to distinguish `key:` (section) from `key: value`
   (top-level scalar), so `pure_torch`/`pure_te` reach the schema.

## Already done (docs side, 2026-07-19)

- Regenerated `website/src/generated/*.schema.json` — it was stale since `0f6ef6a`, four commits back.
  Recovered `model.num_kv_heads` and `model.moe_kernel_backend`, which were invisible to readers.
- Rewrote the prose that contradicted the new values: `guides/optimizers.md` (default optimizer is
  `normuon` not `adamw`; `max_grad_norm` clipping on; `adam_beta2` 0.95), `troubleshooting.md` and
  `guides/optimizers.md` (the muonh/`wo_init` trap now fires on hand-written configs, not stock
  templates), `pipeline/pretrain.mdx`, `reference/pretrain.mdx`.
- Note that the docs now describe the **template's** values throughout. If step 1 above resolves the
  other way — dataclasses win — that prose needs another pass.
