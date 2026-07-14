---
title: Choose a backend
description: nanoPLM pretrains on pure-torch or pure-TE. Use pure-torch unless you specifically need Transformer Engine kernels — and know what pure-TE cannot do.
---

Use `--pure-torch`. It is the backend every feature in nanoPLM supports, and it is the one the project's own launcher uses.

```bash
nanoplm pretrain from-yaml --pure-torch
```

```
Using pure-torch model and training loop
Total Trainable Parameters: 62893760
Flash attention available: True
Batch setup: target_global_batch_tokens=256,000 tokens, micro_step_tokens=32,768, grad_accum_steps=8, effective_global_batch_tokens=262,144 tokens
Starting pure-torch training: total_steps=39 (token-budget driven), warmup_steps=39, global_batch_warmup_steps=0, grad_accum=8, achieved_global_batch_tokens=262,144 tokens
# ...
```

Read on only if you are considering the alternative.

## Pick a backend from two

There are exactly two pretraining backends.

| Backend | Select with | What it is |
| --- | --- | --- |
| **pure-torch** | `--pure-torch` or `pure_torch: true` | nanoPLM's own model and training loop. Every feature lands here first; most land here only. |
| **pure-TE** | `--pure-te` or `pure_te: true` | Transformer Engine model and training loop: TE kernels and the full TE FP8 recipe surface. |

The flags live on both `pretrain from-yaml` and `pretrain run`. `pure_torch` and `pure_te` are **top-level** keys in `pretrain.yaml` — siblings of `model:`, not children of it:

```yaml title="pretrain.yaml"
model:
  # ...
pretraining:
  # ...

pure_torch: true   # top level; the CLI flag overrides this if both are set
```

A CLI flag beats the YAML key. The two are mutually exclusive — setting both raises `pure_torch and pure_te cannot both be true.`

## Run without a backend flag and the run stops

There is no third, default backend. Earlier versions of nanoPLM ran a Hugging Face `Trainer` when you passed no flag; **that path was removed.** If you have seen older docs or an older README, this is the change that invalidates them.

```bash
nanoplm pretrain from-yaml
```

```
# ...
ValueError: Pretraining requires pure_torch: true or pure_te: true.
The HF Trainer pretraining path is no longer supported.
```

The fix is to add `--pure-torch` (or set `pure_torch: true`).

## Know what pure-TE costs you

Pure-TE gives you Transformer Engine's fused kernels and the complete TE FP8 recipe surface (`fp8_recipe`, `fp8_format`, `fp8_amax_history_len`, `fp8_reduce_amax` — see [Scale up](/nanoplm/guides/scale-up/)). FP8 itself is **not** a reason to switch: `fp8: true` works on pure-torch too, and it goes through Transformer Engine either way.

What pure-TE cannot do is the deciding factor. It supports only:

- `lr_schedule: warmup_stable | cosine | linear` — **not `wsd`**, which is the default schedule
- `resume.mode: continue` — **not `decay`**

and it rejects these model and training features outright:

| Rejected on pure-TE | Where it lives |
| --- | --- |
| `use_moe` | `model:` |
| `use_noble` | `model:` |
| `use_canon_layers` | `model:` |
| `use_repo` | `model:` |
| `use_diff_attn_v2` | `model:` |
| `use_paired_head_attention` | `model:` |
| `recycling_loop_count > 1` | `model:` |
| `mlp_activation: glu` | `model:` |
| `fused_qkv: false`, `fused_up_gate: false` | `model:` |
| `mhc_lite_wrapping_level: sublayers` | `model:` |
| `debug_layerwise_metrics: true` | `pretraining:` |

:::caution
`lr_schedule` defaults to `wsd`, so a stock `pretrain.yaml` run with `--pure-te` fails immediately with `lr_schedule: wsd and resume mode: decay require pure_torch: true (TE supports warmup_stable/cosine/linear + continue only).` Switch the schedule to `warmup_stable`, `cosine`, or `linear` before you switch backends.
:::

## Compare the two

| | pure-torch | pure-TE |
| --- | --- | --- |
| `lr_schedule` | `wsd`, `warmup_stable`, `cosine`, `linear` | `warmup_stable`, `cosine`, `linear` |
| `resume.mode` | `continue`, `decay` | `continue` |
| FP8 | yes (`fp8_target`, `fp8_skip_last_n_layers`) | yes (full TE recipe surface) |
| FSDP2 / HSDP / DDP | yes | yes |
| MoE, NOBLE, Canon, RePO, DiffAttnV2, paired-head attention | yes | no |
| [Recycling](/nanoplm/guides/recycling/) | yes | no |
| `mlp_activation: glu` | yes | no |
| `debug_layerwise_metrics` | yes | no |
| `torch.compile` autotune + Triton reduction flags | yes | no |

The full per-key rules are in the [compatibility matrix](/nanoplm/reference/compatibility/).

## Where to go next

- [Scale up](/nanoplm/guides/scale-up/) — multi-GPU, FSDP/HSDP, packing, FP8.
- [pretrain.yaml reference](/nanoplm/reference/pretrain/) — every key, with defaults.
- [Compatibility matrix](/nanoplm/reference/compatibility/) — which feature needs which backend.
