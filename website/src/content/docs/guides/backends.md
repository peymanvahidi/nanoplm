---
title: Choose a backend
description: "nanoPLM pretrains on pure-torch or pure-TE. Pure-torch is the default; use pure-TE only when you need Transformer Engine kernels, and know what it cannot do."
---

Use pure-torch. It is the backend every feature in nanoPLM supports, the one the stock `pretrain.yaml` selects, and the one the project's own launcher uses.

```bash
nanoplm pretrain from-yaml
```

```
Using pure-torch model and training loop
Total Trainable Parameters: 62893760
Flash attention available: True
Batch setup: target_global_batch_tokens=256,000 tokens, micro_step_tokens=32,768, grad_accum_steps=8, effective_global_batch_tokens=262,144 tokens
Starting pure-torch training: total_steps=39 (token-budget driven), warmup_steps=39, global_batch_warmup_steps=0, grad_accum=8, achieved_global_batch_tokens=262,144 tokens
# ...
```

No backend CLI flag is required for a normal YAML run: `get-yaml` writes `pure_torch: true` at the bottom of the file. Read on only if you are considering the alternative.

## Pick a backend from two

There are exactly two pretraining backends.

| Backend | Select with | What it is |
| --- | --- | --- |
| **pure-torch** | `pure_torch: true` (template default) or `--pure-torch` | nanoPLM's own model and training loop. Every feature lands here first; most land here only. |
| **pure-TE** | `pure_te: true` or `--pure-te` | Transformer Engine model and training loop: TE kernels and the full TE FP8 recipe surface. |

The flags live on both `pretrain from-yaml` and `pretrain run`. `pure_torch` and `pure_te` are **top-level** keys in `pretrain.yaml` (siblings of `model:`, not children of it):

```yaml title="pretrain.yaml"
model:
  # ...
pretraining:
  # ...

pure_torch: true   # top level; the CLI flag overrides this if both are set
```

A CLI flag can force a backend on, but it does not clear the other key in the YAML. With the stock `pure_torch: true`, passing only `--pure-te` raises `pure_torch and pure_te cannot both be true.` — set `pure_torch: false` (or remove it) when you switch. The two backends are always mutually exclusive.

## When a run still fails without a backend

You still need **some** backend selected — either in the YAML or on the CLI. Earlier versions of nanoPLM ran a Hugging Face `Trainer` when neither was set; **that path was removed.**

With a stock template this does not come up. It fires when you hand-write a config without `pure_torch` / `pure_te`, delete the key, set both to false, or use `pretrain run` (flag-driven, no YAML defaults) without a backend flag:

```bash
nanoplm pretrain run --dataset-dir output/data/pretrain_data
```

```
# ...
ValueError: Pretraining requires pure_torch: true or pure_te: true.
The HF Trainer pretraining path is no longer supported.
```

The fix is `pure_torch: true` in the YAML (the template default), or `--pure-torch` / `--pure-te` on the command line.

## Know what pure-TE costs you

Pure-TE gives you Transformer Engine's fused kernels and the complete TE FP8 recipe surface (`fp8_recipe`, `fp8_format`, `fp8_amax_history_len`, `fp8_reduce_amax`: see [Scale up](/nanoplm/guides/scale-up/)). FP8 itself is **not** a reason to switch: `fp8: true` works on pure-torch too, and it goes through Transformer Engine either way.

What pure-TE cannot do is the deciding factor. It supports only:

- `lr_schedule: warmup_stable | cosine | linear`: **not `wsd`**, which is the default schedule
- `resume.mode: continue`: **not `decay`**

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
| `mlp_activation: geglu` | `model:` |
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
| `mlp_activation: geglu` | yes | no |
| `debug_layerwise_metrics` | yes | no |
| `torch.compile` autotune + Triton reduction flags | yes | no |

The full per-key rules are in the [compatibility matrix](/nanoplm/reference/compatibility/).

## Where to go next

- [Scale up](/nanoplm/guides/scale-up/): multi-GPU, FSDP/HSDP, packing, FP8.
- [pretrain.yaml reference](/nanoplm/reference/pretrain/): every key, with defaults.
- [Compatibility matrix](/nanoplm/reference/compatibility/): which feature needs which backend.
