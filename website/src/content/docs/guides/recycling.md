---
title: Recycling
description: Huginn-style latent recurrence loops a weight-tied block to buy effective depth without extra parameters. Turn it on with recycling_loop_count, and know what it refuses to run alongside.
---

Recycling buys depth without buying parameters.

The layer stack is cut into three pieces: a **prelude** that runs once, a **recurrent block** whose weights are shared and which is looped several times, and a **coda** that runs once. A 12-layer model with a 2-layer prelude, an 8-layer recurrent block looped 4 times, and a 2-layer coda does the work of a 36-layer forward pass while holding 12 layers' worth of weights.

You pay for it in compute, not memory: a loop count of `r` costs roughly `r` times the recurrent block's FLOPs. Gradients are truncated to the last few iterations, so activation memory stays bounded no matter how deep the loop goes.

## Turn recycling on

Recycling activates when [`model.recycling_loop_count`](/nanoplm/reference/pretrain/#pretrain-model-recycling_loop_count) is greater than `1`. The default is `1`, which is off.

```yaml title="pretrain.yaml"
model:
  num_hidden_layers: 12
  recycling_loop_count: 4            # target mean loops through the recurrent block (1 = off)
  recycling_num_prelude_layers: 2    # P: first 2 layers run once
  recycling_num_coda_layers: 2       # C: last 2 layers run once
                                     # recurrent block = L - P - C = 8 weight-tied layers
```

It is **pure-torch only** (the stock template already selects pure-torch):

```bash
nanoplm pretrain from-yaml
```

```
Using pure-torch model and training loop
Total Trainable Parameters: 62893760
Flash attention available: True
# ...
```

Running it with `--pure-te` stops the config load: `recycling: incompatible pipeline flag 'pure_te'. te_pipeline.py is out of scope for the v1 recycling implementation. Use pure_torch.`

## Split the stack

Three `model:` keys define the shape.

| Key | Default | Meaning |
| --- | --- | --- |
| [`recycling_loop_count`](/nanoplm/reference/pretrain/#pretrain-model-recycling_loop_count) | `1` | Target **mean** number of times the recurrent block runs. `1` disables recycling entirely. |
| [`recycling_num_prelude_layers`](/nanoplm/reference/pretrain/#pretrain-model-recycling_num_prelude_layers) | `0` | P: the first P layers run once, before the loop. |
| [`recycling_num_coda_layers`](/nanoplm/reference/pretrain/#pretrain-model-recycling_num_coda_layers) | `0` | C: the last C layers run once, after the loop. |

The recurrent block is whatever is left: `num_hidden_layers - P - C` layers, weight-tied and looped. With `P = C = 0` the entire stack becomes the recurrent block.

Two more knobs follow the Huginn paper and are fine at their defaults: `recycling_adapter_bias` (`false`), `recycling_s0_noise_sigma`, and `recycling_use_takase_init` (`true`).

## Tune the training side

These live under `pretraining:` and do nothing unless `recycling_loop_count > 1`.

```yaml title="pretrain.yaml"
pretraining:
  recycling_bptt: 8                                   # only the last 8 iterations get gradients
  recycling_loop_count_sampling: "lognormal_poisson"  # fixed | lognormal_poisson
  recycling_loop_count_sigma: 0.5                     # spread of the sampled loop count
  recycling_loop_count_curriculum: "1_sqrt"           # none | linear | 1_sqrt
  recycling_loop_count_curriculum_steps: -1           # -1 = auto (25% of total training steps)
  recycling_pretraining_eval_loop_counts: null        # null = eval at the target mean only
```

| Key | Default | What it controls |
| --- | --- | --- |
| [`recycling_bptt`](/nanoplm/reference/pretrain/#pretrain-pretraining-recycling_bptt) | `8` | Truncated-BPTT depth. Only the last K iterations keep gradients, which is what keeps memory flat as the loop count rises. |
| [`recycling_loop_count_sampling`](/nanoplm/reference/pretrain/#pretrain-pretraining-recycling_loop_count_sampling) | `lognormal_poisson` | `fixed` uses `recycling_loop_count` every step. `lognormal_poisson` samples a loop count per step around that mean, so the model learns to be useful at any depth. |
| [`recycling_loop_count_sigma`](/nanoplm/reference/pretrain/#pretrain-pretraining-recycling_loop_count_sigma) | `0.5` | Width of that distribution. Ignored when sampling is `fixed`. |
| [`recycling_loop_count_curriculum`](/nanoplm/reference/pretrain/#pretrain-pretraining-recycling_loop_count_curriculum) | `1_sqrt` | Ramps the mean loop count up over training instead of starting deep. `none` disables the ramp. |
| [`recycling_loop_count_curriculum_steps`](/nanoplm/reference/pretrain/#pretrain-pretraining-recycling_loop_count_curriculum_steps) | `-1` | Length of that ramp in optimizer steps. `-1` means 25% of the run. |
| [`recycling_pretraining_eval_loop_counts`](/nanoplm/reference/pretrain/#pretrain-pretraining-recycling_pretraining_eval_loop_counts) | `null` | A list, e.g. `[1, 4]`, evaluates at several depths each eval step: the cheapest way to see whether extra loops are actually buying accuracy. |

## Check compatibility before you run

Recycling's compatibility rules are enforced at config load, not written down as advice. Four model flags are **rejected**:

| Rejected flag | Why |
| --- | --- |
| `use_moe` | MoE's per-layer routing state is overwritten on every iteration; only the last iteration's decisions would survive. |
| `use_x0_lambdas` | `x0` is captured once before the loop; re-applying the blend every iteration has unvalidated semantics. |
| `use_resid_lambdas` | Per-layer residual scaling re-applied every iteration interacts with the adapter and the `s_0` noise in untested ways. |
| `use_mhc_lite` | mHC-lite's stability guarantees have not been tested against a recycled residual stream. |

The error names the flag: `recycling: incompatible model flag 'use_moe=True'. ...`

And the pure-TE backend is rejected, as above.

Three more are **allowed with a warning**: they run, but nobody has validated them:

- `use_repo`: RePO re-predicts head positions on every iteration. Set `repo_after_n_layers` so RePO lands in the coda.
- `fp8`: FP8 combined with truncated BPTT is not validated for gradient correctness.
- `use_compile_max_autotune`: compiles the recurrent *step* rather than the unrolled loop, and triggers an extra smoke test at startup.

:::caution
`fp8_cpu_activation_offload` is not "warn", it is a hard stop: `fp8_cpu_activation_offload is not supported with recycling_loop_count > 1.`
:::

The full matrix is on the [compatibility reference](/nanoplm/reference/compatibility/).

## A worked config

A 12-layer model turned into an effective 36 layers, with per-step depth sampling and a curriculum:

```yaml title="pretrain.yaml"
model:
  hidden_size: 768
  intermediate_size: 1536
  num_hidden_layers: 12
  num_attention_heads: 8
  # 2 prelude + (8 recurrent x 4 loops) + 2 coda = 36 effective layers, 12 layers of weights
  recycling_num_prelude_layers: 2
  recycling_num_coda_layers: 2
  recycling_loop_count: 4
  use_moe: false          # rejected under recycling
  use_resid_lambdas: false  # rejected under recycling
  use_x0_lambdas: false     # rejected under recycling
  use_mhc_lite: false       # rejected under recycling

pretraining:
  dataset_dir: "output/data/pretrain_data"
  micro_batch_seqs: 64
  global_batch_tokens: 256000
  lr_schedule: "wsd"
  warmup_steps: 1000
  stable_tokens: 9000000
  decay_tokens: 1000000
  recycling_bptt: 8                                   # gradients on the last 8 of 4+ iterations
  recycling_loop_count_sampling: "lognormal_poisson"  # train at varying depth, eval at any
  recycling_loop_count_curriculum: "1_sqrt"           # start shallow, deepen over the first 25%
  recycling_pretraining_eval_loop_counts: [1, 4]      # is depth 4 actually beating depth 1?
# ...

pure_torch: true
```

```bash
nanoplm pretrain from-yaml
```

```
Starting pure-torch training: total_steps=39 (token-budget driven), warmup_steps=39, global_batch_warmup_steps=0, grad_accum=8, achieved_global_batch_tokens=262,144 tokens
# ...
[step 10] in-loop eval r=1: 3.1204, r=4: 2.8871
```

If the two eval losses stay close, the extra loops are not earning their compute: lower `recycling_loop_count`.

## Where to go next

- [Choose a backend](/nanoplm/guides/backends/): recycling is pure-torch only.
- [Compatibility matrix](/nanoplm/reference/compatibility/): every flag recycling rejects or warns on.
- [pretrain.yaml reference](/nanoplm/reference/pretrain/): every `recycling_*` key, with defaults.
