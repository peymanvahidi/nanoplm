---
title: Scale up
description: "Add GPUs, shard with FSDP2 or HSDP, pack tokens, trade VRAM for throughput, and turn on FP8: the knobs that make a nanoPLM run bigger or faster."
---

You have trained one model on one GPU. This page is the order in which you reach for more: more GPUs first, then the sharding strategy, then the knobs that squeeze the GPUs you already have.

Every key here lives in `pretrain.yaml` and is documented in the [pretrain.yaml reference](/nanoplm/reference/pretrain/).

## Add GPUs

Launch with `torchrun`. nanoPLM's own launcher is a single line:

```bash
torchrun --nproc-per-node=4 $(which nanoplm) pretrain from-yaml --pure-torch
```

```
Batch setup: target_global_batch_tokens=256,000 tokens, micro_step_tokens=131,072, grad_accum_steps=2, effective_global_batch_tokens=262,144 tokens
Distributed config: multi_gpu=True mode=fsdp
Starting pure-torch training: total_steps=39 (token-budget driven), warmup_steps=39, global_batch_warmup_steps=0, grad_accum=2, achieved_global_batch_tokens=262,144 tokens
# ...
```

`torchrun` runs a *Python file*, not a shell command, so you hand it the path to the installed console script with `$(which nanoplm)` and put the subcommand after it. `torchrun nanoplm pretrain ...` does not work.

The two YAML keys that matter:

```yaml title="pretrain.yaml"
pretraining:
  multi_gpu: true      # master switch; false forces world_size to 1
  world_size: 'auto'   # reads $WORLD_SIZE from torchrun, else torch.cuda.device_count()
```

Leave [`world_size`](/nanoplm/reference/pretrain/#pretrain-pretraining-world_size) at `'auto'`: `torchrun --nproc-per-node` already told it the answer.

Adding GPUs does **not** change your token budget or your global batch. It changes how many micro-steps it takes to reach one optimizer step: `grad_accum` is inferred from `global_batch_tokens / (micro_batch_seqs × max_seq_len × world_size)` and shrinks as `world_size` grows. Four GPUs means the same batch, four times faster.

## Choose a sharding strategy

[`distributed_mode`](/nanoplm/reference/pretrain/#pretrain-pretraining-distributed_mode) picks how parameters, gradients, and optimizer state are spread across ranks.

| Mode | What it does | Reach for it when |
| --- | --- | --- |
| `fsdp` *(default)* | FSDP2: shards params, grads, and optimizer state across **all** ranks. | Almost always. This is the right answer for a single node and for most multi-node runs. |
| `hsdp` | Shards *within* a group of GPUs, replicates *across* groups. | Multi-node, where the inter-node fabric is the bottleneck and the model fits sharded inside one node. |
| `ddp` | Full replica per GPU, gradient all-reduce. | Small models where sharding overhead outweighs the memory it saves. `ddp_bucket_cap_mb` (default `25`) is valid only in this mode. |

HSDP is the interesting one. With `distributed_mode: hsdp`, nanoPLM builds a 2-D device mesh: parameters are sharded over the fast intra-node links (NVLink) and gradients are all-reduced over the slow inter-node link once per step, instead of all-gathering parameters across the whole cluster on every layer.

```yaml title="pretrain.yaml"
pretraining:
  distributed_mode: "hsdp"
  hsdp_shard_size: 'auto'   # 'auto' = LOCAL_WORLD_SIZE -> one full sharded replica per node
```

```
HSDP enabled: mesh=(replicate=2, shard=8), world_size=16
```

[`hsdp_shard_size`](/nanoplm/reference/pretrain/#pretrain-pretraining-hsdp_shard_size) is the number of GPUs per replica. `'auto'` resolves to `LOCAL_WORLD_SIZE`, which puts exactly one sharded replica on each node: the setting HSDP exists for. If you set it by hand, two rules are enforced:

- it must evenly divide `world_size`, otherwise `hsdp_shard_size=6 must evenly divide world_size=16.`
- it must not *equal* `world_size`: that would leave a replicate dimension of 1, and the run tells you to use `distributed_mode: 'fsdp'` instead.

:::caution
HSDP is **YAML-only**. `nanoplm pretrain run` has no `--distributed-mode hsdp` and no `--hsdp-shard-size` flag; its choices are `fsdp` and `ddp`. Use `pretrain from-yaml`: it is the only path that exposes the full surface.
:::

## Tune FSDP for your interconnect

Two keys change how FSDP2 wraps the model. Both apply to `fsdp` and `hsdp`.

| Key | Default | Set it to |
| --- | --- | --- |
| [`fsdp_shard_granularity`](/nanoplm/reference/pretrain/#pretrain-pretraining-fsdp_shard_granularity) | `layer` | `layer` wraps whole transformer blocks: fewer communication boundaries, less dispatch overhead. Right for NVLink/SXM machines. `sublayer` wraps attention and MLP separately: more boundaries, finer communication/compute overlap. Right for PCIe machines, where communication is slow enough that overlapping it pays. |
| [`fsdp_reshard_after_forward`](/nanoplm/reference/pretrain/#pretrain-pretraining-fsdp_reshard_after_forward) | `false` | `true` frees the all-gathered parameters after the forward pass and re-gathers them in backward. It buys memory and costs one extra all-gather per group. Turn it on when you are out of VRAM and have already turned on activation checkpointing. |

Setting `fsdp_reshard_after_forward` while `distributed_mode: ddp` is an error, not a no-op: `fsdp_reshard_after_forward applies only when distributed_mode=fsdp or hsdp.`

## Pack tokens instead of padding them

Protein sequences vary in length, and a padded batch spends real FLOPs on `<pad>`. [`use_packing`](/nanoplm/reference/pretrain/#pretrain-pretraining-use_packing) (default `true`) concatenates variable-length sequences into one flat row and hands attention a `cu_seqlens` index that says where each sequence starts and ends, so no attention mass crosses a sequence boundary and no compute lands on padding.

This is *why* the batch surface is measured in tokens. `global_batch_tokens` is a promise about how many real amino acids reach the optimizer; `micro_batch_seqs` is only how many sequences a single forward pass tries to fit. The run logs the padding waste it did not pay:

```
[step 20/39] loss=2.9814 lr=9.74e-05 grad_norm=0.9127 tok/s=412,880 real_tok/s=411,203 real_tok/step=261,120 dt=634.51ms wall=13.2s waste=0.4% h100_mfu=38.11% vram=41.2GB
```

Two things packing depends on:

- **Flash attention.** The packed varlen path needs it. The run prints `Flash attention available: True` at startup; without it, turn packing off and accept the pad waste.
- **Static shapes**, if you want `torch.compile` and CUDA graphs to help. [`use_static_inp_size`](/nanoplm/reference/pretrain/#pretrain-pretraining-use_static_inp_size) (default `true`) rounds each packed batch's sequence count and max sequence length up to powers of two, so the compiler sees a handful of shapes instead of a new one every step. With `use_packing: true` and `use_static_inp_size: true` the model compiles in static mode; drop either and it falls back to dynamic.

MoE (`model.use_moe: true`) hard-requires both: `MoE requires use_packing=True` / `MoE requires use_static_inp_size=True`.

## Trade VRAM for throughput

Two knobs, in the order you should reach for them.

**Activation checkpointing** recomputes activations in the backward pass instead of storing them. It lives under `model:`.

```yaml title="pretrain.yaml"
model:
  activation_checkpointing: true
  activation_checkpointing_mode: "attn"  # layer | attn | attn+mlp
```

[`activation_checkpointing_mode: attn`](/nanoplm/reference/pretrain/#pretrain-model-activation_checkpointing_mode) has the best return: attention activations are the largest and the cheapest to recompute. `layer` and `attn+mlp` save more memory for more recompute. (MoE rejects `attn+mlp`.)

**`micro_batch_seqs`** is the VRAM dial. It is the number of sequences one GPU holds in one forward pass: halve it and peak memory roughly halves.

```yaml title="pretrain.yaml"
pretraining:
  micro_batch_seqs: 32       # was 64; halves activation memory per GPU
  global_batch_tokens: 256000  # unchanged: this is the number that defines the run
```

Halving [`micro_batch_seqs`](/nanoplm/reference/pretrain/#pretrain-pretraining-micro_batch_seqs) does not change your training recipe. Gradient accumulation is inferred, so it doubles to absorb the change and the token budget per optimizer step stays where you put it. You are trading wall-clock for memory, not changing the math.

## Warm up the batch size

Large-batch runs are unstable early, when the gradient is dominated by trivially predictable tokens. [`global_batch_warmup_steps`](/nanoplm/reference/pretrain/#pretrain-pretraining-global_batch_warmup_steps) (default `0`, off) ramps the global batch up over the first N optimizer steps: three equal thirds at 1/8, then 1/4, then 1/2 of the target, reaching full size at step N.

```yaml title="pretrain.yaml"
pretraining:
  global_batch_warmup_steps: 300  # 1/8 for steps 0-99, 1/4 for 100-199, 1/2 for 200-299
```

```
[step 100] Batch-size warmup -> grad_accum=2 micro_batch=64 split_divisor=1
[step 200] Batch-size warmup -> grad_accum=4 micro_batch=64 split_divisor=1
```

The ramp comes out of gradient accumulation first, and only splits the micro-batch if a factor is left over.

:::note
The run warns that `the token budget is approximate under batch-size warmup`. Steps during the ramp consume fewer tokens than a full step, and the token-to-step conversion assumes full steps, so the realized budget undershoots slightly. Keep `global_batch_warmup_steps` small relative to the run.
:::

## Train in FP8

`fp8: true` halves the bytes moved through the eligible linear layers. Two requirements before you enable it:

- **Transformer Engine must be installed**: even on pure-torch. nanoPLM swaps eligible blocks for TE modules and runs them under a TE autocast. A normal install gets TE from `nanoplm deps install` (see [Install](/nanoplm/start/install/)). Without TE: `fp8=True in the pure-torch pipeline now requires Transformer Engine. Install transformer-engine or run without fp8.`
- **Hopper (H100) or newer.** FP8 tensor cores do not exist on Ampere.

```yaml title="pretrain.yaml"
pretraining:
  fp8: true
  fp8_target: "mlp"            # pure-torch: mlp | attn_wqkv | attn_wo (combine: "attn_wqkv,mlp")
  fp8_skip_last_n_layers: 2    # pure-torch: keep the final layers in bf16 for stability
```

```
FP8 enabled (Transformer Engine conversion): target=mlp, recipe=delayed, skip_last_n_layers=2, converted_blocks=10
```

Which keys apply where:

| Key | Default | Honored by | Notes |
| --- | --- | --- | --- |
| [`fp8_target`](/nanoplm/reference/pretrain/#pretrain-pretraining-fp8_target) | `mlp` | pure-torch | `mlp`, `attn_wqkv`, `attn_wo`. Comma-separated or a YAML list. Start at `mlp`; it is the largest and most forgiving. |
| [`fp8_skip_last_n_layers`](/nanoplm/reference/pretrain/#pretrain-pretraining-fp8_skip_last_n_layers) | `0` | pure-torch | Keeps the final N transformer layers in bf16. The first thing to try if FP8 loss diverges. |
| [`fp8_recipe`](/nanoplm/reference/pretrain/#pretrain-pretraining-fp8_recipe) | `delayed` | both | `delayed \| current \| block \| mxfp8 \| nvfp4`. `mxfp8` and `nvfp4` need **Blackwell or newer**; the other three run on Hopper+. |
| [`fp8_format`](/nanoplm/reference/pretrain/#pretrain-pretraining-fp8_format) | `hybrid` | both | `hybrid \| e4m3`. Ignored by `nvfp4`. |
| [`fp8_amax_history_len`](/nanoplm/reference/pretrain/#pretrain-pretraining-fp8_amax_history_len) | `64` | both | Scaling-factor history window. Used **only** with `fp8_recipe: delayed`. |
| [`fp8_reduce_amax`](/nanoplm/reference/pretrain/#pretrain-pretraining-fp8_reduce_amax) | `true` | both | Synchronizes the scaling factors across ranks. `delayed` only. |

`fp8_cpu_activation_offload` also exists: it pushes FP8 activations to host memory through saved-tensor hooks. It was hand-tuned for a 4×GH200 machine, performance elsewhere is not guaranteed, and it is incompatible with `activation_checkpointing: true` and with [recycling](/nanoplm/guides/recycling/). Leave it off unless you are on that hardware.

## Turn on compile autotune

Three pure-torch-only keys, all `false` by default, all in the same category: try it, measure tokens/second, keep it if it wins.

| Key | What it does |
| --- | --- |
| [`use_compile_max_autotune`](/nanoplm/reference/pretrain/#pretrain-pretraining-use_compile_max_autotune) | Lets Inductor benchmark kernel variants instead of using heuristics. Can improve throughput; adds a long compile/autotune phase at the start of the run. |
| [`compile_triton_persistent_reductions`](/nanoplm/reference/pretrain/#pretrain-pretraining-compile_triton_persistent_reductions) | Faster reduction kernels, if they work on your shapes and GPU. |
| [`compile_triton_mix_order_reduction`](/nanoplm/reference/pretrain/#pretrain-pretraining-compile_triton_mix_order_reduction) | Mix-order reductions. Can be faster; can hit shared-memory limits. |

The `tok/s` and `h100_mfu` fields in the step log are how you tell whether any of them helped.

## Where to go next

- [Choose a backend](/nanoplm/guides/backends/): why `--pure-torch`.
- [Cooldown and resuming](/nanoplm/guides/cooldown-and-resume/): WSD, and picking a long run back up.
- [pretrain.yaml reference](/nanoplm/reference/pretrain/): every key, with defaults.
