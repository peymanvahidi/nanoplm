---
title: Troubleshooting
description: Every error nanoPLM raises at config load and startup, grouped by symptom, with the exact message and the concrete fix.
---

nanoPLM validates your config **before** it touches a GPU. Almost everything on this page fails in the first second of a run, which is the good case: you fix a key and try again. Find your error message below.

If your run starts and then misbehaves, jump to [Data pipeline](#data-pipeline) or [Distributed errors](#distributed-errors).

## Config won't load

The loaders reject unknown keys. If you carried a config over from an older version of nanoPLM, this is where it surfaces.

**`Unexpected training configuration keys: ...`** (also `Unexpected model configuration keys` / `Unexpected resume configuration keys`)

A key in that block is not a field. The batch and duration surface was redesigned, and these keys no longer exist:

| Dead key | What replaced it |
| --- | --- |
| `micro_batch_size` | [`micro_batch_seqs`](/nanoplm/reference/pretrain/#pretrain-pretraining-micro_batch_seqs): **sequences** per GPU per forward |
| `global_batch_size` | [`global_batch_tokens`](/nanoplm/reference/pretrain/#pretrain-pretraining-global_batch_tokens): **tokens** per optimizer step |
| `num_epochs` | Nothing. Duration is a token budget: `stable_tokens` + `decay_tokens`, or `train_tokens`. |
| `warmup_ratio` | [`warmup_steps`](/nanoplm/reference/pretrain/#pretrain-pretraining-warmup_steps): absolute optimizer steps |
| `reset_scheduler`, `resume_epochs`, `extra_epochs` | [`resume.mode`](/nanoplm/reference/pretrain/#pretrain-resume-mode): `continue` or `decay` |
| `min_lr` | Nothing. It was never a valid key. Use [`lr_decay_to_fraction`](/nanoplm/reference/pretrain/#pretrain-pretraining-lr_decay_to_fraction), a fraction of peak LR. |

**`gradient_accumulation_steps is inferred automatically from global_batch_tokens, micro_batch_seqs, max_seq_len, and world_size. Remove gradient_accumulation_steps from your pretraining config.`**

Accumulation is derived, never configured. Delete the key. To change the effective accumulation, change `global_batch_tokens` or `micro_batch_seqs`.

**`dataset_dir is required in pretraining configuration`**

Set [`dataset_dir`](/nanoplm/reference/pretrain/#pretrain-pretraining-dataset_dir) to the `output_dir` your data pipeline produced: the directory holding `train/`, `val/`, and `.data_manifest`.

Never set `inferred_grad_accum_steps`, `global_batch_seqs`, or `achieved_global_batch_tokens`. They are runtime outputs that happen to live on the same dataclass.

:::danger[The silent one: a leftover `pure_torch:` mapping]
Old configs wrote `pure_torch` as a **block**:

```yaml
pure_torch:
  enabled: false     # <- does nothing
  use_compile: true
```

On current nanoPLM, `pure_torch` is a **top-level boolean**, and the loader evaluates it with `bool(...)`. A non-empty mapping is truthy, so `{enabled: false}` turns pure-torch **on**. There is no error and no warning: you get a backend you did not ask for. Delete the block and use a bare `pure_torch: true` (the template default).
:::

## Pretraining won't start

**`Pretraining requires pure_torch: true. The HF Trainer pretraining path is no longer supported.`**

Pure-torch must be selected: either in the YAML or on the CLI. The stock template already sets `pure_torch: true`, so this usually means a hand-written config is missing the key, has it set to false, or you used `pretrain run` without the flag. There is no HF-Trainer path.

```bash
# YAML path (template default is pure_torch: true)
nanoplm pretrain from-yaml

# Flag-driven path still needs the explicit flag
nanoplm pretrain run --dataset-dir output/data/pretrain_data --pure-torch
```

### Duration and schedule disagree

`lr_schedule` decides which duration keys are legal. `wsd` (the default) uses the two-phase `stable_*` / `decay_*` budget; every other schedule uses a single budget.

**`wsd uses stable_*/decay_* budgets; remove train_tokens/max_steps.`**

```yaml title="pretrain.yaml"
pretraining:
  lr_schedule: "wsd"
  stable_tokens: 9000000   # includes warmup
  decay_tokens: 1000000    # added on top
  # train_tokens: 10000000   <- delete this
```

**`cosine uses train_tokens/max_steps; remove stable_*/decay_* budgets.`** (also for `linear` and `warmup_stable`)

The inverse. A single-phase schedule takes one budget:

```yaml title="pretrain.yaml"
pretraining:
  lr_schedule: "cosine"
  train_tokens: 10000000
  # stable_tokens / decay_tokens   <- delete these
```

**`Set exactly one of stable_tokens or stable_steps for the stable budget, not both.`**

Each budget has a token form and a step form. Pick one. The same message appears for [`decay_tokens`](/nanoplm/reference/pretrain/#pretrain-pretraining-decay_tokens)/[`decay_steps`](/nanoplm/reference/pretrain/#pretrain-pretraining-decay_steps) and [`train_tokens`](/nanoplm/reference/pretrain/#pretrain-pretraining-train_tokens)/[`max_steps`](/nanoplm/reference/pretrain/#pretrain-pretraining-max_steps).

**`wsd requires a stable budget: set stable_tokens (tokens) or stable_steps (steps).`**

You removed both forms. Supply one: [`stable_tokens`](/nanoplm/reference/pretrain/#pretrain-pretraining-stable_tokens) or [`stable_steps`](/nanoplm/reference/pretrain/#pretrain-pretraining-stable_steps).

**`lr_schedule must be one of ['cosine', 'linear', 'warmup_stable', 'wsd'], got 'onecycle'`**

Those four values are the whole list. `decay` is an internal schedule name used by resume-decay; it is not settable here.

**`stable_tokens must be a positive integer, got -1`**

Budgets must be greater than zero.

## Optimizer errors

**`optimizer=muonh is incompatible with model.wo_init='zero'. MuonH initializes each matrix hyperball radius from ||W_init||_F, so zero-initialized Wo matrices are locked at zero. Set model.wo_init='kaiming' (or --wo-init kaiming).`**

The `get-yaml` template ships [`wo_init`](/nanoplm/reference/pretrain/#pretrain-model-wo_init) as `kaiming`, so a stock template does not hit this. It fires on a **hand-written `model:` block that omits the key**, because the underlying default is `zero`, not the template's value. The fix is one line:

```yaml title="pretrain.yaml" ins={3}
model:
  # ...
  wo_init: "kaiming"   # MuonH seeds its radius from ||W_init||, so zero-init is fatal
```

**`Optimizer 'normuon' requires the 'dion' package which is not installed.`**

`muon`, `normuon`, `muonh`, and `aurora` all come from Microsoft's `dion`. A full install puts it in
place via `nanoplm deps install` (see [Install](/nanoplm/start/install/)). The message may still
suggest `pip install nanoplm[cuda]`; **there is no `cuda` extra**. Re-run the deps installer and
verify:

```bash
nanoplm deps install
nanoplm deps verify
```

**`muonh_scope must be one of: all, mlp (got 'ffn')`**

**`No eligible matrix parameters found for Muon (expected 2D hidden-layer weights).`**

A Muon-family optimizer needs 2-D weight matrices to act on. This usually means the model is too small or too unusual for the split-optimizer grouping.

## Resume errors

The stock `resume:` block ships `mode: "decay"` with `decay_tokens`, `decay_shape`, and `lr_decay_to_fraction` all populated. Flipping only `is_resume: true` therefore gives you a **decay** run.

**`resume.mode=continue takes no decay_* fields, but got: decay_tokens, decay_shape (they belong to mode=decay).`**

You set `mode: continue` but left the template's decay fields in place. Delete them:

```yaml title="pretrain.yaml" del={5-7}
resume:
  is_resume: true
  checkpoint_dir: "output/pretraining_checkpoints/run-14071932/checkpoint-5000"
  mode: "continue"
  decay_tokens: 1000000
  decay_shape: "1-sqrt"
  lr_decay_to_fraction: 0.1
```

**`resume decay requires lr_decay_to_fraction`**

Decay mode needs a floor. The template's own comment calls this field an *optional override*; that comment is wrong. Set it:

```yaml title="pretrain.yaml"
resume:
  is_resume: true
  checkpoint_dir: ".../checkpoint-stable-end"
  mode: "decay"
  decay_tokens: 1000000       # or decay_steps, not both
  lr_decay_to_fraction: 0.1   # required, despite what the template comment says
```

**`resume.mode=decay requires a budget: set decay_tokens or decay_steps.`** and **`resume decay: set exactly one of decay_tokens or decay_steps, not both.`**

Exactly one budget, and it must be present.

**`Checkpoint directory does not exist: output/pretraining_checkpoints/run-1/checkpoint-1`**

The template ships a placeholder path. Point [`checkpoint_dir`](/nanoplm/reference/pretrain/#pretrain-resume-checkpoint_dir) at a real `checkpoint-N` directory.

**`resume.mode must be 'continue' or 'decay', got 'reset'`**

The EXTEND/RESET model is gone. See [Cooldown and resuming](/nanoplm/guides/cooldown-and-resume/).

**`resume continue requires total_steps (none provided and the checkpoint's training_state.json has no 'total_steps')`**

The checkpoint predates the current training-state format, so the original budget cannot be recovered. Start a fresh run, or use `mode: decay` with an explicit budget.

## Distributed errors

**`distributed_mode=ddp requires multi_gpu=true.`** (also for `hsdp`)

**`ddp_bucket_cap_mb applies only when distributed_mode=ddp.`**

This one surprises people: it fires whenever `ddp_bucket_cap_mb` is not its default of `25` while you are on FSDP or HSDP. If you tuned the bucket size and later switched to FSDP, reset it to `25`.

**`fsdp_reshard_after_forward applies only when distributed_mode=fsdp or hsdp.`**

**`Unsupported distributed_mode='zero3'. Expected one of: fsdp, hsdp, ddp.`**

HSDP shards inside a node and replicates across nodes, so its shard size has to divide the world cleanly:

**`hsdp_shard_size=3 must evenly divide world_size=8.`**

**`hsdp_shard_size=16 out of range for world_size=8.`**

**`hsdp_shard_size=8 equals world_size=8; the replicate dim would be 1. Use distributed_mode='fsdp' instead of 'hsdp'.`**

If one replica would span the whole world there is nothing to replicate: that is plain FSDP.

:::note[HSDP is YAML-only]
`pretrain run --distributed-mode` accepts only `fsdp` and `ddp`, and there is no `--hsdp-shard-size` flag. To use HSDP, set [`distributed_mode`](/nanoplm/reference/pretrain/#pretrain-pretraining-distributed_mode) and [`hsdp_shard_size`](/nanoplm/reference/pretrain/#pretrain-pretraining-hsdp_shard_size) in `pretrain.yaml` and launch with `from-yaml`. See [Scale up](/nanoplm/guides/scale-up/).
:::

**The job starts, prints nothing, and hangs.** The launched process count disagrees with the configured world size, so some ranks wait on a collective that the others never post. Leave [`world_size`](/nanoplm/reference/pretrain/#pretrain-pretraining-world_size) at `auto` and let `torchrun --nproc-per-node=N` set the topology.

## FP8 errors

FP8 goes through Transformer Engine: pure-torch swaps eligible blocks for TE modules and runs them under a TE autocast. Transformer Engine is therefore required for `fp8: true`, even though nanoPLM has no separate TE backend.

**`fp8=True in the pure-torch pipeline now requires Transformer Engine. Install transformer-engine or run without fp8.`**

Transformer Engine is part of the hard stack from `nanoplm deps install` (built from source as
v2.16). If it is missing, re-run that command rather than `pip install transformer-engine` (prebuilt
wheels often break against cluster CUDA toolkits). Or set `fp8: false`. FP8 tensor cores need Hopper
(H100) or newer; `fp8_recipe: mxfp8` and `nvfp4` need Blackwell.

**`fp8=True requires at least one eligible block for Transformer Engine FP8 conversion target='mlp', skip_last_n_layers=12.`**

You skipped every layer you targeted. Lower [`fp8_skip_last_n_layers`](/nanoplm/reference/pretrain/#pretrain-pretraining-fp8_skip_last_n_layers) (default `0`) or widen [`fp8_target`](/nanoplm/reference/pretrain/#pretrain-pretraining-fp8_target) (default `mlp`; also accepts `attn_wqkv` and `attn_wo`).

**`fp8_recipe must be one of: delayed, current, block, mxfp8, nvfp4 (got 'tensorwise')`** and **`fp8_format must be one of: hybrid, e4m3 (got 'e5m2')`**

CPU activation offload is the fussiest FP8 feature. All four of these are hard errors:

- **`fp8_cpu_activation_offload=true requires fp8=true.`**
- **`fp8_cpu_activation_offload=true requires CUDA.`**
- **`fp8_cpu_activation_offload is not supported with activation_checkpointing=true.`**: pick one; both trade compute for memory.
- **`fp8_cpu_activation_offload is not supported with recycling_loop_count > 1.`**
- **`fp8_cpu_activation_offload=true found no eligible TE wrappers; set fp8_target to include 'attn_wqkv' and/or 'mlp'.`**

[`fp8_cpu_activation_offload`](/nanoplm/reference/pretrain/#pretrain-pretraining-fp8_cpu_activation_offload) was hand-tuned for the 4×GH200 SKU. Performance elsewhere is not guaranteed.

## Model config errors

Several architecture features are mutually exclusive, and several are pure-torch only.

**`model.use_canon_layers=true requires pure_torch: true (or --pure-torch). Canon layers are not implemented in the HF path.`**

The same message shape appears for `use_noble`, `use_diff_attn_v2`, and `use_paired_head_attention`. MoE and recycling have their own wording:

- **`MoE is currently implemented only in the pure-torch path. Disable use_moe or use --pure-torch.`**
- **`Huginn-style recycling (recycling_loop_count > 1) is implemented only in the pure-torch path. Use --pure-torch with recycling_loop_count>1.`**

Keep pure-torch (the template default).

### Incompatible pairs

| Error | Why |
| --- | --- |
| `use_paired_head_attention is not compatible with use_diff_attn_v2.` | Both restructure the head layout. |
| `use_paired_head_attention is not compatible with use_repo.` | RePO replaces the positional scheme paired heads assume. |
| `use_paired_head_attention is not compatible with Canon-B. Remove 'b' from canon_layers_mode.` | Canon-B mixes across the paired-head axis. |
| `use_diff_attn_v2 is not compatible with use_repo.` | |
| `GQA (num_kv_heads != num_attention_heads) is not compatible with use_repo.` | |
| `model.use_mhc_lite=true is not compatible with model.use_resid_lambdas=true.` | `resid_lambdas` rescales the hidden state before each layer, which breaks mHC-lite's doubly-stochastic stability guarantee. |
| `Paired head attention requires an even head_dim. Got head_dim=65.` | |

### MoE

MoE has three prerequisites beyond pure-torch:

- **`use_moe=True requires mlp_activation in {'swiglu', 'srelu'}.`**
- **`MoE requires use_packing=True`** and **`MoE requires use_static_inp_size=True`**: both default to `true`, so this fires only if you turned them off.
- **`MoE is not compatible with activation_checkpointing_mode='attn+mlp'. Use 'attn' or 'layer' mode instead.`**

Routing constraints: `moe_top_k` must satisfy `1 <= moe_top_k <= moe_num_experts`, `moe_num_experts` must be divisible by `moe_n_group`, and `moe_topk_group` must be `<= moe_n_group`.

### Value errors

**`rope_dim_fraction must be in (0.0, 1.0], got 1.5`**

**`Unsupported mlp_activation: gelu. Supported: ['swiglu', 'geglu', 'srelu']`**: `geglu` is the gated gelu MLP used by stock ModernBERT (`gelu(up) * gate`). It was called `glu` before, which was inaccurate.

**`norm_bias=True is not compatible with norm_type='rmsnorm'. RMSNorm does not support bias. Set norm_bias=False.`**

**`use_canon_layers=True requires non-empty canon_layers_mode (for example: 'abcd' or 'ac').`**

**`Invalid character 'X' in attn_layer_pattern. Use 'F' for full attention and 'S' for sliding attention.`**

## Data pipeline

### The `check` stage fails on invalid sequences

By default [`drop_invalid_sequences`](/nanoplm/reference/data/#data-data_params-drop_invalid_sequences) is `false`, which means the `check` stage **exits non-zero** the moment it finds an invalid sequence and halts the pipeline. That is deliberate: you should see what is in your FASTA before you train on it.

Inspect `invalid.fasta` and `validation.json` in the `validation_dir`. Note that `invalid_ids.txt` is **empty** in this mode, so nothing downstream drops records by accident.

To proceed, drop them:

```bash
nanoplm data from-yaml --fix
```

`--fix` rewrites `data_params.drop_invalid_sequences: true` in your config **in place**, then reruns. The `check` stage now records the offending IDs in `invalid_ids.txt`, `filter` excludes them, and the run continues. Setting the key by hand is equivalent:

```yaml title="data.yaml"
data_params:
  drop_invalid_sequences: true
```

### The dataset was prepared for the wrong pipeline

**`Dataset was prepared for 'distillation' pipeline, but 'pretrain' was expected.`**

[`dataset_dir`](/nanoplm/reference/pretrain/#pretrain-pretraining-dataset_dir) points at a distillation dataset (HDF5 teacher embeddings), not pretraining shards. Point it at the `output_dir` from a `nanoplm data from-yaml` run made **without** `--distillation`.

### Outputs land somewhere unexpected

Paths in `data.yaml` resolve relative to the directory you **run the command from**, not to the YAML file. Run every command from one project root, or use absolute paths.

### Editing `.data_manifest` by hand

Don't. It is generated, it carries a `# DO NOT EDIT THIS FILE MANUALLY.` header, and it records the exact parameters the binary shards were built with. If you need different `max_seq_len` or `samples_per_shard`, change `data.yaml` and re-run `nanoplm data from-yaml` so the shards and the manifest are regenerated together.

## Still stuck?

Check that the key you are setting actually exists: the [pretrain.yaml](/nanoplm/reference/pretrain/), [data.yaml](/nanoplm/reference/data/), and [eval.yaml](/nanoplm/reference/eval/) reference tables are generated from the source, so a key that is not in them is not a key. If a run fails and the message is not on this page, open an issue on [GitHub](https://github.com/peymanvahidi/nanoplm) with the config and the full traceback.
