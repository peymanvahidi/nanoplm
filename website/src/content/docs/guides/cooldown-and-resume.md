---
title: Cooldown and resuming
description: WSD anneals inside a single run, so most people need nothing here. Resume a dead run with mode continue, and fan several cooldowns out of one trunk with mode decay.
---

**If you want one model, use the default schedule and stop reading.** `lr_schedule: wsd` runs warmup, stable, and decay inside a single run. Older versions of nanoPLM made you fake a cooldown with a scheduler-resetting resume; that workaround is obsolete and the keys it used no longer exist.

The rest of this page is for two other situations: a run that died, and a trunk you want to anneal several different ways.

## Anneal inside one run

WSD (warmup, stable, decay) is the default `lr_schedule`. The learning rate ramps to its peak, sits there for the bulk of the run, then anneals to a floor over a short tail.

```yaml title="pretrain.yaml"
pretraining:
  lr_schedule: "wsd"        # wsd | warmup_stable | cosine | linear
  warmup_steps: 1000        # optimizer steps, 0 -> peak LR
  stable_tokens: 9000000    # tokens up to the decay boundary: warmup INCLUDED
  decay_tokens: 1000000     # additional anneal tokens, on top of stable_tokens
  decay_shape: "1-sqrt"     # 1-sqrt | linear
  lr_decay_to_fraction: 0.1 # LR floor = 0.1 x peak
```

Total duration is `stable_tokens + decay_tokens`: 10M tokens in the example above, of which the last 1M are the anneal.

:::note
`stable_tokens` **includes** warmup. It is the position of the stable-to-decay boundary, not the length of the stable phase. `decay_tokens` is a length, added on top. Getting this backwards is the most common way to end up with a decay that is twice as long as you meant.
:::

`decay_shape` picks how the LR falls between the peak and the floor:

| Value | Curve | Use it when |
| --- | --- | --- |
| `1-sqrt` *(default)* | Drops fast early, then a long shallow tail. | Most runs. This is the WSD anneal as published. |
| `linear` | Constant slope from peak to floor. | You want the simpler, more predictable curve to compare against. |

[`lr_decay_to_fraction`](/nanoplm/reference/pretrain/#pretrain-pretraining-lr_decay_to_fraction) (default `0.1`) is the floor expressed as a fraction of peak LR, not an absolute value. There is no `min_lr` key.

At the boundary the run saves a checkpoint on its own:

```
Checkpoint saved -> output/pretraining_checkpoints/<run>/checkpoint-stable-end
[step 36/39] loss=2.4109 lr=6.31e-05 grad_norm=0.8410 tok/s=411,204 dt=637.02ms wall=23.1s h100_mfu=38.02%
```

That `checkpoint-stable-end` is the model as it stood immediately before the anneal. Keep it: the last section of this page is built on it.

## Resume a run that died

A node dropped, a job hit its wall-clock limit, the process was killed. You want to finish the run you started, on the schedule you started it with.

That is `resume.mode: continue`.

```yaml title="pretrain.yaml" ins={4} del={5-7}
resume:
  is_resume: true
  checkpoint_dir: "output/pretraining_checkpoints/<run>/checkpoint-15000"
  mode: "continue"
  decay_tokens: 1000000
  decay_shape: "1-sqrt"
  lr_decay_to_fraction: 0.1
```

Set `mode: "continue"` and **delete the three decay keys**. Then relaunch exactly as before:

```bash
nanoplm pretrain from-yaml
```

```
Resuming from checkpoint: output/pretraining_checkpoints/<run>/checkpoint-15000 (mode=continue)
Resumed at global_step=15000, epoch=1
Resume schedule built [continue]: start_step=15000, total_steps=39000, schedule=wsd, peaks=['1.00e-04']
# ...
```

Continue reads the *original* schedule back out of the checkpoint's `training_state.json` (the schedule name, warmup length, total steps, stable and decay boundaries, shape, and floor) and fast-forwards two things: the LR curve, so step 15,001 gets exactly the LR it would have had, and the dataloader, so you resume at the sequence you were on rather than replaying data you have already seen. (If the dataset fingerprint changed, it says so and restarts the pass from the beginning instead of pretending.)

Nothing in `pretraining:` needs to change. Continue does not take a new budget, a new peak LR, or a new schedule.

:::caution[The scaffold does not ship a continue block]
`nanoplm pretrain get-yaml` writes `mode: "decay"` with `decay_tokens`, `decay_shape`, and `lr_decay_to_fraction` already filled in. Two consequences:

- Flip `is_resume: true` and change nothing else, and you get a **decay** run (a short anneal that overwrites your LR schedule), not the continue you wanted.
- Set `mode: continue` but leave the decay keys in place, and the config will not load:

  ```
  resume.mode=continue takes no decay_* fields, but got: decay_tokens,
  decay_shape, lr_decay_to_fraction (they belong to mode=decay).
  ```

Delete the three keys. Leaving them commented out is fine; leaving them set is not.
:::

## Fan several cooldowns out of one trunk

`mode: decay` exists for one workflow: comparing anneal recipes without paying for the stable phase more than once.

The stable phase is where nearly all of the compute goes. If you want to know whether a 1M-token `1-sqrt` anneal beats a 2M-token `linear` anneal, you do not want to retrain the trunk twice.

**Step 1: train a trunk that never decays.** `lr_schedule: warmup_stable` warms up, then holds the peak LR to the end of its budget, and saves `checkpoint-stable-end` when it finishes.

```yaml title="pretrain.yaml"
pretraining:
  lr_schedule: "warmup_stable"  # warmup -> peak -> hold; no anneal
  warmup_steps: 1000
  train_tokens: 9000000         # single-phase budget: no stable_*/decay_* keys with this schedule
  global_batch_tokens: 256000   # remember this number: the branches must match it
```

```bash
torchrun --nproc-per-node=4 $(which nanoplm) pretrain from-yaml
```

```
Checkpoint saved -> output/pretraining_checkpoints/<trunk>/checkpoint-stable-end
Pure-torch pretraining complete.
```

**Step 2: branch a cooldown off it.** Point `resume.checkpoint_dir` at that `checkpoint-stable-end` and give it a decay budget. The peak LR of the branch is the LR the trunk was sitting at, read straight out of the checkpoint's optimizer state, so you do not set a learning rate here.

```yaml title="pretrain.yaml"
pretraining:
  global_batch_tokens: 256000   # identical to the trunk
  # ...

resume:
  is_resume: true
  checkpoint_dir: "output/pretraining_checkpoints/<trunk>/checkpoint-stable-end"
  mode: "decay"
  decay_tokens: 1000000    # exactly one of decay_tokens / decay_steps
  decay_shape: "1-sqrt"    # 1-sqrt | linear
  lr_decay_to_fraction: 0.1  # REQUIRED for decay mode
```

```
Resuming from checkpoint: output/pretraining_checkpoints/<trunk>/checkpoint-stable-end (mode=decay)
Resume schedule built [decay]: decay_steps=4, floor=0.1, shape=1-sqrt, peaks=['1.00e-04']
Resume decay: total_steps=4 (decay), peak LR from checkpoint, warmup=0, terminal checkpoint -> checkpoint-final
# ...
Checkpoint saved -> output/pretraining_checkpoints/<branch>/checkpoint-final
```

Change `decay_tokens`, `decay_shape`, or `lr_decay_to_fraction` and run it again against the same trunk to get a second cooldown, a third, a fourth. Each branch is a fresh short run: `global_step` restarts at 0, warmup is 0, and the terminal checkpoint is named `checkpoint-final`.

:::caution[`lr_decay_to_fraction` is required, whatever the comment says]
The scaffold's comment calls `decay_shape` and `lr_decay_to_fraction` *"OPTIONAL overrides"*. That is true of `decay_shape` (it falls back to `1-sqrt`). It is **not** true of `lr_decay_to_fraction`: omit it and the run raises `resume decay requires lr_decay_to_fraction`. Always set it.
:::

Three more rules for decay branches:

- **Exactly one of `decay_tokens` or `decay_steps`.** Both is `resume decay: set exactly one of decay_tokens or decay_steps, not both.`; neither is `resume.mode=decay requires a budget: set decay_tokens or decay_steps.`
- **Keep `global_batch_tokens` and `world_size` identical to the trunk's.** They set how many tokens one optimizer step consumes, which is what turns your `decay_tokens` into a step count. Change either and the branch's LR curve stops matching a single-run WSD decay. The run warns you if it notices.
- **A branch continues the trunk's data.** The decay fast-forwards the dataloader to the position saved in the trunk checkpoint, the same way `continue` does, so it trains on the data a single-run WSD decay would see next, and every branch from one trunk sees the same cooldown data.

## Where to go next

- [Choose an optimizer](/nanoplm/guides/optimizers/): a Muon run puts two learning rates on this schedule.
- [pretrain.yaml reference](/nanoplm/reference/pretrain/): every `pretraining:` and `resume:` key, with defaults.
