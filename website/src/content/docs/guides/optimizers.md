---
title: Choose an optimizer
description: nanoPLM ships AdamW and four Muon-family optimizers. Switching to Muon means thinking about two learning rates, and avoiding one out-of-the-box failure.
---

`pretrain.yaml` ships `optimizer: adamw`, and AdamW is a defensible choice for every run you will ever do. The reason to read this page is that the Muon family is not a drop-in swap: it introduces a *second* learning rate, and one of its four variants fails on the first step unless you change a model key.

```yaml title="pretrain.yaml"
pretraining:
  optimizer: "adamw"  # adamw | stable_adamw | muon | normuon | muonh | aurora
```

| Value | What it is |
| --- | --- |
| `adamw` | Fused AdamW. The template default. |
| `stable_adamw` | AdamW with update-norm clipping instead of a global gradient clip. |
| `muon` | Muon on 2-D weight matrices, AdamW on everything else. |
| `normuon` | Muon with per-neuron second-moment normalization. |
| `muonh` | Muon with a per-matrix hyperball constraint. **See the caution below.** |
| `aurora` | Hyperball variant with the `muonh_scope` control. |

`adamw` and `stable_adamw` need nothing beyond the base package. The four Muon-family values need
Microsoft's **dion** package, which `nanoplm deps install` already pins into the environment (see
[Install](/nanoplm/start/install/)).

If you skipped the deps installer, or your env is incomplete, the run fails with
`Optimizer 'muon' requires the 'dion' package which is not installed.` Re-run:

```bash
nanoplm deps install
nanoplm deps verify
```

:::caution
Older error text still suggests `pip install nanoplm[cuda]`. **There is no `cuda` extra.** The
supported fix is `nanoplm deps install`, which installs the pinned dion commit along with the rest
of the hard GPU stack.
:::

## Think in two learning rates

This is the concept that makes the Muon family different from swapping one string for another.

Muon is defined on 2-D weight matrices. It has nothing sensible to say about a bias vector, a LayerNorm gain, an embedding table, or an MoE router. So when you select a Muon-family optimizer, nanoPLM builds a **split optimizer**:

| Parameter group | Optimizer | Learning-rate key |
| --- | --- | --- |
| 2-D hidden-layer weight matrices (attention and MLP projections) | Muon / NorMuon / MuonH / Aurora | [`muon_learning_rate`](/nanoplm/reference/pretrain/#pretrain-pretraining-muon_learning_rate) (default `1e-3`) |
| Everything else: 1-D params, embeddings, unembeddings, MoE routers | Fused AdamW | [`adam_learning_rate`](/nanoplm/reference/pretrain/#pretrain-pretraining-adam_learning_rate) (default `1e-4`) |

So a Muon run has **two** peak learning rates, and both are on the same LR schedule: warmup, stable, and decay scale both curves by the same multiplier. Tuning one and leaving the other at its default is the most common way to get a disappointing Muon result.

```yaml title="pretrain.yaml"
pretraining:
  optimizer: "normuon"
  muon_learning_rate: 1e-3   # 2-D matrices: attention + MLP projections
  muon_weight_decay: 0.01
  adam_learning_rate: 1e-4   # 1-D params, embeddings, unembeddings, MoE routers
  adam_weight_decay: 0.0
```

Both show up in the step log, which is how you confirm the split is live:

```
[step 20/39] loss=2.9814 lr=9.74e-05 muon_lr=9.74e-04 grad_norm=0.9127 tok/s=412,880 dt=634.51ms wall=13.2s h100_mfu=38.11%
```

If the model has no 2-D hidden weights to give Muon, or nothing left for the AdamW side, the run stops rather than silently degrading: `No eligible matrix parameters found for Muon (expected 2D hidden-layer weights).`

## Fix muonh before you run it

:::caution[muonh fails out of the box]
`optimizer: muonh` on a stock config raises immediately:

```
optimizer=muonh is incompatible with model.wo_init='zero'. MuonH initializes
each matrix hyperball radius from ||W_init||_F, so zero-initialized Wo matrices
are locked at zero. Set model.wo_init='kaiming' (or --wo-init kaiming).
```

MuonH constrains each matrix to a ball whose radius is seeded from the Frobenius norm of its *initial* weights. `model.wo_init` defaults to `"zero"`, which makes that radius zero for the residual out-projections: they can never move. The fix is one key:

```yaml title="pretrain.yaml"
model:
  wo_init: "kaiming"  # required by muonh; zero-init Wo has a zero hyperball radius
pretraining:
  optimizer: "muonh"
```
:::

The same applies to `aurora`, which is also a hyperball method.

## Scope and normalize the hyperball variants

Two keys apply only to the hyperball optimizers.

| Key | Applies to | Default | Values |
| --- | --- | --- | --- |
| [`muonh_scope`](/nanoplm/reference/pretrain/#pretrain-pretraining-muonh_scope) | `muonh`, `aurora` | `all` | `all` puts every eligible matrix under the hyperball constraint. `mlp` restricts it to MLP matrices and leaves the attention matrices on NorMuon. |
| [`muonh_normalization`](/nanoplm/reference/pretrain/#pretrain-pretraining-muonh_normalization) | `muonh` only | `neuron` | `neuron \| short_axis \| none`. |

## Tune the AdamW knobs

These apply to `adamw` and `stable_adamw`, and to the AdamW side of every Muon-family run.

| Key | Default | When to touch it |
| --- | --- | --- |
| [`adam_learning_rate`](/nanoplm/reference/pretrain/#pretrain-pretraining-adam_learning_rate) | `1e-4` | The peak LR. The first thing you tune on any run. |
| [`adam_weight_decay`](/nanoplm/reference/pretrain/#pretrain-pretraining-adam_weight_decay) | `0.0` | Raise it for long runs on large data; `0.0` is a reasonable default for short ones. |
| [`adam_beta1`](/nanoplm/reference/pretrain/#pretrain-pretraining-adam_beta1) / [`adam_beta2`](/nanoplm/reference/pretrain/#pretrain-pretraining-adam_beta2) | `0.9` / `0.999` | Lower `beta2` (0.95) if loss spikes on large batches. |
| [`adam_epsilon`](/nanoplm/reference/pretrain/#pretrain-pretraining-adam_epsilon) | `1e-8` | Rarely. |
| [`max_grad_norm`](/nanoplm/reference/pretrain/#pretrain-pretraining-max_grad_norm) | `.inf` | `.inf` means gradient clipping is **off**. Set a finite value (e.g. `1.0`) if `grad_norm` in the step log spikes. The run tells you which it is: `Gradient clipping: disabled (max_grad_norm=inf)`. |

The peak LR does not act alone: it is scaled by the schedule. See [Cooldown and resuming](/nanoplm/guides/cooldown-and-resume/) for `lr_schedule`, `warmup_steps`, and `lr_decay_to_fraction`.

## Leave the Muon internals alone

`muon_momentum` (`0.95`), `muon_nesterov` (`true`), `muon_use_polar_express` (`true`), `muon_cautious_weight_decay` (`true`), and `muon_eps` (`1e-7`) are the orthogonalization and momentum internals. The defaults are the published ones and are a fine place to stay. They are listed with their types in the [pretrain.yaml reference](/nanoplm/reference/pretrain/).

## Where to go next

- [Cooldown and resuming](/nanoplm/guides/cooldown-and-resume/): the LR schedule that both learning rates ride on.
- [Scale up](/nanoplm/guides/scale-up/): batch size, sharding, FP8.
- [pretrain.yaml reference](/nanoplm/reference/pretrain/): every optimizer key, with defaults.
