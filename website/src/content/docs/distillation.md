---
title: Distillation (unmaintained)
description: The knowledge-distillation pipeline still ships in nanoPLM but is unmaintained, has drifted out of step with the tool, and should not be used for new work.
---

:::danger[Do not use this for new work]
The knowledge-distillation pipeline is **unmaintained**. It has not been kept in step with the rest of nanoPLM, it is not tested against current configs, and nobody is fixing it. If you came here to train a protein language model, go to [Pretrain](/nanoplm/pipeline/pretrain/) instead.
:::

This page exists so that a reader who runs `nanoplm --help`, sees a `distill` group, and wonders what it is does not have to guess. It is not a guide.

## It has already drifted

Two concrete symptoms, not hypotheticals:

- **Its templates still reference `pipeline_mode`.** That key was removed from the data pipeline and is now *rejected* on load. The distillation templates never got the memo.
- **`nanoplm eval` cannot read a distillation checkpoint.** Evaluation loads pure-torch pretrained models only. So even a distillation run that completes leaves you with a checkpoint the rest of the tool cannot score.

The pretraining path, by contrast, is what the project is for and what everything else is built around.

## What exists

Two halves, both reachable from the CLI:

**Teacher embeddings.** `nanoplm data get-yaml --distillation` writes a `distillation_data.yaml` whose pipeline terminates in a `kd-embed` stage. That stage (also available standalone as `nanoplm data kd-embed`, with `nanoplm data kd-reshard` to split its output) runs ProtT5 over your FASTA and stores per-residue teacher embeddings as HDF5 shards.

**Student training.** `nanoplm distill get-yaml` writes a `distill.yaml`; `nanoplm distill from-yaml` trains a smaller student model against those stored ProtT5 embeddings. ProtT5 is the only teacher.

That is the whole feature. There is no reference table for `distill.yaml` on this site, and there will not be one.

## If you were going to use it anyway

Prefer [pretraining](/nanoplm/pipeline/pretrain/): it is maintained, it supports the full model and optimizer surface, and its checkpoints work with [`nanoplm eval`](/nanoplm/pipeline/eval/). The [CLI reference](/nanoplm/reference/cli/) lists the distillation commands and their flags at the very bottom, marked as unmaintained.
