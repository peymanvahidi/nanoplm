---
title: Publish to Hugging Face
description: nanoplm export turns a checkpoint into a portable Hugging Face repo, verified against the native model before anything is written. Where to run it, what --verify checks, and --dtype trade-offs.
---

`nanoplm export` converts a pretraining checkpoint into a self-contained Hugging Face model repo: config, weights, tokenizer, a generated model card, and the modeling code itself, vendored so a consumer needs only `torch` and `transformers`.

## Export, then push

Export locally first and inspect what was written:

```bash
nanoplm export output/pretraining_checkpoints/<run>/checkpoint-15000
```

```
Wrote output/pretraining_checkpoints/<run>/checkpoint-15000/hf_export
```

`CHECKPOINT` can be a `checkpoint-*` directory, or a `run-*` directory containing one — the newest checkpoint inside is used, same rule as `nanoplm eval run --checkpoint-dir`.

The output directory must not exist, or must be empty; a non-empty directory is refused before anything is loaded. Internally, everything is written into a fresh staging directory next to it, `--verify` (on by default) runs there, and only a passing staging directory is moved into place. If verification fails, or anything else raises, the staging directory is removed and nothing is left at the output path — a failed export never leaves a half-written repo behind. To re-export, remove the old `hf_export` directory first (or pass a different `--out`).

Once you're happy with the local export, push it:

```bash
nanoplm export output/pretraining_checkpoints/<run>/checkpoint-15000 --push-to-hub you/nanoplm-base
```

```
Wrote output/pretraining_checkpoints/<run>/checkpoint-15000/hf_export
Pushed to https://huggingface.co/you/nanoplm-base
```

Remove the previous `hf_export` directory first (or use a different `--out`) — this exports and verifies again, then uploads the freshly published directory. See [Push to the Hub](#push-to-the-hub) for authentication.

## Where to run it

A dense checkpoint — including a cutlass-MoE or a recycling model — exports on any machine; without CUDA it just logs a warning that kernel-backed paths (Triton sReLU, sonicmoe) cannot be exercised, so `--verify` proves less than it would on the training machine.

Two source types are refused outright off-CUDA:

- **FP8 checkpoints** (`fp8: true` in `pretrain_config.yaml`) are stored in Transformer Engine's module layout. Restoring them to dense weights needs Transformer Engine, which needs CUDA — export refuses on CPU, and also refuses on CUDA if Transformer Engine isn't importable.
- **sonicmoe checkpoints** (`model_config.yaml` sets `moe_kernel_backend: sonicmoe`) use a CUDA-only kernel and refuse off-CUDA.

So "export on the training machine" is a hard requirement for FP8 and sonicmoe sources, and a recommendation (for a fully-exercised `--verify`) everywhere else.

## Device support for the exported model

The exported model runs a portable, pure-PyTorch path with no custom kernels — no CUTLASS grouped GEMM, no Transformer Engine, no Triton. This project has verified:

| Device | Status |
| --- | --- |
| CPU | Verified — export load and forward tested in this project. |
| CUDA, Hopper (sm90 / H100) | Verified — export load and forward tested in this project. |
| CUDA, sm70+ (other architectures) | Expected to work (portable pure-PyTorch path, no custom kernels) — not verified. |
| MPS (Apple Silicon) | Expected to work (portable pure-PyTorch path, no custom kernels) — not verified. |

This is the exact table the generated model card ships with each export.

## The model card

Every export writes a `README.md` with:

- **Architecture** and parameter count.
- **Training provenance**, read from `pretrain_config.yaml` and the data manifest: optimizer, LR schedule, token budget (in whichever unit the schedule specifies — tokens or steps, never converted between them), global batch, micro-batch, max sequence length, training sequences and tokens, training steps, and the nanoPLM version that produced the checkpoint. Anything absent from those files prints as `unknown` rather than being guessed.
- **Config-driven notes**, added only when the relevant flag is set: recycling's `output_hidden_states` semantics (one entry per recurrent iteration, not per physical layer, with a note about the seeded noise draw when `recycling_s0_noise_sigma > 0`), mHC-lite's extra hidden-state dimension, and a warning that MoE runs the portable reference kernels (no CUTLASS grouped GEMM) and will be slower than the native training path.
- The **device support** table above.
- A **Quickstart** Python snippet, embedded verbatim, that loads the model, tokenizes a couple of sequences, and mean-pools per-residue embeddings excluding special tokens.

Without `--push-to-hub`, the quickstart's `REPO` line reads `"unknown"` — edit the README by hand, or re-export with `--push-to-hub` before uploading manually.

## What `--verify` checks

`--verify` (the default; disable with `--no-verify`) loads the just-written bundle exactly the way a consumer does — through `transformers`' `AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)`, not nanoPLM's own model classes — and compares it against the native checkpoint:

- **`last_hidden_state` and logits**, forwarded on a handful of fixed protein sequences.
- **Relative L2 error** over the non-padded positions, checked against one of two tolerances:
  - a tight `1e-4` when both sides compute the same eager fp32 math;
  - a looser `5.5e-2` whenever anything keeps the two from matching bit-for-bit: an FP8/Transformer-Engine source, a non-fp32 parameter on either side (a bf16/fp16 export, or a native model with non-fp32 weights such as sonicmoe's bf16 experts), or the CUDA-only Triton sReLU kernel.
- **The tokenizer round-trip**: `input_ids`, `attention_mask`, and the special-token ids (`pad`, `mask`, `eos`, `cls`/`bos`) must match exactly.

A non-finite output (NaN/Inf) fails immediately, before any tolerance check.

If verification fails, the error names which tensor diverged along with its numbers, e.g. `logits (rel_l2=..., max|delta|=...)`, `rel_tol=... (class=...)`. Read the hint it prints — for an FP8 source it points at the TE gate/up remap (`fp8.te_fc1_to_native_wi` / `te_fc2_to_native_wo`). Don't reach for `--no-verify` to make the failure go away: that skips the check, it doesn't fix the export. Re-export after fixing the underlying issue, and report it if you can't explain the mismatch.

## `--dtype`

`--dtype` sets the exported weights' precision; the checkpoint file itself is untouched.

| Value | Notes |
| --- | --- |
| `fp32` *(default)* | Portable, and checked at the exact `1e-4` tolerance when the rest of the comparison is also exact. |
| `bf16` | Halves the download size. Checked at the looser `5.5e-2` tolerance. Poorly supported on sm70 (V100), and only partially supported on MPS. |
| `fp16` | Halves the download size. Checked at the looser tolerance. Values outside fp16's range overflow to non-finite, which `--verify` catches and refuses. |

## A quirk when loading from a local directory

When you load an export straight from a local directory, `transformers` copies the config class and the model class into two separate dynamic-module folders, so `isinstance(model.config, model.config_class)` comes back `False` even though the model loaded and runs correctly. It's cosmetic: nothing about loading or running the model depends on it.

## Consumers

Loading the export needs nothing from this project:

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModel.from_pretrained(REPO, trust_remote_code=True)
```

Both calls need `trust_remote_code=True` — the export's `config.json` carries an `auto_map` pointing at the vendored modeling code, and `transformers` needs that flag to load it. `REPO` can be a Hub id or a local path. No `nanoplm` install is required; the bundle imports only `torch` and `transformers` (and their own hard dependencies).

The [quickstart notebook](https://github.com/peymanvahidi/nanoplm/blob/master/examples/nanoplm_quickstart.ipynb) (resolves once this branch merges) loads an exported model, tokenizes protein sequences, computes mean-pooled per-sequence embeddings, and visualizes a public subcellular-localization dataset with UMAP.

## Push to the Hub

`--push-to-hub REPO_ID` uploads the published export after a successful write (and verification, unless `--no-verify`). It creates the repo if it doesn't already exist; `--private` applies only at creation, so it has no effect on a repo that already exists.

You need to be authenticated first: `hf auth login` (the current `huggingface_hub` CLI command — older installs use `huggingface-cli login`), or set the `HF_TOKEN` environment variable.

## Where to go next

- [CLI reference](/nanoplm/reference/cli/#export): the full flag table with defaults.
