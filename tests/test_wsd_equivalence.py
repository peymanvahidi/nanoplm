"""Headline WSD equivalence invariant (design spec section 7).

Proves the core invariant of the token-budget + Warmup-Stable-Decay (WSD) feature:
annealing a stable trunk *in one run* produces the same final weights as saving
the trunk and resuming later with ``mode="decay"``.

    Run A  : lr_schedule="wsd" with stable_tokens=Ts, decay_tokens=Td  (one process)
             -> checkpoint-final/
    Run B1 : lr_schedule="warmup_stable" with train_tokens=Ts          (the trunk)
             -> checkpoint-stable-end/
    Run B2 : resume mode="decay" from B1's checkpoint-stable-end,
             decay_tokens=Td                                           (anneal the trunk)
             -> checkpoint-final/

    assert  Run A's checkpoint-final  ==  Run B2's checkpoint-final  (within atol)

Determinism notes (so the assertion is meaningful)
--------------------------------------------------
* Both runs use the *same* seed, model dims, batch sizing and dataset, so the
  achieved_global_batch_tokens (hence decay_steps) match exactly and Run A's
  stable phase reaches the same weights as Run B1.
* The model is re-seeded *immediately before* construction in ``_fresh_model`` so
  both runs start from identical random initialization (``run_pure_pretraining``
  only seeds the data/training RNG, not the already-constructed model weights).
* The resume-``decay`` path fast-forwards the dataloader to the trunk's saved
  position, exactly like ``continue``, so Run B2's decay consumes the data Run A's
  decay does and both finish at the same data position. The test runs twice:
  with the stable->decay boundary on a data-pass (epoch) boundary, and mid-pass,
  where a decay that restarted the pass would end somewhere else in the data.
* Final weights are compared only for the pass-aligned boundary. Mid-pass, the
  MLM masking RNG is not restored across the process boundary, which leaves a
  ~2e-4 gap even when both decays see the same sequences (spec section 13: the
  invariant holds "only as well as existing crash-resume reproducibility").

GPU requirement
---------------
``run_pure_pretraining`` always calls ``torch.compile`` and the ModernBert MLM
forward uses a CUDA SDPA / flash-attention path; it cannot run single-process on
CPU. This test is therefore marked ``slow`` + ``gpu`` and skips cleanly when CUDA
is unavailable. To run it manually on a GPU host:

    python -m pytest tests/test_wsd_equivalence.py -v --run-slow

(``--run-slow`` is required: the suite's conftest skips ``slow`` tests by default.)
"""

import random
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.slow, pytest.mark.gpu]


# Tiny, CI-friendly dimensions. The batch math is chosen so that:
#   achieved_global_batch_tokens = micro_batch_seqs * max_seq_len * grad_accum
#   with grad_accum inferred from global_batch_tokens (= 2 here).
_MAX_LEN = 32
_SEED = 123
_MICRO = 2
_GRAD_ACCUM = 2
_GBT = _MICRO * _MAX_LEN * _GRAD_ACCUM          # tokens per optimizer step
# 8 train sequences / (micro=2) = 4 micro-batches/pass / (grad_accum=2)
#   = 2 optimizer steps per data pass.
# decay=2 steps == 1 pass. The stable budget is a test parameter: 4 steps == 2
# full passes (boundary-aligned), 3 steps ends half-way through the second pass.
_STEPS_PER_PASS = 2
_DECAY_STEPS = 2
_DECAY_TOKENS = _GBT * _DECAY_STEPS

_AA = "ACDEFGHIKLMNPQRSTVWY"


def _make_fasta(path: Path, n: int, lo: int, hi: int, seed: int) -> None:
    rng = random.Random(seed)
    records = []
    for i in range(n):
        length = rng.randint(lo, hi)
        seq = "".join(rng.choice(_AA) for _ in range(length))
        records.append(f">seq{i}\n{seq}")
    path.write_text("\n".join(records) + "\n")


def _build_dataset(root: Path, tokenizer):
    """Write a tiny sharded pretrain dataset + manifest into ``root``."""
    from nanoplm.data.sharding.pretrain_shard_writer import ShardWriter
    from nanoplm.data.infra.manifest import PretrainManifest, write_manifest
    from nanoplm.cli.data import (
        _count_pretrain_shard_sequences,
        _sum_pretrain_shard_tokens,
    )

    _make_fasta(root / "train.fasta", n=8, lo=20, hi=30, seed=1)
    _make_fasta(root / "val.fasta", n=4, lo=20, hi=30, seed=2)
    for split in ("train", "val"):
        ShardWriter(
            fasta_path=str(root / f"{split}.fasta"),
            tokenizer=tokenizer,
            max_length=_MAX_LEN,
            output_dir=str(root / split),
            samples_per_shard=10000,
        ).create_shards()

    train_count = _count_pretrain_shard_sequences(root / "train")
    val_count = _count_pretrain_shard_sequences(root / "val")
    train_tokens = _sum_pretrain_shard_tokens(root / "train")
    manifest = PretrainManifest(
        pipeline_mode="pretrain",
        seqs_num=train_count + val_count,
        min_seq_len=20,
        max_seq_len=_MAX_LEN,
        val_ratio=0.1,
        train_dir="train",
        val_dir="val",
        train_sequences=train_count,
        val_sequences=val_count,
        num_register_tokens=0,
        sharded=True,
        samples_per_shard=10000,
        total_train_tokens=train_tokens,
    )
    write_manifest(root, manifest)
    return manifest


def _fresh_model(tokenizer):
    """Build the tiny pure-torch MLM with deterministic, identical init.

    Seed immediately before construction: ``run_pure_pretraining`` seeds the
    data/training RNG but not the (already-built) model weights, so both runs
    must start from the same initialization for the invariant to be testable.
    """
    from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
    from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM

    random.seed(_SEED)
    torch.manual_seed(_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_SEED)
    return PureProtModernBertMLM(
        ProtModernBertMLMConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=tokenizer.vocab_size,
            mlp_dropout=0.0,
            attention_dropout=0.0,
        )
    )


def _base_config(ckp_dir: Path, dataset_dir: Path, stable_steps: int, **overrides):
    from nanoplm.pretraining.config import PretrainingConfig
    from nanoplm.cli.pretrain import _populate_batch_setup

    cfg = PretrainingConfig(
        dataset_dir=str(dataset_dir),
        ckp_dir=str(ckp_dir),
        micro_batch_seqs=_MICRO,
        global_batch_tokens=_GBT,
        warmup_steps=1,
        decay_shape="1-sqrt",
        lr_decay_to_fraction=0.1,
        # Single-process, no flash-attention (SDPA fallback), full fp32 so the
        # comparison is exact and runs on any CUDA device.
        bf16=False,
        tf32=False,
        fp8=False,
        use_packing=False,
        use_static_inp_size=False,
        multi_gpu=False,
        world_size=1,
        num_workers=0,
        # eval fires at the stable boundary in both Run A (forced) and Run B1
        # (terminal at stable_steps); keep save/log out of the way.
        eval_steps=stable_steps,
        save_steps=10_000_000,
        logging_steps=10_000_000,
        seed=_SEED,
        optimizer="adamw",
        **overrides,
    )
    # Populate inferred_grad_accum_steps / global_batch_seqs /
    # achieved_global_batch_tokens (run_pure_pretraining requires these).
    _populate_batch_setup(cfg)
    return cfg


def _find_checkpoint(ckp_dir: Path, role: str) -> Path:
    """Locate the single ``checkpoint-<role>`` under the run's timestamped dir."""
    hits = list(Path(ckp_dir).glob(f"*/checkpoint-{role}"))
    assert len(hits) == 1, f"expected exactly one checkpoint-{role} in {ckp_dir}, got {hits}"
    return hits[0]


def _data_position(checkpoint_dir: Path):
    import json

    state = json.loads((checkpoint_dir / "training_state.json").read_text())
    return {
        k: state.get(k)
        for k in ("epoch", "epoch_data_step", "epoch_sub_batch_step")
    }


def _load_final_weights(checkpoint_dir: Path):
    # _save_checkpoint writes the model weights as pytorch_model.bin (verified
    # against pure_pipeline._save_checkpoint).
    return torch.load(
        checkpoint_dir / "pytorch_model.bin", map_location="cpu", weights_only=True
    )


@pytest.mark.parametrize("stable_steps", [4, 3], ids=["pass-aligned", "mid-pass"])
def test_single_run_wsd_decay_equals_resume_decay(tmp_path, monkeypatch, stable_steps):
    """Run A (single-run wsd) and Run B (warmup_stable trunk + resume decay)
    must produce the same final weights within resume determinism."""
    if not torch.cuda.is_available():
        pytest.skip(
            "WSD equivalence test requires CUDA: run_pure_pretraining always "
            "torch.compiles and the ModernBert MLM forward uses a CUDA "
            "attention path (cannot run single-process on CPU). "
            "Run on a GPU host with: "
            "python -m pytest tests/test_wsd_equivalence.py -v --run-slow"
        )

    # Keep W&B fully offline/quiet and pin to a single visible GPU.
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.chdir(tmp_path)
    # The comparison runs in fp32, which FlashAttention rejects; take the SDPA
    # path even on hosts that have flash_attn_interface installed.
    from nanoplm.pretraining.models.modern_bert import modeling
    monkeypatch.setattr(modeling, "_HAS_FLASH_VARLEN", False)

    from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
    from nanoplm.pretraining.config import ResumeConfig
    from nanoplm.pretraining.pure_pipeline import run_pure_pretraining

    stable_tokens = _GBT * stable_steps
    tokenizer = ProtModernBertTokenizer()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    manifest = _build_dataset(dataset_dir, tokenizer)

    # ---- Run A: single-run WSD (stable -> decay in one process) ----
    run_a_dir = tmp_path / "run_a"
    cfg_a = _base_config(
        run_a_dir,
        dataset_dir,
        stable_steps,
        lr_schedule="wsd",
        stable_tokens=stable_tokens,
        decay_tokens=_DECAY_TOKENS,
    )
    run_pure_pretraining(_fresh_model(tokenizer), cfg_a, manifest)

    # ---- Run B part 1: warmup_stable trunk to the same stable budget ----
    run_b_dir = tmp_path / "run_b"
    cfg_b1 = _base_config(
        run_b_dir,
        dataset_dir,
        stable_steps,
        lr_schedule="warmup_stable",
        train_tokens=stable_tokens,
    )
    run_pure_pretraining(_fresh_model(tokenizer), cfg_b1, manifest)

    # Sanity: Run A's pre-decay trunk must equal Run B1's warmup_stable terminal
    # (both reach checkpoint-stable-end). If this diverges the decay comparison
    # is meaningless, so assert it explicitly.
    a_stable = _load_final_weights(_find_checkpoint(run_a_dir, "stable-end"))
    b_trunk = _load_final_weights(_find_checkpoint(run_b_dir, "stable-end"))
    assert len(a_stable) > 0, "stable-end checkpoint has no weights"
    assert set(a_stable.keys()) == set(b_trunk.keys())
    for k in a_stable:
        assert torch.allclose(a_stable[k], b_trunk[k], atol=1e-5), (
            f"stable trunk weight {k} diverged between wsd and warmup_stable"
        )

    # ---- Run B part 2: resume mode="decay" from the trunk ----
    trunk_ckpt = _find_checkpoint(run_b_dir, "stable-end")
    cfg_b2 = _base_config(
        run_b_dir,
        dataset_dir,
        stable_steps,
        lr_schedule="warmup_stable",  # ignored on decay resume; trunk's config governs
        train_tokens=stable_tokens,
    )
    resume = ResumeConfig(
        is_resume=True,
        checkpoint_dir=str(trunk_ckpt),
        mode="decay",
        decay_tokens=_DECAY_TOKENS,
        decay_shape="1-sqrt",
        lr_decay_to_fraction=0.1,
    )
    run_pure_pretraining(_fresh_model(tokenizer), cfg_b2, manifest, resume_config=resume)

    # ---- Compare the two post-decay checkpoints ----
    # Both decays must end at the same place in the data: the resume continued
    # from the trunk's position instead of restarting the pass.
    assert _data_position(_find_checkpoint(run_b_dir, "final")) == _data_position(
        _find_checkpoint(run_a_dir, "final")
    )
    if stable_steps % _STEPS_PER_PASS:
        return  # mid-pass: weights differ by the masking-RNG gap noted above
    a = _load_final_weights(_find_checkpoint(run_a_dir, "final"))
    b = _load_final_weights(_find_checkpoint(run_b_dir, "final"))
    assert len(a) > 0, "final checkpoint has no weights"
    assert set(a.keys()) == set(b.keys())
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-5), f"weight {k} diverged"
