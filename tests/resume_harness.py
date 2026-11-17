"""One training phase for test_resume_terminal_data_position.py and
test_fsdp_epoch_boundary_grad.py, run in its own process.

    python resume_harness.py SPEC.json            # single process (RandomSampler)
    torchrun --nproc_per_node=K resume_harness.py SPEC.json   # gloo DDP, K ranks

SPEC keys: dataset_dir, ckp_dir, out (per-rank JSON path prefix), micro, ga,
steps, bwarm, mode ("single" | "dist"), resume (checkpoint dir or null),
dist_mode (pipeline distributed_mode; default "ddp" for "dist" and "fsdp" for
"single", since the pipeline rejects ddp without multi_gpu), len_overreport
(add this to the train loader length the pipeline syncs; default 0),
eval_steps / logging_steps (default: never), max_wallclock_hours (default unset).

Forces CPU and makes torch.compile the identity. Records, for every training
sub-batch the loop moves to the device, (global_step, epoch, micro_step,
sub_batch_idx, original token rows). Eval batches are not recorded. Also
records the order of _evaluate and _save_checkpoint calls as ("eval"|"save", step).
"""

import json
import logging
import os
import random
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("WANDB_MODE", "disabled")

import torch


def _no_compile(m=None, *a, **k):
    return m if m is not None else (lambda f: f)


torch.compile = _no_compile

_AA = "ACDEFGHIKLMNPQRSTVWY"
MAX_LEN = 32
SEED = 123


def build_dataset(root: Path, n_train: int):
    from nanoplm.data.sharding.pretrain_shard_writer import ShardWriter
    from nanoplm.data.infra.manifest import PretrainManifest, write_manifest
    from nanoplm.cli.data import _count_pretrain_shard_sequences, _sum_pretrain_shard_tokens
    from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer

    tok = ProtModernBertTokenizer()
    root.mkdir(parents=True, exist_ok=True)
    for split, n, seed in (("train", n_train, 1), ("val", 4, 2)):
        rng = random.Random(seed)
        recs = [
            f">s{i}\n" + "".join(rng.choice(_AA) for _ in range(rng.randint(20, 30)))
            for i in range(n)
        ]
        (root / f"{split}.fasta").write_text("\n".join(recs) + "\n")
        ShardWriter(fasta_path=str(root / f"{split}.fasta"), tokenizer=tok,
                    max_length=MAX_LEN, output_dir=str(root / split),
                    samples_per_shard=10000).create_shards()
    tc = _count_pretrain_shard_sequences(root / "train")
    vc = _count_pretrain_shard_sequences(root / "val")
    write_manifest(root, PretrainManifest(
        pipeline_mode="pretrain", seqs_num=tc + vc, min_seq_len=20, max_seq_len=MAX_LEN,
        val_ratio=0.1, train_dir="train", val_dir="val", train_sequences=tc,
        val_sequences=vc, num_register_tokens=0, sharded=True, samples_per_shard=10000,
        total_train_tokens=_sum_pretrain_shard_tokens(root / "train"),
    ))


def _fresh_model():
    from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
    from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
    from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer

    random.seed(SEED)
    torch.manual_seed(SEED)
    return PureProtModernBertMLM(ProtModernBertMLMConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        vocab_size=ProtModernBertTokenizer().vocab_size, mlp_dropout=0.0, attention_dropout=0.0,
    ))


def run_phase(spec: dict) -> None:
    from nanoplm.data.infra.manifest import read_manifest
    from nanoplm.pretraining.config import PretrainingConfig, ResumeConfig
    from nanoplm.cli.pretrain import _populate_batch_setup
    import nanoplm.pretraining.pure_pipeline as pp

    dist_mode = spec["mode"] == "dist"
    world = int(os.environ.get("WORLD_SIZE", "1")) if dist_mode else 1
    rank = int(os.environ.get("RANK", "0"))
    gbt = spec["micro"] * MAX_LEN * spec["ga"] * world
    cfg = PretrainingConfig(
        dataset_dir=spec["dataset_dir"], ckp_dir=spec["ckp_dir"],
        micro_batch_seqs=spec["micro"], global_batch_tokens=gbt, warmup_steps=1,
        lr_schedule="warmup_stable", train_tokens=gbt * spec["steps"],
        global_batch_warmup_steps=spec.get("bwarm", 0),
        bf16=False, tf32=False, fp8=False, use_packing=False, use_static_inp_size=False,
        multi_gpu=dist_mode, world_size=world, num_workers=0,
        eval_steps=spec.get("eval_steps", 10_000_000), save_steps=10_000_000,
        logging_steps=spec.get("logging_steps", 10_000_000),
        max_wallclock_hours=spec.get("max_wallclock_hours"),
        seed=SEED, optimizer="adamw", distributed_mode=spec.get("dist_mode", "ddp" if dist_mode else "fsdp"),
    )
    _populate_batch_setup(cfg)
    if dist_mode:
        pp._bootstrap_single_process_env()
    # The FSDP2 path builds its mesh on "cuda"; on CPU/gloo it must be "cpu".
    _mesh = pp.init_device_mesh
    pp.init_device_mesh = lambda _t, *a, **k: _mesh("cpu", *a, **k)
    over = spec.get("len_overreport", 0)
    if over:  # a loader whose __len__ over-reports, like packing with workers
        _sync = pp._TrainLenSync.__call__
        pp._TrainLenSync.__call__ = lambda self, n: _sync(self, n + over)

    log, msgs = [], []
    orig_move = pp._move_batch_to_device

    def _hook(batch, device):
        f = sys._getframe(1)
        if f.f_code.co_name == "run_pure_pretraining":
            ids, lab = batch["input_ids"], batch.get("labels")
            if lab is not None:  # undo masking: masked ids depend on RNG draws
                ids = torch.where(lab != -100, lab, ids)
            # Reads pure_pipeline's loop locals by name: renaming global_step/epoch/
            # micro_step/sub_batch_idx there breaks this with a KeyError.
            L = f.f_locals
            log.append([L["global_step"], L["epoch"], L["micro_step"],
                        L["sub_batch_idx"], ids.tolist()])
        return orig_move(batch, device)

    pp._move_batch_to_device = _hook

    events = []
    orig_eval, orig_save = pp._evaluate, pp._save_checkpoint

    def _eval_hook(*a, **k):
        events.append(["eval", sys._getframe(1).f_locals["global_step"]])
        return orig_eval(*a, **k)

    def _save_hook(*a, **k):
        events.append(["save", a[3]])  # (model, optimizer, scheduler, global_step, ...)
        return orig_save(*a, **k)

    pp._evaluate, pp._save_checkpoint = _eval_hook, _save_hook

    class _H(logging.Handler):
        def emit(self, record):
            msgs.append(record.getMessage())

    logging.getLogger("nanoplm").addHandler(_H())

    rc = None
    if spec.get("resume"):
        rc = ResumeConfig(is_resume=True, checkpoint_dir=spec["resume"], mode="continue")
    manifest = read_manifest(Path(spec["dataset_dir"]))
    pp.run_pure_pretraining(_fresh_model(), cfg, manifest, resume_config=rc)
    resume_lines = [m for m in msgs if m.startswith("Resume continue:")]
    pass_lines = [m for m in msgs if m.startswith("Data pass ")]  # rank 0 only
    Path(f"{spec['out']}.rank{rank}.json").write_text(json.dumps(
        {"log": log, "resume_lines": resume_lines, "pass_lines": pass_lines,
         "events": events}))


if __name__ == "__main__":
    run_phase(json.loads(Path(sys.argv[1]).read_text()))
