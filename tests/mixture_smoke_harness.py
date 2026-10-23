"""CPU smoke for test_mixture_training.py: run_pure_pretraining on a 2-source mixture
(plain + cluster) with packing, in its own process.

    python mixture_smoke_harness.py DIR

Packing needs flash-attn varlen, which CPU torch lacks: the harness patches it with a
per-segment scaled_dot_product_attention, makes torch.compile the identity and builds
the FSDP mesh on CPU (as resume_harness.py does). Writes DIR/result.json: the log lines
and the keys of every W&B payload that holds an eval key.
"""
import json, logging, os, sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("WANDB_MODE", "disabled")
import torch
import torch.nn.functional as F
torch.compile = lambda m=None, *a, **k: m if m is not None else (lambda f: f)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
from tests.data.datasets.mixture_sources import plain_source, cluster_source
from nanoplm.data.infra.mixture import write_mixture
a = plain_source(root / "a", 120, tag=5, max_len=64)
c, _ = cluster_source(root / "c", seed=3)
write_mixture(root, [("a", a, 0.4, "a.fasta"), ("c", c, 0.6, "c.fasta")])

import nanoplm.pretraining.models.modern_bert.modeling as M
def fake_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                softmax_scale=None, window_size=None, **kw):
    out = torch.empty_like(q)
    cu = cu_seqlens_q.tolist()
    for s, e in zip(cu[:-1], cu[1:]):
        qs, ks, vs = (t[s:e].transpose(0, 1) for t in (q, k, v))
        out[s:e] = F.scaled_dot_product_attention(qs, ks, vs, scale=softmax_scale).transpose(0, 1)
    return out
M._HAS_FLASH_VARLEN = True
M._flash_varlen_fn = fake_varlen

from nanoplm.data.infra.manifest import read_manifest
from nanoplm.pretraining.config import PretrainingConfig
from nanoplm.cli.pretrain import _populate_batch_setup
import nanoplm.pretraining.pure_pipeline as pp
_mesh = pp.init_device_mesh
pp.init_device_mesh = lambda _t, *a, **k: _mesh("cpu", *a, **k)
msgs = []
class H(logging.Handler):
    def emit(self, r): msgs.append(r.getMessage())
logging.getLogger("nanoplm").addHandler(H())
logged = []
import wandb
_init = wandb.init
def _init_and_record(*a, **k):
    # wandb.init rebinds wandb.log to the run's, so wrap it afterwards
    run = _init(*a, **k)
    _log = wandb.log
    def _record(payload, *a, **k):
        if any(key.startswith("eval/") for key in payload):
            logged.append(sorted(payload))
        return _log(payload, *a, **k)
    wandb.log = _record
    return run
wandb.init = _init_and_record
micro, ga, steps = 4, 1, 12
gbt = micro * 128 * ga
cfg = PretrainingConfig(
    dataset_dir=str(root), ckp_dir=str(root / "ckp"), micro_batch_seqs=micro,
    global_batch_tokens=gbt, warmup_steps=1, lr_schedule="warmup_stable",
    train_tokens=gbt * steps, bf16=False, tf32=False, fp8=False, use_packing=True,
    use_static_inp_size=False, multi_gpu=False, world_size=1, num_workers=0,
    eval_steps=6, save_steps=10_000_000, logging_steps=6, seed=7, optimizer="adamw")
_populate_batch_setup(cfg)
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
model = PureProtModernBertMLM(ProtModernBertMLMConfig(
    hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
    vocab_size=ProtModernBertTokenizer().vocab_size, mlp_dropout=0.0, attention_dropout=0.0))
pp.run_pure_pretraining(model, cfg, read_manifest(root))
(root / "result.json").write_text(json.dumps({"log": msgs, "eval_keys": logged}))
