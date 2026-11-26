"""Grad/weight norms must not count a shard once per replica.

flush() SUM-reduces each rank's local squared sum over the whole world. Under
HSDP every shard lives on every rank of its replicate group, so before the fix
the logged norms were sqrt(replicate size) too large (5.66x on a 32-node run).
Four CPU ranks on a (2 replicate, 2 shard) mesh cover each layout the tracker
can see: an HSDP shard, a fully sharded DTensor, a replicated DTensor and a
plain tensor. Each logged norm must equal the norm of the full tensor.
"""

import json
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

WORLD = 4


def _layouts():
    from torch.distributed.tensor import Replicate, Shard

    return {
        "hsdp": [Replicate(), Shard(0)],        # FSDP2 HSDP: replicate x shard
        "fsdp2d": [Shard(0), Shard(1)],         # every rank a distinct shard
        "replicated": [Replicate(), Replicate()],
    }


def _worker(rank, store_path, out_path):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor

    from nanoplm.pretraining.debug_metrics import LayerwiseDebugTracker

    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=WORLD
    )
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("replicate", "shard"))
        gen = torch.Generator().manual_seed(0)  # same full tensors on every rank
        model = nn.Module()
        expect = {}
        for name, placements in _layouts().items():
            w = torch.randn(8, 6, generator=gen)
            g = torch.randn(8, 6, generator=gen)
            p = nn.Parameter(distribute_tensor(w, mesh, placements))
            p.grad = distribute_tensor(g, mesh, placements)
            model.register_parameter(name, p)
            expect[name] = (w.norm().item(), g.norm().item())
        w = torch.randn(5, generator=gen)
        g = torch.randn(5, generator=gen)
        p = nn.Parameter(w.clone())
        p.grad = g.clone()
        model.register_parameter("plain", p)
        expect["plain"] = (w.norm().item(), g.norm().item())

        tracker = LayerwiseDebugTracker(
            model=model,
            log_every=1,
            device=torch.device("cpu"),
            output_dir=os.path.dirname(out_path),
            is_main=rank == 0,
            distributed=True,
        )
        tracker.record_step_boundary()
        payload = tracker.flush(1)
        tracker.close()
        if rank == 0:
            got = {}
            for name in expect:
                wk = next(k for k in payload if k.startswith("debug/weight_norm/") and k.endswith("/" + name))
                gk = next(k for k in payload if k.startswith("debug/grad_norm/") and k.endswith("/" + name))
                got[name] = (payload[wk], payload[gk])
            with open(out_path, "w") as f:
                json.dump({"expect": expect, "got": got}, f)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="needs gloo")
def test_norms_count_each_shard_once(tmp_path):
    out = tmp_path / "result.json"
    mp.spawn(_worker, args=(str(tmp_path / "store"), str(out)), nprocs=WORLD, join=True)
    res = json.loads(out.read_text())
    for name, (w_exp, g_exp) in res["expect"].items():
        w_got, g_got = res["got"][name]
        assert w_got == pytest.approx(w_exp, rel=1e-5), (name, "weight", w_got / w_exp)
        assert g_got == pytest.approx(g_exp, rel=1e-5), (name, "grad", g_got / g_exp)
