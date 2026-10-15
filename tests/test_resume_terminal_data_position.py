"""End-to-end CPU check that `resume.mode: continue` from a warmup_stable
terminal checkpoint (checkpoint-stable-end) trains on exactly the samples an
uninterrupted run would have seen.

Each training phase runs in a subprocess (resume_harness.py) on CPU/gloo with
torch.compile disabled. The oracle is the per-rank sequence of
(global_step, epoch, micro_step, sub_batch, original token rows) entering the
training step: run(N) + resume(N -> N+M) must equal run(N+M).

"dist" mode (DDP/gloo, DistributedSampler seeded by seed+epoch) is the oracle
mode. Production GPU runs are always distributed (single-GPU bf16 is promoted
to a 1-rank FSDP group), so this is the path that matters. It also reads
model_config after the DDP wrap, which these runs cover.

Point NANOPLM_SRC at the src/ tree under test (default: this checkout's src).
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "resume_harness.py"
SRC = os.environ.get("NANOPLM_SRC", str(HERE.parent / "src"))
PY = sys.executable
TORCHRUN = str(Path(PY).parent / "torchrun")

pytestmark = pytest.mark.slow

_SCENARIOS = {
    # 5 micro-batches/pass, 2 steps/pass + 1 trailing discarded micro-batch;
    # N=3 ends at micro 2 of pass 2.
    "mid_pass": dict(n=10, micro=2, ga=2, N=3, M=3),
    # 4 micro-batches/pass; N=2 ends exactly on the pass boundary.
    "pass_boundary": dict(n=8, micro=2, ga=2, N=2, M=3),
    # ga=3 over 5 micro-batches: one step per pass, then a 2-micro partial
    # window discarded. N=1 resumes inside the discarded tail.
    "partial_window_tail": dict(n=10, micro=2, ga=3, N=1, M=3),
    # Batch-size warmup: steps 0-2 split each micro-batch 4 ways with ga=1
    # (3-5: 2 ways). N=2 lands at sub_batch 2/4 of the first micro-batch.
    "split_divisor": dict(n=32, micro=8, ga=2, bwarm=9, N=2, M=6),
}


def _run(tmp: Path, name: str, spec: dict, nproc: int) -> list[dict]:
    spec = dict(spec, out=str(tmp / name))
    spec_path = tmp / f"{name}.spec.json"
    spec_path.write_text(json.dumps(spec))
    if spec["mode"] == "dist":
        cmd = [TORCHRUN, "--standalone", f"--nproc_per_node={nproc}", str(HARNESS), str(spec_path)]
    else:
        cmd = [PY, str(HARNESS), str(spec_path)]
    env = dict(os.environ, PYTHONPATH=SRC, CUDA_VISIBLE_DEVICES="", WANDB_MODE="disabled",
               OMP_NUM_THREADS="1")
    r = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=900)
    (tmp / f"{name}.log").write_text(r.stdout + r.stderr)
    assert r.returncode == 0, f"{name} failed:\n{(r.stdout + r.stderr)[-4000:]}"
    return [json.loads((tmp / f"{name}.rank{k}.json").read_text()) for k in range(nproc)]


def _dataset(tmp: Path, n: int) -> Path:
    d = tmp / "ds"
    code = ("import sys; sys.path.insert(0, %r); import resume_harness as h; "
            "from pathlib import Path; h.build_dataset(Path(%r), %d)") % (str(HERE), str(d), n)
    env = dict(os.environ, PYTHONPATH=SRC, CUDA_VISIBLE_DEVICES="")
    subprocess.run([PY, "-c", code], cwd=tmp, env=env, check=True, capture_output=True)
    return d


def _ckpt(ckp_dir: Path, role: str = "stable-end") -> Path:
    hits = sorted(ckp_dir.glob(f"*/checkpoint-{role}"))
    assert len(hits) == 1, hits
    return hits[0]


def _state(ckpt: Path) -> dict:
    return json.loads((ckpt / "training_state.json").read_text())


def _setup(tmp, sc, nproc, mode):
    ds = _dataset(tmp, sc["n"] * nproc)
    base = dict(dataset_dir=str(ds), micro=sc["micro"], ga=sc["ga"],
                bwarm=sc.get("bwarm", 0), mode=mode)
    return base


@pytest.mark.parametrize("nproc", [1, 2])
@pytest.mark.parametrize("scenario", list(_SCENARIOS))
def test_continue_from_terminal_matches_uninterrupted(tmp_path, scenario, nproc):
    sc = _SCENARIOS[scenario]
    N, M = sc["N"], sc["M"]
    base = _setup(tmp_path, sc, nproc, "dist")

    full = _run(tmp_path, "full", dict(base, ckp_dir=str(tmp_path / "full"), steps=N + M), nproc)
    a = _run(tmp_path, "a", dict(base, ckp_dir=str(tmp_path / "split"), steps=N), nproc)
    trunk = _ckpt(tmp_path / "split")
    st = _state(trunk)
    k = st["global_step"]
    assert k == N
    saved = tuple(st.get(f) for f in ("epoch_data_step", "epoch_sub_batch_step",
                                       "batch_split_divisor"))

    # Keep the trunk read-only: the resume writes its own stable-end into the
    # same run root, so resume from a copy.
    src = tmp_path / "trunk_copy" / trunk.parent.name / trunk.name
    shutil.copytree(trunk, src)
    b = _run(tmp_path, "b", dict(base, ckp_dir=str(tmp_path / "split"), steps=N + M,
                                 resume=str(src)), nproc)

    line = b[0]["resume_lines"][0]
    for r in range(nproc):
        got = a[r]["log"] + b[r]["log"]
        want = full[r]["log"]
        first = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y),
                     min(len(got), len(want)))
        assert got == want, (
            f"rank {r}: resumed sample stream diverges from uninterrupted run "
            f"(saved position {saved}; {line}); first diff at record {first}: "
            f"got {got[first][:4] if first < len(got) else None} "
            f"want {want[first][:4] if first < len(want) else None}; "
            f"len got={len(got)} want={len(want)}"
        )
    assert None not in saved, "terminal checkpoint has no data position"
    assert f"start_step={k}," in line, line
    # Fast-forward target is the saved position, or the top of the next pass
    # when the save landed exactly on the pass end.
    exact = f"data_step={saved[0]} sub_batch={saved[1]} divisor={saved[2]}"
    next_pass = f"epoch={st['epoch'] + 1} data_step=0 sub_batch=0 divisor=1"
    assert exact in line or next_pass in line, line
    assert _state(_ckpt(tmp_path / "split"))["global_step"] == N + M


def test_terminal_inside_split_micro_batch_stops_at_budget(tmp_path):
    """A terminal step inside a split micro-batch must not keep stepping to the
    end of that micro-batch."""
    sc = _SCENARIOS["split_divisor"]
    base = _setup(tmp_path, sc, 1, "dist")
    _run(tmp_path, "a", dict(base, ckp_dir=str(tmp_path / "split"), steps=sc["N"]), 1)
    assert _state(_ckpt(tmp_path / "split"))["global_step"] == sc["N"]


def test_position_less_checkpoint_replays_pass_from_top(tmp_path):
    """Pre-PR terminal checkpoints carry no epoch_data_step: the documented
    fallback replays the current pass from its first micro-batch."""
    sc = _SCENARIOS["mid_pass"]
    base = _setup(tmp_path, sc, 1, "dist")
    a = _run(tmp_path, "a", dict(base, ckp_dir=str(tmp_path / "split"), steps=sc["N"]), 1)
    trunk = _ckpt(tmp_path / "split")
    st = _state(trunk)
    for k in ("epoch_data_step", "epoch_sub_batch_step", "batch_split_divisor"):
        st.pop(k, None)
    (trunk / "training_state.json").write_text(json.dumps(st))
    b = _run(tmp_path, "b", dict(base, ckp_dir=str(tmp_path / "split"),
                                 steps=sc["N"] + sc["M"], resume=str(trunk)), 1)
    first = b[0]["log"][0]
    ep = st["epoch"]
    assert first[1:4] == [ep, 0, 0], first[:4]
    seen_this_pass = [x[4] for x in a[0]["log"] if x[1] == ep]
    replayed = [x[4] for x in b[0]["log"][: len(seen_this_pass)]]
    assert replayed == seen_this_pass


@pytest.mark.parametrize("scenario", ["mid_pass", "pass_boundary"])
def test_zero_step_resume_keeps_data_position(tmp_path, scenario):
    """Resume `continue` at an already-met budget runs no step; the post-loop
    terminal save rewrites checkpoint-stable-end in place. It must not drop the
    data position the original save recorded. On a pass boundary the rewrite
    stores the normalised (epoch + 1, data_step 0), so compare where each
    checkpoint resumes rather than the raw fields."""
    sc = _SCENARIOS[scenario]
    base = _setup(tmp_path, sc, 1, "dist")
    _run(tmp_path, "a", dict(base, ckp_dir=str(tmp_path / "split"), steps=sc["N"]), 1)
    trunk = _ckpt(tmp_path / "split")
    before = _state(trunk)
    assert "epoch_data_step" in before
    noop = dict(base, ckp_dir=str(tmp_path / "split"), steps=sc["N"], resume=str(trunk))
    first = _run(tmp_path, "noop1", noop, 1)
    after = _state(_ckpt(tmp_path / "split"))
    assert after["global_step"] == before["global_step"]
    for k in ("epoch_data_step", "epoch_sub_batch_step", "batch_split_divisor"):
        assert k in after, (k, after)
    second = _run(tmp_path, "noop2", noop, 1)
    assert first[0]["resume_lines"] == second[0]["resume_lines"]
    assert first[0]["log"] == second[0]["log"] == []
