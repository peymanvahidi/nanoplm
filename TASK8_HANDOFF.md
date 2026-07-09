# Session handoff — nanoPLM dependency-install v2, remaining work (Task 8)

You are picking up a nearly-finished effort. This file is the full sync: what exists, what was proven, what remains, and every gotcha discovered along the way. The work was executed on JURECA (an HPC cluster) by a previous session; **you are on Peyman's laptop/workstation, and your job is the Docker build + GHCR push** (the one step JURECA cannot do — it has no Docker). The final on-GPU container check then happens back on JURECA.

## 1. What this branch is

nanoPLM (protein language model training framework, repo `peymanvahidi/nanoplm`, **private**) supports exactly ONE environment: Linux x86_64 · NVIDIA Hopper sm90 · CUDA ≥ 13.0 · torch 2.12.1+cu130, always with the full feature stack (TransformerEngine 2.16 FP8, FlashAttention-3, SonicMoE/QuACK kernels, Dion optimizer). This branch (`deps-install-v2`, 9 commits incl. this file) ships two install paths that are literally one implementation:

- **Path 2 (venv):** `pip install nanoplm` (git URL while private) + `nanoplm deps install` — a click CLI wrapping `src/nanoplm/deps/install.sh`, THE single source of truth for every hard-dep pin, followed by `python -m nanoplm.deps.verify` in a fresh interpreter.
- **Path 1 (container):** `docker/Dockerfile` — single-stage on `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04` — which runs path 2 verbatim at build time. Single-stage is deliberate: sonic-moe/quack JIT-compile CUTLASS kernels at RUNTIME and need `nvcc` inside the running container; a `-runtime` final stage would import fine and then fail at the first MoE step.

Key files: `src/nanoplm/deps/install.sh` (pins/preflight/idempotent steps), `src/nanoplm/deps/verify.py`, `src/nanoplm/cli/deps.py`, `docker/Dockerfile`, `.dockerignore`, tests under `tests/deps/`, `tests/test_packaging.py`, `tests/test_dockerfile.py` (45 tests, all green, no GPU needed).

## 2. What was already done and proven (do NOT redo)

Executed via subagent-driven development (implementer + 2 independent reviews per task + 2-reviewer final whole-branch review; everything approved):

- **Tasks 1–5, 7:** installer script, verify+CLI, packaging (`torch==2.12.1` pin; `install.sh` ships as package-data — wheel content verified), FA3 sha256 pin, README rewrite (+ `install_te_jureca.sh` deleted), Dockerfile/.dockerignore.
- **FA3 wheel mirror is LIVE:** GitHub release tag `deps-assets` on `peymanvahidi/nanoplm` carries `flash_attn_3-3.0.0+20260519.cu130torch2120cxx11abitrue.8a8b2f-cp39-abi3-linux_x86_64.whl` (401,274,925 bytes, sha256 `177359694998fee12e293d15d3398482fc0193f5f40d4aca3ceb257105d5bbf2` — pinned in install.sh). Round-trip verified through install.sh's own `fa3_download` (authenticated API path).
- **Task 6, Phase-1 acceptance on JURECA — ALL gates passed:**
  - Real install into a fresh venv: rc=0 in 8m50s (torch repaired to +cu130, TE v2.16 source-built, FA3 sha256 OK, quack[cu13]+sonic-moe pinned; verify all ✓).
  - Idempotent re-run: 21.4 s, all six components "present; skipping.", rc=0.
  - Negative preflight: exact `module load CUDA/13 NCCL cuDNN Ninja` hint, rc=1.
  - H100 node: `nanoplm deps verify` all ✓ incl. `GPU (Hopper sm90) NVIDIA H100 (sm90)` with **no modules loaded** (torch's bundled nvidia-* pip libs suffice at runtime); FP8 smoke `GPU smoke OK: (8, 128, 256) NVIDIA H100`.
- **Acceptance-driven fixes already on the branch** (commit `d7e3c24`): torch-step satisfaction is runtime-based (`torch.__version__ == "2.12.1+cu130"`) because torch 2.12 dist metadata is flavor-blind (plain `2.12.1` for every build — never pin-check torch via importlib.metadata); the quack step requires the `nvidia-cutlass-dsl-libs-cu13` payload dist to be present, because dion dep-pulls plain `quack-kernels` and would otherwise mask the `[cu13]` extra forever.
- **Task-8 de-risk already done:** `pip install --dry-run ".[eval]" --extra-index-url https://download.pytorch.org/whl/cu130` resolves cleanly (rc=0) — biotrainer==1.4.0 co-resolves with the torch==2.12.1 pin, so the image's first pip step will not conflict.

## 3. Your job — build and push the image (plan "Task 8")

Prereqs on this machine: Docker with BuildKit, ~40 GB free disk, no GPU needed. Two tokens:
- `GITHUB_TOKEN` — fine-grained PAT, repo `peymanvahidi/nanoplm`, **Contents: Read** is enough (used at build time to download the FA3 release asset while the repo is private).
- GHCR push — historically ghcr.io needs a **classic** PAT with `write:packages` (fine-grained PATs often don't work for ghcr login; verify at push time).

```bash
git clone git@github.com:peymanvahidi/nanoplm.git && cd nanoplm && git checkout deps-install-v2
export GITHUB_TOKEN=<repo-read PAT>       # do not echo/commit it
docker build -f docker/Dockerfile \
  --secret id=github_token,env=GITHUB_TOKEN \
  --build-arg NVTE_CUDA_ARCHS=90 --build-arg MAX_JOBS=8 \
  -t ghcr.io/peymanvahidi/nanoplm:0.2.0 -t ghcr.io/peymanvahidi/nanoplm:latest .
```

Expected build behavior:
- The `pip install ".[eval]"` layer resolves torch 2.12.1 via the cu130 extra-index (pre-verified, see §2).
- The `nanoplm deps install --arch 90` layer: preflight passes with no GPU (no `nvidia-smi` in the build → the Hopper gate deliberately no-ops; `CUDA_HOME=/usr/local/cuda` from ENV); the torch step should log "present; skipping." (runtime check); TE compiles from source — **this layer dominates, ~30–60 min at MAX_JOBS=8** (raise MAX_JOBS if the machine has cores/RAM to spare; TE needs ~2 GB RAM per job); FA3 downloads via the mounted secret (anonymous download 404s while the repo is private — that's expected, the script falls through to the authenticated API path).
- The `python -m nanoplm.deps.verify` gate layer: rows may show "installed (import needs a GPU driver — re-run verify on a GPU node)" — that degraded note is EXPECTED without a driver; the build must still end `verify: OK`. If a row fails with a message that is NOT driver-related, that's a real finding — stop and report rather than patching around it.
- `--mount=type=secret,...,env=GITHUB_TOKEN` needs the dockerfile frontend the file already pins (`# syntax=docker/dockerfile:1.10`); the token never lands in a layer.
- If no build secret is provided, the FA3 step falls back to the public upstream wheel (still sha256-gated), so the build can succeed without the secret; passing the secret keeps the private mirror as primary.

Push:
```bash
echo <write:packages PAT> | docker login ghcr.io -u peymanvahidi --password-stdin
docker push ghcr.io/peymanvahidi/nanoplm --all-tags
```
Record `docker images --digests` output for the PR evidence.

## 4. After the push — final acceptance happens on JURECA (not this machine)

Hand these back to Peyman / a JURECA session (plan Task 8 Steps 3–4):
```bash
cd /p/scratch/hai_1116/vahidi1    # .sif is multi-GB; scratch, not $HOME
apptainer registry login --username peymanvahidi docker://ghcr.io   # read PAT
apptainer pull nanoplm-0.2.0.sif docker://ghcr.io/peymanvahidi/nanoplm:0.2.0
srun -A hai_1116 -p dc-hwai --gres=gpu:1 --time=00:20:00 --pty \
  apptainer exec --nv /p/scratch/hai_1116/vahidi1/nanoplm-0.2.0.sif bash -lc '
    nanoplm deps verify && python - <<PY
import torch, transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling
import flash_attn_interface, sonicmoe, dion, quack
x = torch.randn(8, 128, 256, device="cuda", dtype=torch.bfloat16)
lin = te.Linear(256, 256, params_dtype=torch.bfloat16).cuda()
with te.fp8_autocast(enabled=True, fp8_recipe=DelayedScaling()):
    y = lin(x)
torch.cuda.synchronize()
print("container GPU smoke OK:", tuple(y.shape), torch.cuda.get_device_name())
PY'
```
Expected: verify fully ✓ (incl. Hopper sm90), then `container GPU smoke OK: (8, 128, 256) NVIDIA H100`. (`fp8_autocast` prints a DeprecationWarning — known, harmless; the snippet is plan-verbatim.)

## 5. Hard rules and conventions (carried from the executed plan)

- **All pins live ONLY in `src/nanoplm/deps/install.sh`.** Never duplicate or "fix" a pin elsewhere. The installer is pip-only by design (quack-kernels forbids uv on CUDA 13.x) — never introduce uv into it.
- **`docs/` is gitignored and must NEVER be committed** (no `git add -f`). Same for `.superpowers/` scratch.
- **CLI surface is frozen:** `nanoplm deps install [--arch 90] [--force]` and `nanoplm deps verify`. No new flags, no per-dep selection, no module auto-loading.
- **No GitHub-Actions image workflow** — image builds are manual by explicit decision.
- Conventional commit messages; do not merge to `master` — Peyman integrates via his own branches. A PR should eventually go up from `deps-install-v2` with the Task 6 evidence (in §2) + your Task 8 evidence; this handoff file should be `git rm`-ed from the branch before that PR is merged.
- If the build surfaces a real defect (not a driver-degradation note), report it with the layer log rather than working around it — the branch's whole value is that the container equals the proven venv path.

## 6. Known non-blocking follow-ups (only if Peyman asks)

Agreed in the final review as SHOULD-FIX-LATER, none block the image: test for hopper_gate's no-nvidia-smi branch; `--arch=<val>` equals-form and `TE_TAG`/`NANOPLM_FA3_WHEEL` env-knob tests; CLI branch "install succeeds but verify fails" test; static Dockerfile assertions for the secret-mount line and `CMD ["bash"]`; distinct behavior for the `smi_fails` test fake; stale `_fake_bin` docstring; optionally tighten verify.py's `_DRIVER_HINTS` substring heuristic and cross-link the torch pin between install.sh and verify.py. Accepted quirks (do not "fix"): `--arch` with a missing value exits 1 not 2 (unreachable via the CLI); `step_te` runs `mktemp` for real in dry-run; the `git grep install_te_jureca` cleanliness check ignores the guard test's own string.

## 7. State ledger (for reference)

- Branch `deps-install-v2` = 8 work commits `1cfa1be..b91b431` + this handoff commit, pushed to origin. `master` untouched at `7aae9a6`.
- On JURECA: worktree `/p/project1/hai_1116/peyman/nanoplm-dep/nanoplm-deps-v2` holds the branch + a fully-installed acceptance venv (H100-verified); FA3 wheel local copy at `/p/scratch/hai_1116/vahidi1/nanoplm-deps-assets/`; full progress ledger at `<worktree>/.superpowers/sdd/progress.md`, PR draft at `<main tree>/.superpowers/sdd/pr-draft.md`.
- nanoplm version in pyproject: 0.2.0 → image tags `0.2.0` + `latest`.
