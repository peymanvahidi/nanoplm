"""Compare a freshly exported model against the native checkpoint.

Export runs on the machine where BOTH models execute -- the FP8 native model
and the dense portable one -- so this check is nearly free and it covers the
failure modes that are otherwise invisible: the TE gate/up ordering and the
sReLU transpose.

This loads the artifact a downstream user actually gets -- the vendored,
portable bundle behind `config.json`'s `auto_map` -- through transformers'
own remote-code loader, not `nanoplm.hf`'s in-process model classes.

A pitfall this module's own review round hit: `wo_init="zero"` is the
DEFAULT (a zero-init-residual scheme; see model.py's `wo_init` and
modeling.py's `_init_modernbert_weights`), which zeroes every attention and
MLP output projection (`attn.Wo`, `mlp.Wo` / `Wo_weight`, and every MoE
expert's `Wo`) at fresh initialization. A model built straight off that
default computes `hidden = final_norm(embeddings)` -- every residual branch
adds exactly zero -- so native and export agree perfectly regardless of
whether the branch itself (TE, Triton srelu, a swapped gate/up, a flipped
transpose) is implemented correctly. The R12-6 measurements taken that way
were real numbers but meaningless: they measured two copies of the SAME
no-op, not the kernel paths this check exists to catch. See `_tolerance_class`
and the constants below for how this was re-measured with LIVE (non-zero)
residual weights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from nanoplm.utils.logger import logger

# Short, long, and non-standard-residue cases. The normalizer maps UZOB -> X,
# so the last sequence also exercises that path.
VERIFY_SEQUENCES: tuple[str, ...] = (
    "MKALCLL",
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKL",
    "MAIGTXBOUZAIGTMKALC",
)

# Measured (R12-7, revised in a round-2 re-review of this comment -- see
# below) with `wo_init="kaiming"` -- i.e. with every residual branch actually
# contributing something, unlike the zero-init default (see the module
# docstring) -- on this node's H100 plus CPU, 2026-09-28, torch 2.12.1+cu130,
# Transformer Engine 2.16.0+4220403. EVERY number in this comment appears in
# one saved log; none is hand-derived or extrapolated between shapes or
# dtypes. Probe script and log:
# .superpowers/sdd/2026-09-09-huggingface-export/task12_scratch/
# probe_live_weights.py, probe_live_weights_round2.log. Relative L2 error
# over valid (attention_mask==1) positions:
#
#   class  | case                       | shape    | device/dtype | rel(logits) | rel(hidden)
#   -------|----------------------------|----------|---------------|-------------|------------
#   EXACT  | dense fp32                 | 64/2     | cpu, cuda     | 0.0         | 0.0
#   EXACT  | dense fp32                 | 768/4    | cpu, cuda     | 0.0         | 0.0
#   EXACT  | dense fp32                 | 1024/16  | cpu, cuda     | 0.0         | 0.0
#   EXACT  | MoE (cutlass) fp32         | 64/2     | cpu           | 0.0         | 0.0
#   EXACT  | MoE (cutlass) fp32         | 64/2     | cuda          | 1.36e-7     | 3.39e-8
#   EXACT  | MoE (cutlass) fp32         | 768/4    | cpu           | 0.0         | 0.0
#   EXACT  | MoE (cutlass) fp32         | 768/4    | cuda          | 5.93e-7     | 3.99e-7
#   EXACT  | recycling fp32 (noise s0)  | 64/3     | cpu, cuda     | 0.0         | 0.0
#   APPROX | srelu (Triton kernel)      | 64/2     | cuda, bf16    | 8.65e-3     | 5.84e-3
#   APPROX | srelu (Triton kernel)      | 768/4    | cuda, bf16    | 1.23e-2     | 9.18e-3
#   APPROX | srelu (Triton kernel)      | 1024/16  | cuda, bf16    | 1.75e-2     | 1.53e-2
#   APPROX | FP8-source                 | 64/2     | cuda, fp32    | 3.62e-4     | 3.13e-4
#   APPROX | FP8-source                 | 64/2     | cuda, bf16    | 7.16e-3     | 4.93e-3
#   APPROX | FP8-source                 | 768/4    | cuda, fp32    | 6.13e-4     | 5.03e-4
#   APPROX | FP8-source                 | 768/4    | cuda, bf16    | 1.17e-2     | 8.38e-3
#   APPROX | FP8-source                 | 1024/16  | cuda, fp32    | 1.04e-3     | 8.62e-4
#   APPROX | FP8-source                 | 1024/16  | cuda, bf16    | 1.77e-2     | 1.50e-2  <- worst correct
#
# R14-5 (2026-09-28, same node/torch/TE as above): `_tolerance_class` was
# fixed to also return "approx" when EITHER side carries a non-fp32
# parameter, not just native's first one -- a bf16/fp16 EXPORT of an
# all-fp32 native model used to be misclassified "exact" (tol=1e-4), and a
# sonicmoe/srelu native compared against a differently-precisioned export
# was only ever checked with native's own dtype driving the decision. These
# are the newly-covered APPROX cases; all correct (no new negative controls
# needed -- this is a classification fix, not a new detector, and APPROX's
# existing negatives above already cover a structural bug at this
# tolerance). Probe script and log:
# .superpowers/sdd/2026-09-09-huggingface-export/task14_scratch/
# probe_r14_5.py, probe_r14_5.log.
#
#   class  | case                                    | shape    | device | rel(logits) | rel(hidden)
#   -------|-----------------------------------------|----------|--------|-------------|------------
#   APPROX | bf16 export of fp32 dense native         | 768/4    | cpu    | 1.11e-2     | 8.33e-3
#   APPROX | bf16 export of fp32 dense native         | 768/4    | cuda   | 1.14e-2     | 8.27e-3
#   APPROX | bf16 export of fp32 dense native         | 1024/16  | cpu    | 1.63e-2     | 1.38e-2
#   APPROX | bf16 export of fp32 dense native         | 1024/16  | cuda   | 1.64e-2     | 1.39e-2  <- worst
#   APPROX | fp16 export of fp32 dense native         | 768/4    | cpu    | 1.44e-3     | 1.04e-3
#   APPROX | fp16 export of fp32 dense native         | 768/4    | cuda   | 1.40e-3     | 1.04e-3
#   APPROX | fp16 export of fp32 dense native         | 1024/16  | cpu    | 1.98e-3     | 1.72e-3
#   APPROX | fp16 export of fp32 dense native         | 1024/16  | cuda   | 2.00e-3     | 1.71e-3
#   APPROX | sonicmoe native (bf16 experts) vs        | 64/2     | cuda   | 1.70e-3     | 1.53e-3
#          | cutlass fp32 export                      |          |        |             |
#   APPROX | sonicmoe native (bf16 experts) vs        | 768/4    | cuda   | 2.64e-3     | 2.38e-3
#          | cutlass fp32 export                      |          |        |             |
#   APPROX | srelu native (bf16, R14-6) vs fp32 export| 768/4    | cuda   | 1.15e-2     | 8.88e-3
#   APPROX | srelu native (bf16, R14-6) vs fp32 export| 1024/16  | cuda   | 1.63e-2     | 1.35e-2
#
# Worst measured across every R14-5 case: 1.64e-2 (bf16 export, 1024/16,
# cuda) -- >=3x under _REL_TOL["approx"] (5.5e-2 / 1.64e-2 = 3.35x) and
# inside the controller's 1.83e-2 acceptance ceiling for this task. No
# tolerance change was needed.
#
# Negative controls, run at EVERY shape/dtype above (round 1 only ran these
# at one or two shapes and the comment cited numbers for shapes it never
# actually measured -- a round-2 finding; every number below is now real):
#
#   NEG head.dense x2                      | 64/2    | cpu        | EXACT  | rel(logits)=8.34e-2, rel(hidden)=0.0
#   NEG FP8 Wi gate/up swapped              | 64/2    | cuda, fp32 | APPROX | rel(logits)=3.19e-1, rel(hidden)=2.91e-1  <- weakest negative
#   NEG FP8 Wi gate/up swapped              | 64/2    | cuda, bf16 | APPROX | rel(logits)=3.19e-1, rel(hidden)=2.91e-1
#   NEG FP8 Wi gate/up swapped              | 768/4   | cuda, fp32 | APPROX | rel(logits)=7.17e-1, rel(hidden)=5.63e-1
#   NEG FP8 Wi gate/up swapped              | 768/4   | cuda, bf16 | APPROX | rel(logits)=7.17e-1, rel(hidden)=5.63e-1
#   NEG FP8 Wi gate/up swapped              | 1024/16 | cuda, fp32 | APPROX | rel(logits)=8.45e-1, rel(hidden)=8.04e-1
#   NEG FP8 Wi gate/up swapped              | 1024/16 | cuda, bf16 | APPROX | rel(logits)=8.46e-1, rel(hidden)=8.04e-1
#   NEG srelu Wo row-flipped                | 64/2    | cuda, bf16 | APPROX | rel(logits)=9.86e-1, rel(hidden)=9.37e-1
#   NEG srelu Wo row-flipped                | 768/4   | cuda, bf16 | APPROX | rel(logits)=1.22e0,  rel(hidden)=1.10e0
#   NEG srelu Wo row-flipped                | 1024/16 | cuda, bf16 | APPROX | rel(logits)=1.24e0,  rel(hidden)=1.18e0
#   NEG one NaN weight                      | 64/2    | cpu        | n/a    | caught by the finiteness check (R12-8)
#     before any tolerance is applied, independent of class.
#
# A structural bug's relative error grows with depth: the SAME FP8 swap
# measures 0.29-0.32 at 64/2 (2 layers) but 0.56-0.72 at 768/4 and 0.80-0.85
# at 1024/16 -- an untrained model's gate and up branches are drawn from the
# same random distribution, so a swap at 2 layers is a much weaker signal
# than the same bug would be in a real trained model of realistic depth.
# EXACT's only observed nonzero source is MoE's GPU grouped-gemm (<6e-7,
# presumably nondeterministic reduction order, not an export bug).
#
# R12-7's criterion, applied per shape/dtype with one constant threshold
# across all of them. For EXACT this is straightforward:
#
#   EXACT:  tol >= 3 * worst_correct (1.36e-7)      = 4.08e-7
#           tol <= weakest_negative (8.34e-2) / 10  = 8.34e-3
#           -> feasible range [4.08e-7, 8.34e-3]. Chosen: 1e-4 (correct
#              margin 735x, negative margin 834x).
#
# For APPROX, the >=10x negative-control margin is required at REALISTIC
# DEPTH (>=768/4) -- ruling R12-12, made after reading the round-2 measured
# table above in full: no real export is a 2-layer toy, and the bug signal
# above grows with depth, so 768/4 and 1024/16 are the shapes that matter.
# At tol = 5.5e-2:
#
#   shape    | worst correct margin | weakest negative margin
#   ---------|-----------------------|-------------------------
#   64/2     | 6.4x  (>=3x)          | 5.3x  (still rejected, below 10x)
#   768/4    | 4.5x  (>=3x)          | 10.2x (>=10x)
#   1024/16  | 3.1x  (>=3x)          | 14.6x (>=10x)
#
# The correct-side margin clears >=3x at every shape (6.4x/4.5x/3.1x). The
# negative-side margin clears the required >=10x at every REALISTIC shape
# (10.2x at 768/4, 14.6x at 1024/16); at the toy 64/2 shape every negative
# control is still correctly rejected, just with a smaller margin (>=5.3x)
# that the ruling accepts as an artifact of testing structural bugs on
# untrained 2-layer weights rather than a gap in real detection power.
# _REL_TOL["approx"] = 5.5e-2 is therefore accepted as-is.
_REL_TOL = {"exact": 1e-4, "approx": 5.5e-2}

# Shared by both forwards below so a recycling model's `torch.randn_like`
# noise draw (modeling.py's recurrent step) comes out identical on both
# sides. Arbitrary; not tuned.
_VERIFY_SEED = 20260909


class ExportVerificationError(RuntimeError):
    """Raised when an exported model disagrees with the native checkpoint."""


@dataclass(frozen=True)
class VerificationResult:
    max_logit_diff: float
    max_hidden_diff: float
    tolerance: float
    passed: bool
    rel_logit_diff: float
    rel_hidden_diff: float


def _tokenizer_special_ids(tokenizer) -> dict:
    """The special-token ids compared between the two tokenizers.

    `cls_token_id` falls back to `bos_token_id` so this covers both a
    CLS-style and a BOS-style scheme; ProtModernBertTokenizer defines neither
    when built with `use_bos_token=False` (the default), in which case both
    sides read None and the comparison passes trivially.
    """
    return {
        "pad_token_id": tokenizer.pad_token_id,
        "mask_token_id": tokenizer.mask_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "cls_token_id": (
            tokenizer.cls_token_id
            if tokenizer.cls_token_id is not None
            else tokenizer.bos_token_id
        ),
    }


def _check_tokenizer(native_tokenizer, exported_tokenizer) -> dict:
    """Tokenizer round-trip (spec Section 5.7 item 4). Returns the native encoding."""
    encode = dict(
        return_tensors="pt", padding=True, truncation=False, add_special_tokens=True
    )
    encoded_native = native_tokenizer(list(VERIFY_SEQUENCES), **encode)
    encoded_exported = exported_tokenizer(list(VERIFY_SEQUENCES), **encode)

    mismatches = []
    if not torch.equal(encoded_native["input_ids"], encoded_exported["input_ids"]):
        mismatches.append("input_ids")
    if not torch.equal(
        encoded_native["attention_mask"], encoded_exported["attention_mask"]
    ):
        mismatches.append("attention_mask")
    native_ids = _tokenizer_special_ids(native_tokenizer)
    exported_ids = _tokenizer_special_ids(exported_tokenizer)
    mismatches += [
        name for name in native_ids if native_ids[name] != exported_ids[name]
    ]
    if mismatches:
        raise ExportVerificationError(
            "export verification failed: tokenizer differs from the native "
            f"tokenizer in {', '.join(mismatches)}"
        )
    return encoded_native


def _any_non_fp32(model: torch.nn.Module) -> bool:
    return any(p.dtype != torch.float32 for p in model.parameters())


def _tolerance_class(
    native_model: torch.nn.Module,
    exported_model: torch.nn.Module,
    device: torch.device,
    fp8_source: bool,
) -> str:
    """ "exact" when native and export both compute the same eager fp32 math;
    "approx" whenever EITHER side exercises something that keeps the two from
    matching bit-for-bit: FP8/TE (fp8_source), a non-fp32 parameter anywhere
    in native (e.g. sonicmoe's bf16 experts) or in the loaded export (a
    bf16/fp16 export -- transformers 5.17 `from_pretrained` defaults to
    `dtype="auto"`, `modeling_utils.py:4099-4100`, so it loads at the saved
    dtype), or the CUDA-only Triton sReLU kernel (native only, gated on device
    + activation; the export always runs sReLU in eager).

    R14-5: the earlier version of this function read only the dtype of
    native's FIRST parameter and never looked at the export at all, so a
    bf16/fp16 export of an all-fp32 native model was wrongly classified
    "exact" (tol=1e-4) -- a bf16-rounded weight comparison routinely exceeds
    that by orders of magnitude, so this only ever failed loudly, but the
    class was still the wrong reason to fail, and a case that happened to be
    close enough would have been accepted at the wrong tolerance.
    """
    mlp_activation = str(getattr(native_model.config, "mlp_activation", "")).lower()
    triton_srelu = mlp_activation == "srelu" and device.type == "cuda"
    if (
        fp8_source
        or triton_srelu
        or _any_non_fp32(native_model)
        or _any_non_fp32(exported_model)
    ):
        return "approx"
    return "exact"


def _relative_l2(native: torch.Tensor, exported: torch.Tensor) -> float:
    """||native - exported||_2 / ||native||_2, over whatever positions the
    caller already sliced down to. Falls back to the absolute diff norm when
    the native tensor is exactly zero (so this never divides by zero and
    never reports NaN/inf on an otherwise well-defined case)."""
    diff_norm = (native - exported).norm().item()
    denom = native.norm().item()
    return diff_norm / denom if denom > 0 else diff_norm


@torch.no_grad()
def verify_exported(
    native_model: torch.nn.Module,
    exported_dir: Union[str, Path],
    *,
    tokenizer,
    device: torch.device,
    fp8_source: bool,
) -> VerificationResult:
    exported_dir = Path(exported_dir)
    config_path = exported_dir / "config.json"
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExportVerificationError(
            f"export verification failed: no config.json in {exported_dir}"
        ) from exc
    if "auto_map" not in raw_config:
        raise ExportVerificationError(
            f"export verification failed: {config_path} has no auto_map; "
            "AutoModelForMaskedLM would not resolve to the vendored bundle"
        )

    # The artifact a downstream user gets, loaded exactly the way they would
    # load it -- never nanoplm.hf's own NanoPLMForMaskedLM. trust_remote_code
    # on BOTH calls: the bundle is already trusted at the model load above,
    # and AutoTokenizer.from_pretrained would otherwise independently hit
    # transformers' resolve_trust_remote_code (config.json carries auto_map)
    # and block on a stdin prompt -- fatal outside a TTY, and up to a 15s
    # SIGALRM-timed hang in one.
    exported = (
        AutoModelForMaskedLM.from_pretrained(
            exported_dir, trust_remote_code=True, local_files_only=True
        )
        .to(device)
        .eval()
    )
    exported_tokenizer = AutoTokenizer.from_pretrained(
        exported_dir, trust_remote_code=True, local_files_only=True
    )

    encoded = _check_tokenizer(tokenizer, exported_tokenizer)

    tol_class = _tolerance_class(native_model, exported, device, fp8_source)
    rel_tol = _REL_TOL[tol_class]

    native_model = native_model.to(device).eval()
    ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)
    keep = mask.bool()

    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # torch.manual_seed() reseeds every CUDA device's global generator, not
    # just `device` -- so every visible CUDA device (not only the target
    # one) must be in fork_rng's list, or a CPU-only verify would still leak
    # a GPU RNG-state change to the caller.
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    try:
        # Every native path computes logits as decoder(head(model_output)),
        # and native_model.model(...) below -- the SDPA path -- is exactly
        # what the export's own model(...) forward computes, so this
        # reproduces the native forward exactly. It also sidesteps a CUDA
        # trap: ModernBertForMaskedLM.forward (i.e. calling native_model(...)
        # directly) takes the flash-attention varlen branch whenever a mask
        # is given on a CUDA tensor, and flash-attn only accepts
        # fp16/bf16/fp8 -- an fp32 checkpoint would crash there.
        # native_model.model(...) never takes that branch, on any device.
        # Note this genuinely runs the native kernels (TE LayerNormMLP /
        # LayerNormLinear / Linear for an FP8 source; the Triton sReLU kernel
        # on CUDA) -- torch.compile is not involved and there is no shortcut
        # that would make native and export compute the same thing by
        # construction, EXCEPT when a residual branch's own weights are zero
        # (see the module docstring).
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(_VERIFY_SEED)
            native_hidden = native_model.model(input_ids=ids, attention_mask=mask)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(_VERIFY_SEED)
            exported_hidden = exported.model(input_ids=ids, attention_mask=mask)

        native_logits = native_model.decoder(native_model.head(native_hidden))
        exported_logits = exported.decoder(exported.head(exported_hidden))

        # Only real (unpadded) positions: a pad position's hidden state is
        # not meaningful and need not match.
        native_logits_kept = native_logits.float()[keep]
        exported_logits_kept = exported_logits.float()[keep]
        native_hidden_kept = native_hidden.float()[keep]
        exported_hidden_kept = exported_hidden.float()[keep]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    # A non-finite output fails outright, independent of tolerance: NaN/inf
    # cannot be compared against a threshold meaningfully (`nan > tol` is
    # False, which would otherwise silently pass a broken export).
    non_finite = []
    if not (
        torch.isfinite(native_logits_kept).all()
        and torch.isfinite(exported_logits_kept).all()
    ):
        non_finite.append("logits")
    if not (
        torch.isfinite(native_hidden_kept).all()
        and torch.isfinite(exported_hidden_kept).all()
    ):
        non_finite.append("last_hidden_state")
    if non_finite:
        logger.info(
            "export verify: non-finite output in %s -> FAIL", ", ".join(non_finite)
        )
        raise ExportVerificationError(
            "export verification failed: "
            + "; ".join(f"{name} (non-finite)" for name in non_finite)
        )

    logit_diff = (native_logits_kept - exported_logits_kept).abs().max().item()
    hidden_diff = (native_hidden_kept - exported_hidden_kept).abs().max().item()
    rel_logit = _relative_l2(native_logits_kept, exported_logits_kept)
    rel_hidden = _relative_l2(native_hidden_kept, exported_hidden_kept)

    failing = []
    if rel_logit > rel_tol:
        failing.append(f"logits (rel_l2={rel_logit:.3e}, max|delta|={logit_diff:.3e})")
    if rel_hidden > rel_tol:
        failing.append(
            f"last_hidden_state (rel_l2={rel_hidden:.3e}, max|delta|={hidden_diff:.3e})"
        )
    passed = not failing

    logger.info(
        "export verify [%s]: rel(logits)=%.3e rel(hidden)=%.3e rel_tol=%.1e "
        "max|dlogits|=%.3e max|dhidden|=%.3e -> %s",
        tol_class,
        rel_logit,
        rel_hidden,
        rel_tol,
        logit_diff,
        hidden_diff,
        "PASS" if passed else "FAIL",
    )
    if failing:
        raise ExportVerificationError(
            "export verification failed: "
            + "; ".join(failing)
            + f", rel_tol={rel_tol:.1e} (class={tol_class}). "
            "For an FP8 source check the TE gate/up ordering in "
            "fp8.te_fc1_to_native_wi / te_fc2_to_native_wo."
        )
    return VerificationResult(
        logit_diff, hidden_diff, rel_tol, passed, rel_logit, rel_hidden
    )
