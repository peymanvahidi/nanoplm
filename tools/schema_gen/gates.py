# tools/schema_gen/gates.py
"""Translate recycling_compatibility.yaml into machine rules for the docs/validator,
and declare field-visibility gates the form uses for progressive disclosure."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import yaml
from tools.schema_gen.sources import COMPAT_MATRIX

@lru_cache(maxsize=1)
def compat_rules() -> dict:
    matrix = yaml.safe_load(Path(COMPAT_MATRIX).read_text(encoding="utf-8"))
    mf = matrix["model_flags"]
    pf = matrix["pipeline_flags"]
    return {
        "trigger": matrix["trigger"],
        "feature": matrix.get("feature", ""),
        # `condition` is retained on every list because, per recycling_compatibility.py,
        # it is human-readable doc text that disambiguates inverted-polarity flags
        # (e.g. pure_torch is rejected only "pure_torch is False" — it is REQUIRED).
        "model_reject": [{"flag": e["flag"], "reason": e["reason"], "condition": e.get("condition", "")}
                         for e in mf.get("reject", [])],
        "model_warn": [{"flag": e["flag"], "message": e.get("message", ""),
                        "condition": e.get("condition", "")} for e in mf.get("allow_with_warning", [])],
        "model_allow": list(mf.get("allow", [])),
        "pipeline_reject": [{"flag": e["flag"], "reason": e["reason"], "condition": e.get("condition", "")}
                            for e in pf.get("reject", [])],
        "pipeline_caveat": [{"flag": e["flag"], "message": e.get("message", ""),
                             "condition": e.get("condition", "")} for e in pf.get("allow_with_caveat", [])],
        "pipeline_allow": list(pf.get("allow", [])),
    }

# Field -> the controller that must be set for the field to be relevant (progressive disclosure).
# Grounded in the templates' "(active only when …)" comments and the model gating.
def visibility_gates() -> dict:
    gt1 = {"field": "recycling_loop_count", "op": ">", "value": 1}
    eq_true = lambda f: {"field": f, "op": "==", "value": True}
    g: dict[str, dict] = {}
    # recycling_pretraining_eval_loop_counts is deliberately absent: the in-loop
    # r sweep it configures is read for loopie runs too, and a gate expresses
    # one condition, not "either looping mechanism".
    for f in ("recycling_bptt", "recycling_loop_count_sampling", "recycling_loop_count_sigma",
              "recycling_loop_count_curriculum", "recycling_loop_count_curriculum_steps",
              "recycling_num_prelude_layers", "recycling_num_coda_layers",
              "recycling_adapter_bias", "recycling_s0_noise_sigma",
              "recycling_use_takase_init"):
        g[f] = gt1
    g["loopie_loop_count"] = eq_true("use_loopie")
    g["loopie_loop_count_jitter"] = eq_true("use_loopie")
    for f in ("canon_layers_mode", "canon_layers_kernel_size"):
        g[f] = eq_true("use_canon_layers")
    for f in ("noble_rank", "noble_alpha", "noble_half_kaiming", "noble_targets",
              "noble_base_lr", "noble_lr_gamma", "noble_lr_gamma_M",
              "noble_freq_lr_mult", "noble_phase_lr_mult"):
        g[f] = eq_true("use_noble")
    for f in ("mhc_n_streams", "mhc_lite_wrapping_level", "mhc_triton_fused"):
        g[f] = eq_true("use_mhc_lite")
    g["repo_after_n_layers"] = eq_true("use_repo")
    g["prores_T"] = eq_true("use_prores")
    for f in ("moe_num_experts", "moe_top_k", "moe_use_bias_correction", "moe_aux_loss_coef",
              "moe_z_loss_coef", "moe_routed_scaling_factor", "moe_n_group", "moe_topk_group",
              "moe_bias_update_rate", "moe_leading_dense_layers"):
        g[f] = eq_true("use_moe")
    return g
