"""Verify nanoPLM's hard-dependency environment.

Run as `python -m nanoplm.deps.verify` — a FRESH interpreter, so a torch build
that `nanoplm deps install` just replaced is re-imported cleanly (the CLI
process itself imported the old torch at startup and must not self-check).

Import rules: this module imports only the stdlib at module level; torch and
the hard deps are imported inside the checks. On machines without a GPU driver
(login/build nodes) an installed dep whose import needs libcuda degrades to a
presence check with an explicit note instead of failing.
"""
import importlib
import importlib.util
import sys

PINNED_TORCH = "2.12.1+cu130"
IMPORT_CHECKS = (
    "dion",
    "transformer_engine.pytorch",
    "flash_attn_interface",   # FA3's import name (dist name is flash_attn_3)
    "quack",
    "sonicmoe",               # sonic-moe's import name (NOT sonic_moe)
    "gram_newton_schulz",     # default Muon-family orthogonalizer
)
_DRIVER_HINTS = ("libcuda", "cuda driver", "driver library")


def check_torch():
    try:
        import torch
    except Exception as e:  # noqa: BLE001 — any import failure is a finding
        return False, f"import failed: {e}"
    if torch.__version__ != PINNED_TORCH:
        return False, f"found {torch.__version__}, expected {PINNED_TORCH}"
    return True, torch.__version__


def check_import(mod: str):
    try:
        importlib.import_module(mod)
        return True, "imports cleanly"
    except Exception as e:  # noqa: BLE001 — .so load errors surface as OSError/ImportError
        msg = str(e).lower()
        if any(h in msg for h in _DRIVER_HINTS) and importlib.util.find_spec(mod.split(".", 1)[0]):
            return True, "installed (import needs a GPU driver — re-run verify on a GPU node)"
        return False, str(e)


def check_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            return True, "no GPU visible — device check skipped (re-run on a GPU node)"
        cap = torch.cuda.get_device_capability()
        if cap != (9, 0):
            return False, f"compute capability {cap}; nanoPLM requires Hopper (9, 0)"
        return True, f"{torch.cuda.get_device_name()} (sm{cap[0]}{cap[1]})"
    except Exception as e:  # noqa: BLE001
        return False, f"GPU probe failed: {e}"


def run_checks():
    rows = [(f"torch=={PINNED_TORCH}", *check_torch())]
    rows += [(m, *check_import(m)) for m in IMPORT_CHECKS]
    rows.append(("GPU (Hopper sm90)", *check_gpu()))
    return rows


def main() -> int:
    rows = run_checks()
    width = max(len(name) for name, _, _ in rows)
    ok_all = True
    for name, ok, detail in rows:
        ok_all = ok_all and ok
        print(f"  {'✓' if ok else '✗'} {name:<{width}}  {detail}")
    print("verify:", "OK" if ok_all else "FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
