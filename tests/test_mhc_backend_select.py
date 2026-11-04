"""MHCLiteBlock._backend selection on CPU: the CuTe kernels are only picked when
the DSL can build them, and anything else routes to the v2 Triton kernels."""
import pytest
import torch
import torch.nn as nn

from nanoplm.pretraining.models.modern_bert import modeling as mod


@pytest.fixture
def sm90(monkeypatch):
    monkeypatch.delenv("NANOPLM_MHC_BACKEND", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    mod._cute_available.cache_clear()
    yield monkeypatch
    monkeypatch.undo()
    mod._cute_available.cache_clear()


def _select(C=1024, T=16):
    blk = mod.MHCLiteBlock(4, C, nn.Identity(), triton_fused=True)
    return blk._backend(torch.zeros(T, 4, C, dtype=torch.bfloat16))


def test_cute_needs_map_dsmem_ptr(sm90):
    cute = pytest.importorskip("cutlass.cute")
    pytest.importorskip("tvm_ffi")
    # positive control: with the DSL >= 4.5 attribute present, C=1024 on sm90 picks cute
    sm90.setattr(cute.arch, "map_dsmem_ptr", object(), raising=False)
    mod._cute_available.cache_clear()
    assert _select() == "cute"
    # an older DSL (4.4.x) lacks it: fall back to v2
    sm90.delattr(cute.arch, "map_dsmem_ptr", raising=False)
    mod._cute_available.cache_clear()
    assert not mod._cute_available()
    assert _select() == "v2"


def test_cute_unavailable_without_dsl(sm90):
    sm90.setattr(mod, "_cute_available", lambda: False)
    assert _select() == "v2"


@pytest.fixture
def forced_cute(sm90):
    from nanoplm.utils.logger import logger

    warned = []
    sm90.setenv("NANOPLM_MHC_BACKEND", "cute")
    sm90.setattr(logger, "warning", lambda msg, *a, **k: warned.append(msg))
    mod._warn_cute_fallback.cache_clear()
    yield sm90, warned
    mod._warn_cute_fallback.cache_clear()


@pytest.mark.parametrize("case", ["C=384", "sm100", "no-dsl"])
def test_forced_cute_falls_back_to_v2(forced_cute, case):
    mp, warned = forced_cute
    mp.setattr(mod, "_cute_available", lambda: case != "no-dsl")
    if case == "sm100":
        mp.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    C = 384 if case == "C=384" else 1024
    assert _select(C) == "v2"
    assert _select(C) == "v2"
    assert len(warned) == 1 and "NANOPLM_MHC_BACKEND=cute" in warned[0]


def test_forced_cute_where_it_runs(forced_cute):
    mp, warned = forced_cute
    mp.setattr(mod, "_cute_available", lambda: True)
    assert _select() == "cute" and not warned


@pytest.mark.parametrize("forced", ["v2", "pytorch"])
def test_forced_other_backends(sm90, forced):
    sm90.setenv("NANOPLM_MHC_BACKEND", forced)
    assert _select() == forced
