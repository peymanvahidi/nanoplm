"""MoE layout markers are checked on the raw checkpoint, so the FSDP/HSDP resume
path refuses unmarked (pre-#170) checkpoints too.

``set_model_state_dict(full_state_dict=True)`` fills every key missing from the
checkpoint with the live model's own value before ``MoELayer._load_from_state_dict``
runs, so the hook alone sees "version 2" for an unmarked v1 checkpoint. These
tests drive the real ``_load_checkpoint`` FSDP branch on a one-rank gloo group
with an actually ``fully_shard``-ed model.
"""
import socket

import pytest
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.moe import (
    _EXPERT_LAYOUT_BUFFER,
    _WI_LAYOUT_BUFFER,
    check_moe_checkpoint,
)
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM

V1_MSG = (r"pre-#170 MoE expert layout \(E, in, out\).*no longer supported.*"
          r"commit before PR #170, e\.g\. 7fbb03a\.")


@pytest.fixture
def gloo_mesh():
    from torch.distributed.device_mesh import init_device_mesh

    owns = not dist.is_initialized()
    if owns:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                                rank=0, world_size=1)
    yield init_device_mesh("cpu", (1,))
    if owns:
        dist.destroy_process_group()


def _cfg(act, inter, layers=2):
    # hidden 32; srelu with intermediate == hidden makes both Wi and Wo square,
    # the case the shape check cannot catch.
    return ProtModernBertMLMConfig(
        hidden_size=32, intermediate_size=inter, num_hidden_layers=layers, num_attention_heads=4,
        use_moe=True, moe_num_experts=4, moe_top_k=2, mlp_activation=act,
        use_canon_layers=False, use_repo=False,
    )


def _model(cfg, mesh=None):
    torch.manual_seed(0)
    m = PureProtModernBertMLM(cfg)
    if mesh is not None:
        from torch.distributed.fsdp import fully_shard
        for layer in m.model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(m, mesh=mesh)
    return m


def _v1(sd, drop_gated_marker=False, transpose=True):
    """Strip markers (as a pre-#170 / pre-gated-v2 save would) and store v1 layout."""
    out = {}
    for k, v in sd.items():
        if k.endswith(_EXPERT_LAYOUT_BUFFER) and transpose:
            continue
        if k.endswith(_WI_LAYOUT_BUFFER) and drop_gated_marker:
            continue
        if transpose and (k.endswith("mlp.Wi") or k.endswith("mlp.Wo")) and v.ndim == 3:
            v = v.transpose(-1, -2).contiguous()
        out[k] = v.clone()
    return out


def _save(tmp_path, sd):
    torch.save(sd, tmp_path / "pytorch_model.bin")
    return tmp_path


CASES = [
    # (activation, intermediate, v1 transform, expected message)
    pytest.param("srelu", 32, dict(), V1_MSG, id="srelu-square-v1"),
    pytest.param("swiglu", 48, dict(), V1_MSG, id="swiglu-v1"),
    pytest.param("swiglu", 48, dict(transpose=False, drop_gated_marker=True),
                 "Wi layout mismatch", id="swiglu-gated-v1-concat"),
]


@pytest.mark.parametrize("act,inter,kw,msg", CASES)
def test_fsdp_resume_refuses_unmarked_checkpoint(tmp_path, gloo_mesh, act, inter, kw, msg):
    from nanoplm.pretraining.pure_pipeline import _load_checkpoint

    cfg = _cfg(act, inter)
    ckpt = _save(tmp_path, _v1(_model(cfg).state_dict(), **kw))
    target = _model(cfg, gloo_mesh)
    before = {k: v.full_tensor().clone() for k, v in target.state_dict().items()
              if hasattr(v, "full_tensor")}
    with pytest.raises(RuntimeError, match=msg) as exc:
        _load_checkpoint(target, None, None, ckpt, "cpu",
                         distributed=True, distributed_mode="fsdp")
    assert "size mismatch" not in str(exc.value)
    after = {k: v.full_tensor() for k, v in target.state_dict().items()
             if hasattr(v, "full_tensor")}
    assert all(torch.equal(before[k], after[k]) for k in before), "refused, yet weights changed"


@pytest.mark.parametrize("act,inter", [("srelu", 32), ("swiglu", 48)])
def test_bare_set_model_state_dict_bypasses_hook(gloo_mesh, act, inter):
    """Documents why the raw-dict check is needed: torch fills the marker."""
    cfg = _cfg(act, inter)
    v1 = _v1(_model(cfg).state_dict())
    with pytest.raises(RuntimeError, match=V1_MSG):
        check_moe_checkpoint(_model(cfg), v1)
    if act == "srelu":  # square: loads silently without the check
        target = _model(cfg, gloo_mesh)
        set_model_state_dict(target, dict(v1), options=StateDictOptions(full_state_dict=True))
        wi = next(v for k, v in target.state_dict().items() if k.endswith("mlp.Wi"))
        ref = next(v for k, v in v1.items() if k.endswith("mlp.Wi"))
        assert torch.equal(wi.full_tensor(), ref)


@pytest.mark.parametrize("act,inter", [("srelu", 32), ("swiglu", 48)])
def test_current_checkpoint_passes_and_loads(gloo_mesh, act, inter):
    cfg = _cfg(act, inter)
    sd = _model(cfg).state_dict()
    check_moe_checkpoint(_model(cfg), sd)  # no raise
    target = _model(cfg, gloo_mesh)
    set_model_state_dict(target, dict(sd), options=StateDictOptions(full_state_dict=True))


def test_non_moe_model_unaffected():
    cfg = ProtModernBertMLMConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                                  num_attention_heads=4, use_canon_layers=False, use_repo=False)
    m = PureProtModernBertMLM(cfg)
    check_moe_checkpoint(m, m.state_dict())


def test_eval_loader_refuses_v1(tmp_path):
    from nanoplm.eval.checkpoint import load_checkpoint

    cfg = _cfg("srelu", 32)
    ckpt = _save(tmp_path, _v1(_model(cfg).state_dict()))
    with pytest.raises(RuntimeError, match=V1_MSG):
        load_checkpoint(ckpt, cfg, torch.device("cpu"))


def test_missing_expert_weights_refused():
    """set_model_state_dict would fill an absent Wi/Wo with the live init."""
    cfg = _cfg("srelu", 32)
    sd = {k: v for k, v in _model(cfg).state_dict().items()
          if not (k.endswith("mlp.Wi") or k.endswith("mlp.Wo"))}
    with pytest.raises(RuntimeError, match=r"model\.layers\.1\.mlp have no Wi in the checkpoint"):
        check_moe_checkpoint(_model(cfg), sd)


# ── one message per problem, not per layer ───────────────────────────────────


def test_refusal_is_one_message_naming_all_layers():
    cfg = _cfg("swiglu", 48, layers=5)
    m = _model(cfg)
    v1 = _v1(m.state_dict(), drop_gated_marker=True)
    with pytest.raises(RuntimeError, match=V1_MSG) as exc:
        check_moe_checkpoint(m, v1)
    msg = str(exc.value)
    assert msg.count("pre-#170") == 1
    assert "at model.layers.{1..4}.mlp (_expert_layout_version is missing;" in msg
    # The gate/up marker is gone too, but its converter cannot help in this
    # build: only the pre-#170 message is shown.
    assert "convert_wi_state_dict_v1_to_v2" not in msg
    assert "Wi layout mismatch" not in msg
    # Plain nn.Module.load_state_dict (the per-layer hook) merges the same way.
    with pytest.raises(RuntimeError, match=V1_MSG) as exc:
        m.load_state_dict(v1)
    msg = str(exc.value)
    assert msg.count("pre-#170") == 1
    assert "model.layers.{1..4}.mlp" in msg
    assert "convert_wi_state_dict_v1_to_v2" not in msg


def test_newer_layout_message():
    cfg = _cfg("srelu", 32, layers=3)
    m = _model(cfg)
    sd = m.state_dict()
    for k in sd:
        if k.endswith(_EXPERT_LAYOUT_BUFFER):
            sd[k] = torch.tensor([3])
    with pytest.raises(RuntimeError) as exc:
        check_moe_checkpoint(m, sd)
    msg = str(exc.value)
    assert ("has expert layout version 3 at model.layers.{1..2}.mlp; this build "
            "supports only version 2. It was written by a newer nanoPLM.") in msg
    assert "pre-#170" not in msg


@pytest.mark.parametrize("buf,act", [(_EXPERT_LAYOUT_BUFFER, "srelu"),
                                     (_EXPERT_LAYOUT_BUFFER, "swiglu"),
                                     (_WI_LAYOUT_BUFFER, "swiglu")])
@pytest.mark.parametrize("bad", [torch.tensor([], dtype=torch.int64), "2"],
                         ids=["empty", "not-a-tensor"])
def test_unreadable_marker_is_refused(buf, act, bad):
    m = _model(_cfg(act, 48))
    sd = m.state_dict()
    sd["model.layers.1.mlp." + buf] = bad
    with pytest.raises(RuntimeError, match=f"unreadable {buf} marker at model.layers.1.mlp"):
        check_moe_checkpoint(m, sd)
    if not isinstance(bad, str):  # load_state_dict itself rejects a str value
        with pytest.raises(RuntimeError, match=f"unreadable {buf} marker"):
            _model(_cfg(act, 48)).load_state_dict(sd)


# ── FSDP/HSDP resume: a missing key must not keep its live init ──────────────


@pytest.mark.parametrize("drop", ["model.layers.1.mlp.router.gate.weight",
                                  "model.layers.1.attn.Wqkv.weight",
                                  "model.layers.1.mlp.Wo",
                                  "decoder.bias"])
def test_fsdp_resume_refuses_missing_key(tmp_path, gloo_mesh, drop):
    from nanoplm.pretraining.pure_pipeline import _load_checkpoint

    cfg = _cfg("swiglu", 48)
    sd = _model(cfg).state_dict()
    del sd[drop]
    ckpt = _save(tmp_path, sd)
    with pytest.raises(RuntimeError, match=f"does not match the model: missing 1 key\\(s\\): {drop}\\."):
        _load_checkpoint(_model(cfg, gloo_mesh), None, None, ckpt, "cpu",
                         distributed=True, distributed_mode="fsdp")


def test_fsdp_resume_refuses_unexpected_key(tmp_path, gloo_mesh):
    from nanoplm.pretraining.pure_pipeline import _load_checkpoint

    cfg = _cfg("swiglu", 48)
    sd = _model(cfg).state_dict()
    sd["model.layers.1.mlp.extra"] = torch.zeros(1)
    ckpt = _save(tmp_path, sd)
    with pytest.raises(RuntimeError, match=r"unexpected 1 key\(s\): model\.layers\.1\.mlp\.extra"):
        _load_checkpoint(_model(cfg, gloo_mesh), None, None, ckpt, "cpu",
                         distributed=True, distributed_mode="fsdp")


@pytest.mark.parametrize("act,inter", [("srelu", 32), ("swiglu", 48)])
def test_fsdp_resume_loads_complete_checkpoint(tmp_path, gloo_mesh, act, inter):
    """Tied decoder/embedding weights and markers included: the key check passes
    and every weight is restored (the optimizer file is absent on purpose)."""
    from nanoplm.pretraining.pure_pipeline import _load_checkpoint

    cfg = _cfg(act, inter)
    torch.manual_seed(1)
    src = PureProtModernBertMLM(cfg).state_dict()
    ckpt = _save(tmp_path, src)
    target = _model(cfg, gloo_mesh)
    with pytest.raises(FileNotFoundError, match="optimizer.pt"):
        _load_checkpoint(target, None, None, ckpt, "cpu",
                         distributed=True, distributed_mode="fsdp")
    got = target.state_dict()
    assert set(got) == set(src)
    for k, v in got.items():
        v = v.full_tensor() if hasattr(v, "full_tensor") else v
        assert torch.equal(v, src[k]), k
