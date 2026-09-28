"""The exported bundle must load with only transformers + torch installed.

transformers' remote-code loader copies the entry files plus every sibling
they name in a `from .x import y` line, requires every absolute import in
them to be installed, and copies nothing else. The static tests check the
bundle with the loader's own scanners; the end-to-end tests load a real
export in a fresh interpreter that cannot import nanoplm or any kernel
package.
"""

import ast
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from transformers.dynamic_module_utils import get_imports

import nanoplm.hf.vendor as vendor
from nanoplm.config import recycling_compatibility
from nanoplm.hf import NanoPLMConfig, NanoPLMForMaskedLM
from nanoplm.hf.vendor import (
    AUTO_MAP,
    assert_no_unguarded_optional_imports,
    vendor_modeling_code,
)

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = Path(vendor.__file__).resolve().parents[1]  # src/nanoplm
GLOBALS = "pretraining/models/modern_bert/_globals.py"
MODELING = "pretraining/models/modern_bert/modeling.py"
MATRIX_YAML = "config/recycling_compatibility.yaml"
PORTABLE_ORIGINAL = (
    'PORTABLE: bool = _os_portable.environ.get("NANOPLM_PORTABLE", "0") == "1"'
    "  # vendor: rewritten to True in exported bundles"
)
PORTABLE_REWRITTEN = "PORTABLE: bool = True  # rewritten by nanoplm.hf.vendor"
WITH_CUDA = torch.cuda.is_available()


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A copy of src/nanoplm that the vendor reads instead of the real tree."""
    root = tmp_path / "src" / "nanoplm"
    shutil.copytree(PACKAGE_ROOT, root, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(vendor, "_PACKAGE_ROOT", root)
    return root


def _replace_once(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"expected exactly one {old!r} in {path}"
    path.write_text(text.replace(old, new), encoding="utf-8")


def _append(path, code):
    with path.open("a", encoding="utf-8") as f:
        f.write("\n\n" + code + "\n")


# --- (a) a flat bundle of Python files -----------------------------------------


def test_bundle_is_flat_python_with_the_generated_modules(tmp_path):
    written = vendor_modeling_code(tmp_path)
    assert sorted(written) == sorted(tmp_path.iterdir())
    assert all(p.parent == tmp_path and p.suffix == ".py" for p in written)
    assert {
        "modeling_nanoplm.py",
        "configuration_nanoplm.py",
        "modeling.py",
        "_globals.py",
        "logger.py",
        "_recycling_matrix.py",
    } <= {p.name for p in written}
    # nanoplm.utils.logger creates output/logs/ on first emit; never ship it.
    real_logger = (PACKAGE_ROOT / "utils" / "logger.py").read_text(encoding="utf-8")
    assert "_LazyFileHandler" in real_logger
    assert not any("_LazyFileHandler" in p.read_text(encoding="utf-8") for p in written)


def test_logger_shim_is_a_bare_named_logger(tmp_path):
    vendor_modeling_code(tmp_path)
    module = ast.parse((tmp_path / "logger.py").read_text(encoding="utf-8"))
    docstring = ast.get_docstring(module)
    assert docstring and "\n" not in docstring
    assert [ast.unparse(node) for node in module.body[1:]] == [
        "import logging",
        "logger = logging.getLogger('nanoplm')",
    ]


# --- (b) no file needs nanoplm -------------------------------------------------


def _nanoplm_imports(path):
    hits = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules = [node.module]
        else:
            continue
        hits += [
            f"{path.name}:{node.lineno}: {module}"
            for module in modules
            if module.split(".")[0] == "nanoplm"
        ]
    return hits


def test_no_vendored_file_imports_nanoplm(tmp_path):
    written = vendor_modeling_code(tmp_path)
    assert [p.name for p in written if "nanoplm" in get_imports(p)] == []
    # Also inside try blocks, which get_imports skips.
    assert [hit for p in written for hit in _nanoplm_imports(p)] == []


@pytest.mark.parametrize(
    "code, message",
    [
        ("import nanoplm.utils.common", r"import nanoplm\.utils\.common"),
        (
            "from nanoplm.pretraining.models.modern_bert import rope",
            r"modern_bert does not resolve to a file",
        ),
        ("from nanoplm.utils.logger import set_log_level", r"nanoplm\.utils\.logger"),
        ("def _probe():\n    from .no_such_module import x", r"no_such_module"),
        ("def _probe():\n    from .rope.inner import x", r"rope\.inner"),
    ],
    ids=["import-form", "package", "logger-name", "missing-sibling", "dotted"],
)
def test_unvendorable_imports_raise(tmp_path, tree, code, message):
    _append(tree / MODELING, code)
    with pytest.raises(RuntimeError, match=message):
        vendor_modeling_code(tmp_path / "out")


def test_two_modules_with_one_basename_raise_across_files(tmp_path, tree):
    # modeling_nanoplm.py reaches modern_bert/config.py through modeling.py.
    (tree / "utils" / "config.py").write_text("X = 1\n", encoding="utf-8")
    _append(tree / "hf/modeling_nanoplm.py", "from nanoplm.utils.config import X")
    with pytest.raises(
        RuntimeError, match=r"utils/config\.py.*both would be config\.py"
    ):
        vendor_modeling_code(tmp_path / "out")


def test_two_modules_with_one_basename_raise_in_one_file(tmp_path, tree):
    # Both imports would read `from .helper import`, and no other file pulls
    # in modern_bert/helper.py, so only the importing file can tell them apart.
    for package in ("utils", "pretraining/models/modern_bert"):
        (tree / package / "helper.py").write_text("X = 1\n", encoding="utf-8")
    _append(
        tree / MODELING, "from .helper import X\nfrom nanoplm.utils.helper import X"
    )
    with pytest.raises(RuntimeError, match=r"both would be helper\.py"):
        vendor_modeling_code(tmp_path / "out")


# --- (c) the bundle is always portable -----------------------------------------


def test_vendored_globals_are_portable(tmp_path):
    source = (PACKAGE_ROOT / GLOBALS).read_text(encoding="utf-8").splitlines()
    assert source.count(PORTABLE_ORIGINAL) == 1, "the source anchor moved"
    vendor_modeling_code(tmp_path)
    lines = (tmp_path / "_globals.py").read_text(encoding="utf-8").splitlines()
    assert lines.count(PORTABLE_REWRITTEN) == 1
    assert PORTABLE_ORIGINAL not in lines


@pytest.mark.parametrize(
    "new",
    ["PORTABLE: bool = False", PORTABLE_ORIGINAL + "\n" + PORTABLE_ORIGINAL],
    ids=["missing", "duplicated"],
)
def test_vendoring_needs_the_portable_anchor_exactly_once(tmp_path, tree, new):
    _replace_once(tree / GLOBALS, PORTABLE_ORIGINAL, new)
    with pytest.raises(RuntimeError, match="PORTABLE"):
        vendor_modeling_code(tmp_path / "out")


# --- (d) the import gate -------------------------------------------------------


def test_gate_passes_on_a_clean_bundle(tmp_path):
    vendor_modeling_code(tmp_path)
    assert_no_unguarded_optional_imports(tmp_path)


def test_gate_rejects_top_level_optional_imports(tmp_path):
    vendor_modeling_code(tmp_path)
    (tmp_path / "bad_module.py").write_text("import triton\n", encoding="utf-8")
    (tmp_path / "also_bad.py").write_text(
        "from flash_attn import flash_attn_func\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError) as excinfo:
        assert_no_unguarded_optional_imports(tmp_path)
    for expected in ("bad_module.py", "triton", "also_bad.py", "flash_attn"):
        assert expected in str(excinfo.value)


def test_gate_rejects_an_import_guarded_by_a_cuda_check(tmp_path):
    # get_imports skips only try blocks and `if is_*_available()` tests: the
    # loader requires triton here even on a machine with no GPU.
    vendor_modeling_code(tmp_path)
    (tmp_path / "cuda_guarded.py").write_text(
        "import torch\n\nif torch.cuda.is_available():\n    import triton\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match=r"cuda_guarded\.py.*triton"):
        assert_no_unguarded_optional_imports(tmp_path)


def test_gate_allows_an_import_inside_try(tmp_path):
    vendor_modeling_code(tmp_path)
    (tmp_path / "try_guarded.py").write_text(
        "try:\n    import triton\nexcept ImportError:\n    triton = None\n",
        encoding="utf-8",
    )
    assert_no_unguarded_optional_imports(tmp_path)


def test_gate_rejects_a_dangling_relative_import(tmp_path):
    vendor_modeling_code(tmp_path)
    _append(tmp_path / "modeling.py", "def _probe():\n    from .not_vendored import x")
    with pytest.raises(RuntimeError, match=r"modeling\.py.*not_vendored"):
        assert_no_unguarded_optional_imports(tmp_path)


# --- (e) auto_map --------------------------------------------------------------


def test_auto_map_points_at_the_vendored_classes():
    assert AUTO_MAP == {
        "AutoConfig": "configuration_nanoplm.NanoPLMConfig",
        "AutoModel": "modeling_nanoplm.NanoPLMModel",
        "AutoModelForMaskedLM": "modeling_nanoplm.NanoPLMForMaskedLM",
    }


# --- (f) the recycling matrix travels as Python --------------------------------


def _embedded_matrix(path):
    namespace = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace["MATRIX_YAML"]


def test_embedded_matrix_is_the_yaml_verbatim(tmp_path):
    vendor_modeling_code(tmp_path)
    source = (PACKAGE_ROOT / MATRIX_YAML).read_text(encoding="utf-8")
    embedded = _embedded_matrix(tmp_path / "_recycling_matrix.py")
    assert embedded == source
    assert yaml.safe_load(embedded) == yaml.safe_load(source)


def test_embedded_matrix_survives_quotes_and_backslashes(tmp_path, tree):
    text = 'a: "back\\\\slash \\n"\nb: \'""" and """"\'\nc: ""'
    (tree / MATRIX_YAML).write_text(text, encoding="utf-8")
    vendor_modeling_code(tmp_path / "out")
    assert _embedded_matrix(tmp_path / "out" / "_recycling_matrix.py") == text


def test_matrix_falls_back_to_the_embedded_copy(tmp_path, monkeypatch):
    vendor_modeling_code(tmp_path)
    name = "nanoplm.config._recycling_matrix"
    spec = importlib.util.spec_from_file_location(
        name, tmp_path / "_recycling_matrix.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(recycling_compatibility, "_MATRIX_PATH", tmp_path / "no.yaml")
    recycling_compatibility.load_compatibility_matrix.cache_clear()
    try:
        matrix = recycling_compatibility.load_compatibility_matrix()
    finally:
        recycling_compatibility.load_compatibility_matrix.cache_clear()
    source = (PACKAGE_ROOT / MATRIX_YAML).read_text(encoding="utf-8")
    assert matrix == yaml.safe_load(source)


# --- (g) end to end: load the export without nanoplm ---------------------------

TINY = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_kv_heads=2,
    vocab_size=32,
)
CASES = {
    "dense": {},
    "moe": {
        "use_moe": True,
        "moe_num_experts": 4,
        "moe_top_k": 2,
        "moe_kernel_backend": "cutlass",
    },
    "recycling": {
        "num_hidden_layers": 3,
        "use_huginn_looping": True,
        "recycling_loop_count": 2,
        "recycling_num_prelude_layers": 1,
        "recycling_num_coda_layers": 1,
        # The default s_0 noise would make even the eval forward random.
        "recycling_s0_noise_sigma": 0.0,
    },
}
# Two masked positions (<mask> = 3) and one padding position (<pad> = 0).
IDS = torch.tensor([[4, 5, 3, 7, 8, 1], [9, 3, 11, 12, 1, 0]])
MASK = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]])
BLOCKED = (
    "nanoplm",
    "triton",
    "flash_attn",
    "flash_attn_interface",
    "kernels",
    "transformer_engine",
    "quack",
    "sonicmoe",
)

_CHILD = r"""
import sys

from import_blocker import block_imports

export_dir, batch_path, out_path, with_cuda, blocked = sys.argv[1:6]
blocked = blocked.split(",")
block_imports(blocked)

import torch  # noqa: E402
from transformers import AutoModelForMaskedLM  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

model = AutoModelForMaskedLM.from_pretrained(export_dir, trust_remote_code=True)
model.eval()
module = type(model).__module__
out = {
    "module": module,
    "portable": sys.modules[module.rpartition(".")[0] + "._globals"].PORTABLE,
}
batch = torch.load(batch_path)
with torch.no_grad():
    out["cpu"] = model(**batch).logits
    if with_cuda == "1":
        model.to("cuda")
        cuda_batch = {k: v.cuda() for k, v in batch.items()}
        out["cuda"] = model(**cuda_batch).logits.cpu()
out["leaked"] = sorted(m for m in sys.modules if m.split(".")[0] in blocked)
torch.save(out, out_path)
"""


@pytest.mark.parametrize("case", list(CASES))
def test_export_loads_and_matches_without_nanoplm(tmp_path, case):
    export = tmp_path / "export"
    torch.manual_seed(0)
    model = NanoPLMForMaskedLM(NanoPLMConfig(**{**TINY, **CASES[case]})).eval()
    model.config.auto_map = AUTO_MAP
    model.save_pretrained(export)
    vendor_modeling_code(export)
    assert_no_unguarded_optional_imports(export)
    with torch.no_grad():
        expected = model(input_ids=IDS, attention_mask=MASK).logits
    torch.save({"input_ids": IDS, "attention_mask": MASK}, tmp_path / "batch.pt")

    env = dict(os.environ)
    env.pop("NANOPLM_PORTABLE", None)  # the bundle must be portable by itself
    env["PYTHONPATH"] = str(HERE)  # import_blocker, and nothing from src/
    env["HF_MODULES_CACHE"] = str(tmp_path / "hf_modules")
    env["NANOPLM_CUTLASS_ROOT"] = str(tmp_path / "no-cutlass-here")
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD,
            str(export),
            str(tmp_path / "batch.pt"),
            str(tmp_path / "out.pt"),
            "1" if WITH_CUDA else "0",
            ",".join(BLOCKED),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert child.returncode == 0, child.stderr[-6000:]

    out = torch.load(tmp_path / "out.pt")
    assert out["module"].startswith("transformers_modules.")
    assert out["portable"] is True
    assert out["leaked"] == []
    torch.testing.assert_close(out["cpu"], expected, rtol=0, atol=1e-5)
    if WITH_CUDA:
        torch.testing.assert_close(out["cuda"], expected, rtol=0, atol=1e-4)
