"""Gate the run-directory pattern documented in pretrain.yaml against the code.

The docs_schema parity gate checks that a template documents every field that
exists. It cannot check that a comment is *true*, which is how the run_name
comment came to describe a ``DDMMHHMM-<run_name>`` layout that the naming code
has never produced. These tests close that gap for the one comment whose claim
is mechanically checkable.

Like the rest of this suite they read the code as source text and never import
nanoplm, so the CI schema job stays torch-free. The runtime counterpart that
actually drives ``prepare_run_and_steps`` is ``tests/test_run_dir_pattern.py``.
"""

import ast
import re

from tools.schema_gen.sources import REPO_SRC
from tools.schema_gen.templates import load_template_text

_STAMP_ENV_VARS = ("SLURM_JOB_ID", "TORCHELASTIC_RUN_ID")
_STAMP_HELPER_SUFFIX = "job_stamp"
_STAMP_HELPER_FILE = REPO_SRC / "utils" / "common.py"
_RUN_DIR_FILE = REPO_SRC / "pretraining" / "utils.py"
# How the comment spells each strftime directive the clock fallback may use.
_CLOCK_TOKENS = {"%d": "DD", "%m": "MM", "%H": "HH", "%M": "MM"}


def _run_name_comment() -> str:
    """Every comment documenting ``run_name:``, whether above it or end-of-line.

    Both placements are read so that moving the comment cannot quietly disable
    this gate.
    """
    lines = load_template_text("pretrain").splitlines()
    index = next(i for i, line in enumerate(lines) if line.strip().startswith("run_name:"))
    block = []
    _, _, trailing = lines[index].partition("#")
    if trailing.strip():
        block.append(trailing.strip())
    above = index
    while above > 0 and lines[above - 1].strip().startswith("#"):
        above -= 1
        block.insert(0, lines[above].strip().lstrip("#").strip())
    assert block, "run_name carries no comment, so nothing documents the run directory"
    return " ".join(block)


def _documented_layout() -> str:
    """The ``<ckp_dir>/...`` path shape the comment claims, e.g. ``<run_name>-<stamp>``."""
    match = re.search(r"<ckp_dir>/(\S+?)\.", _run_name_comment())
    assert match, f"comment names no <ckp_dir>/... layout: {_run_name_comment()!r}"
    return match.group(1)


def _stamp_helper() -> tuple[ast.FunctionDef, str]:
    """The helper that turns the environment into the run-directory stamp, and its source.

    Found by name suffix rather than exact name so that making it public, or
    private again, does not silently disable the checks below.
    """
    text = _STAMP_HELPER_FILE.read_text(encoding="utf-8")
    named = [
        node
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.FunctionDef) and node.name.endswith(_STAMP_HELPER_SUFFIX)
    ]
    assert len(named) == 1, f"expected one *{_STAMP_HELPER_SUFFIX} helper, found {len(named)}"
    return named[0], ast.get_source_segment(text, named[0])


def _code_layout() -> str:
    """The run-directory name the code builds, rendered as ``<run_name>-<stamp>``.

    Found as the one f-string that formats both the configured ``run_name`` and
    the stamp helper's result; any literal text between them is kept verbatim.
    """
    def render(part):
        if isinstance(part, ast.Constant):
            return str(part.value)
        value = part.value
        if isinstance(value, ast.Attribute) and value.attr == "run_name":
            return "<run_name>"
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id.endswith(_STAMP_HELPER_SUFFIX)
        ):
            return "<stamp>"
        return None

    layouts = []
    for node in ast.walk(ast.parse(_RUN_DIR_FILE.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.JoinedStr):
            continue
        parts = [render(part) for part in node.values]
        if None in parts:
            continue
        layout = "".join(parts)
        if "<run_name>" in layout and "<stamp>" in layout:
            layouts.append(layout)
    assert len(layouts) == 1, f"expected one f-string naming the run directory, found {layouts}"
    return layouts[0]


def test_comment_documents_the_layout_the_code_builds():
    assert _documented_layout() == _code_layout()


def test_comment_names_every_stamp_source_the_code_reads():
    """A stamp source added to the code must be documented, not just added."""
    _, source = _stamp_helper()
    read_from_env = set(re.findall(r'"([A-Z][A-Z0-9_]+)"', source)) - {"WORLD_SIZE"}
    assert read_from_env == set(_STAMP_ENV_VARS), read_from_env
    comment = _run_name_comment()
    for name in read_from_env:
        assert name in comment, f"{name} sets the stamp but the template omits it"


def test_comment_documents_the_clock_format_the_code_falls_back_to():
    helper, _ = _stamp_helper()
    formats = [
        node.args[0].value
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "strftime"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    assert len(formats) == 1, f"expected one strftime fallback, found {formats}"
    directives = re.findall(r"%[A-Za-z]", formats[0])
    assert "".join(directives) == formats[0], f"format has text outside its directives: {formats[0]!r}"
    documented = "".join(_CLOCK_TOKENS[d] for d in directives)
    assert documented in _run_name_comment(), f"comment does not spell the clock stamp as {documented}"
