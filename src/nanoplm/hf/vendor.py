"""Write nanoPLM's modeling code into an export directory as a flat bundle.

transformers' remote-code loader copies an entry file plus every sibling it
names in a `from .x import y` line (found by regex, recursively) into one
synthetic package. It copies nothing else, and it requires every absolute
import in those files to be installed. So the bundle is computed, not listed:
starting from the two HF entry files, each file is copied with its absolute
`nanoplm.` imports rewritten to flat relative ones, and its relative imports
are followed with the loader's own regexes until the set is closed.

Three modules change on the way: `_globals.py` is pinned to portable mode,
`nanoplm.utils.logger` becomes a stdlib shim, and the recycling
compatibility YAML travels as a generated Python module.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

from transformers.dynamic_module_utils import get_imports, get_relative_imports

from nanoplm.utils.logger import logger

# src/nanoplm. Read at call time, so tests can point it at a patched copy.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# Paths below are relative to _PACKAGE_ROOT.
_ENTRY_FILES = ("hf/modeling_nanoplm.py", "hf/configuration_nanoplm.py")

AUTO_MAP = {
    "AutoConfig": "configuration_nanoplm.NanoPLMConfig",
    "AutoModel": "modeling_nanoplm.NanoPLMModel",
    "AutoModelForMaskedLM": "modeling_nanoplm.NanoPLMForMaskedLM",
}

# The one line that decides portable mode. A consumer is only guaranteed
# torch and transformers, so the bundle never imports a custom kernel.
_PORTABLE_FILE = "pretraining/models/modern_bert/_globals.py"
_PORTABLE_ANCHOR = (
    'PORTABLE: bool = _os_portable.environ.get("NANOPLM_PORTABLE", "0") == "1"'
    "  # vendor: rewritten to True in exported bundles"
)
_PORTABLE_REWRITE = "PORTABLE: bool = True  # rewritten by nanoplm.hf.vendor"

# Generated rather than copied, keyed by the source path an import resolves to.
# The matrix module has no source file: recycling_compatibility.py imports it
# only when the YAML beside it is missing, as it is in every bundle.
_LOGGER_FILE = "utils/logger.py"
_MATRIX_FILE = "config/_recycling_matrix.py"
_MATRIX_YAML = "config/recycling_compatibility.yaml"

_LOGGER_SHIM = '''\
"""Stands in for nanoplm.utils.logger, which creates output/logs/ on first emit."""

import logging

logger = logging.getLogger("nanoplm")
'''

# What a consumer is guaranteed to have: the stdlib, torch, and transformers
# with its hard requirements tokenizers (tokenizer.py) and pyyaml (the matrix).
_ALLOWED_IMPORTS = frozenset(sys.stdlib_module_names) | {
    "torch",
    "transformers",
    "tokenizers",
    "yaml",
}


def vendor_modeling_code(out_dir: Path) -> list[Path]:
    """Write the modeling bundle flat into `out_dir`; return the written paths.

    Does not run the import gate; call `assert_no_unguarded_optional_imports`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = _PACKAGE_ROOT
    sources: dict[str, Path] = {}  # bundle module name -> the file it came from
    pending = [root / entry for entry in _ENTRY_FILES]
    written: list[Path] = []
    while pending:
        source = pending.pop(0)
        name = source.stem
        if name in sources:
            if sources[name] != source:
                raise RuntimeError(
                    f"the flat bundle cannot hold both {sources[name]} and {source}: "
                    f"both would be {name}.py"
                )
            continue
        sources[name] = source
        text, rewritten = _bundle_text(source, root)
        path = out_dir / f"{name}.py"
        path.write_text(text, encoding="utf-8")
        written.append(path)
        # Exactly what the loader will follow: its own regexes, on the output.
        for module in sorted(get_relative_imports(path)):
            if module in rewritten:
                pending.append(rewritten[module])
            else:
                pending.append(_resolve_relative(source, module, root))
    logger.info("export: vendored %d modeling files into %s", len(written), out_dir)
    return written


def _bundle_text(source: Path, root: Path) -> tuple[str, dict[str, Path]]:
    """The bundle text for `source`, and the file each rewritten import names."""
    relative = source.relative_to(root).as_posix()
    if relative == _LOGGER_FILE:
        return _LOGGER_SHIM, {}
    if relative == _MATRIX_FILE:
        return _matrix_module((root / _MATRIX_YAML).read_text(encoding="utf-8")), {}
    text, rewritten = _rewrite_absolute_imports(source, root)
    if relative == _PORTABLE_FILE:
        text = _pin_portable(text, source)
    return text, rewritten


def _resolves(target: Path, root: Path) -> bool:
    return target.is_file() or target == root / _MATRIX_FILE


def _is_nanoplm(module: str | None) -> bool:
    return module is not None and module.split(".")[0] == "nanoplm"


def _rewrite_absolute_imports(source: Path, root: Path) -> tuple[str, dict[str, Path]]:
    """Rewrite each `from nanoplm.<pkg>.<mod> import` to `from .<mod> import`.

    Only the head of the statement changes, so a parenthesised multi-line
    import keeps its names, and the new line is one the loader's regex sees.
    """
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    siblings = set(get_relative_imports(source))
    rewritten: dict[str, Path] = {}
    for node in ast.walk(ast.parse(text, filename=str(source))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_nanoplm(alias.name):
                    raise RuntimeError(
                        f"{source}:{node.lineno}: `import {alias.name}` cannot be "
                        "vendored; use `from nanoplm.<package>.<module> import ...`"
                    )
            continue
        if not (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and _is_nanoplm(node.module)
        ):
            continue
        where = f"{source}:{node.lineno}"
        target = root.joinpath(*node.module.split(".")[1:]).with_suffix(".py")
        if not _resolves(target, root):
            raise RuntimeError(
                f"{where}: {node.module} does not resolve to a file under {root}"
            )
        if target == root / _LOGGER_FILE and [a.name for a in node.names] != ["logger"]:
            raise RuntimeError(
                f"{where}: only `logger` can be imported from nanoplm.utils.logger; "
                "the bundle replaces that module with a stdlib shim"
            )
        name = target.stem
        other = rewritten.get(name)
        if other is None and name in siblings:
            other = source.parent / f"{name}.py"
        if other is not None and other != target:
            raise RuntimeError(
                f"{where}: the flat bundle cannot hold both {other} and {target}: "
                f"both would be {name}.py"
            )
        rewritten[name] = target
        head = re.compile(rf"^([ \t]*)from[ \t]+{re.escape(node.module)}[ \t]+import\b")
        line, count = head.subn(
            lambda m: f"{m.group(1)}from .{name} import", lines[node.lineno - 1]
        )
        if count != 1:
            raise RuntimeError(
                f"{where}: cannot rewrite `from {node.module} import`; "
                "the statement must start its own line"
            )
        lines[node.lineno - 1] = line
    return "".join(lines), rewritten


def _resolve_relative(source: Path, module: str, root: Path) -> Path:
    """The file that `from .<module> import` in `source` names."""
    if not module.isidentifier():
        raise RuntimeError(
            f"{source}: `from .{module} import` cannot be flattened; "
            "the bundle has no subpackages"
        )
    target = source.parent / f"{module}.py"
    if not _resolves(target, root):
        raise RuntimeError(f"{source}: relative import .{module} has no file")
    return target


def _pin_portable(text: str, source: Path) -> str:
    lines = text.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.rstrip("\n") == _PORTABLE_ANCHOR]
    if len(hits) != 1:
        raise RuntimeError(
            f"{source}: expected this PORTABLE line once, found it {len(hits)} "
            f"times; the bundle must be pinned to portable mode:\n  {_PORTABLE_ANCHOR}"
        )
    lines[hits[0]] = _PORTABLE_REWRITE + "\n"
    return "".join(lines)


def _matrix_module(yaml_text: str) -> str:
    """A module holding `yaml_text` verbatim as `MATRIX_YAML`."""
    # In a triple-quoted literal only backslashes, and quotes that could end it
    # early (one followed by another, or the last character), need escaping.
    body = re.sub(r'"(?="|\Z)', r'\\"', yaml_text.replace("\\", "\\\\"))
    return (
        '"""The recycling compatibility matrix, embedded by nanoplm.hf.vendor.\n\n'
        "transformers' remote-code loader copies only Python files, so the YAML\n"
        "travels as text; recycling_compatibility.py falls back to it.\n"
        '"""\n\n'
        f'MATRIX_YAML = """{body}"""\n'
    )


def assert_no_unguarded_optional_imports(out_dir: Path) -> None:
    """Raise unless the bundle in `out_dir` loads with transformers and torch alone.

    Uses transformers' own scanners, because the loader decides with them.
    `get_imports` lists what a file requires, skipping only `try` blocks and
    `if is_*_available()` tests: `if torch.cuda.is_available():` guards
    nothing. `get_relative_imports` lists the siblings the loader copies.
    Running the code cannot catch this on the export machine, where every
    optional package is installed.
    """
    out_dir = Path(out_dir)
    problems = [
        f"{Path(entry).name}: missing"
        for entry in _ENTRY_FILES
        if not (out_dir / Path(entry).name).is_file()
    ]
    for path in sorted(out_dir.glob("*.py")):
        required = sorted(set(get_imports(path)) - _ALLOWED_IMPORTS)
        if required:
            problems.append(f"{path.name}: requires {', '.join(required)}")
        # Every file, not just the loader's closure from the entries: that
        # closure is a subset, and get_relative_import_files raises
        # FileNotFoundError at the first missing sibling instead of listing it.
        missing = sorted(
            f".{module}"
            for module in get_relative_imports(path)
            if not (out_dir / f"{module}.py").is_file()
        )
        if missing:
            problems.append(
                f"{path.name}: imports {', '.join(missing)}, not in the bundle"
            )
    if problems:
        raise RuntimeError(
            "the vendored bundle would not load with only transformers and torch "
            "installed:\n  " + "\n  ".join(problems)
        )
