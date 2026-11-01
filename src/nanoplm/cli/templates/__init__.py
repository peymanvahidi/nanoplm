"""Package data YAML templates emitted by ``nanoplm * get-yaml``.

Templates live as real ``.yaml`` files next to this module so they can be
edited with normal YAML tooling. At runtime (and in tests) load them via
:func:`load_template`.
"""
from __future__ import annotations

from importlib import resources
from typing import Literal

TemplateName = Literal[
    "pretrain",
    "data_pretrain",
    "data_pretrain_subsampling",
    "data_distill",
    "eval",
]

_TEMPLATE_FILES: dict[str, str] = {
    "pretrain": "pretrain.yaml",
    "data_pretrain": "data_pretrain.yaml",
    "data_pretrain_subsampling": "data_pretrain_subsampling.yaml",
    "data_distill": "data_distill.yaml",
    "eval": "eval.yaml",
}


def load_template(name: TemplateName | str) -> str:
    """Return the UTF-8 text of a packaged template YAML file.

    ``name`` is a logical key (``pretrain``, ``data_pretrain``, …), not a path.
    """
    try:
        filename = _TEMPLATE_FILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(_TEMPLATE_FILES))
        raise ValueError(f"Unknown template {name!r}; expected one of: {known}") from exc

    # importlib.resources works for both editable installs and wheels once the
    # YAML files are declared as package data.
    return resources.files(__name__).joinpath(filename).read_text(encoding="utf-8")


__all__ = ["load_template", "TemplateName"]
