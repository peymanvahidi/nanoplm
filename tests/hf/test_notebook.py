"""Structural and behavioral checks on the quickstart notebook.

A notebook is the least testable artifact in the repo and this one carries the
canonical "how to use a nanoPLM model" recipe, so beyond static assertions
(R16-1, R16-2, R16-5, R16-6) it is actually RUN: every untagged code cell is
collected in order, `REPO` is substituted for a tiny local export directory
(R16-3), and the concatenated script is executed in a fresh subprocess with
nanoplm/triton/flash-attn/TE blocked, the same shape of check
tests/hf/test_model_card.py's `test_quickstart_runs_and_prints_expected_shapes`
uses for the model card's own quickstart snippet.

Cells tagged "network" (the dataset download and the UMAP cell that consumes
it) are excluded: they need a real internet connection and a full
umap-learn/matplotlib/datasets install, which is deliberately NOT part of the
project venv (R16-4) -- those are exercised separately, end to end, in a
throwaway consumer venv.
"""

import json
import re
from pathlib import Path

from nanoplm.hf.model_card import QUICKSTART
from tests.hf.test_model_card import _build_export, _run_quickstart

NOTEBOOK = Path(__file__).resolve().parents[2] / "examples" / "nanoplm_quickstart.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text())


def _cells() -> list[dict]:
    return _notebook()["cells"]


def _source_of(cell: dict) -> str:
    return "".join(cell["source"])


def _source() -> str:
    return "\n".join(_source_of(c) for c in _cells())


def _code_cells() -> list[dict]:
    return [c for c in _cells() if c["cell_type"] == "code"]


def _untagged_code_cells() -> list[dict]:
    return [
        c
        for c in _code_cells()
        if "network" not in c.get("metadata", {}).get("tags", [])
    ]


def _network_code_cells() -> list[dict]:
    return [
        c for c in _code_cells() if "network" in c.get("metadata", {}).get("tags", [])
    ]


def _cell_source_containing(needle: str) -> str:
    """The source of the one code cell containing `needle` (fails loudly if not
    exactly one) -- for checks that must look at a SPECIFIC cell, not the
    whole notebook: the immutable QUICKSTART cell already contains many
    strings (`special_tokens_mask`, `attention_mask`, ...) that would make a
    whole-notebook grep pass even if the cell actually under test were wrong.
    """
    matches = [c for c in _code_cells() if needle in _source_of(c)]
    assert (
        len(matches) == 1
    ), f"expected exactly one code cell containing {needle!r}, found {len(matches)}"
    return _source_of(matches[0])


def _dataset_cell_source() -> str:
    matches = [c for c in _network_code_cells() if "load_dataset(" in _source_of(c)]
    assert (
        len(matches) == 1
    ), "expected exactly one network cell calling load_dataset(...)"
    return _source_of(matches[0])


# --- the plan's structural tests (adapted per R16-1..R16-6) ------------------


def test_notebook_is_valid_json_with_cells():
    nb = _notebook()
    assert nb["cells"], "notebook has no cells"
    assert nb["nbformat"] >= 4


def test_uses_the_documented_public_api():
    src = _source()
    assert "trust_remote_code=True" in src
    assert "AutoTokenizer.from_pretrained" in src
    assert "AutoModel.from_pretrained" in src


def test_does_not_import_nanoplm():
    # The whole point: consumers need only transformers + torch.
    src = _source()
    assert "import nanoplm" not in src
    assert "from nanoplm" not in src


def test_declares_its_extra_dependencies():
    src = _source()
    assert "umap-learn" in src
    assert "pip install" in src


def test_covers_all_four_steps():
    src = _source().lower()
    for marker in ("load", "tokeniz", "embed", "umap"):
        assert marker in src


def test_selects_a_device():
    src = _source()
    assert "mps" in src and "cuda" in src


# --- (a) the first code cell is QUICKSTART, verbatim (R16-1) -----------------


def test_first_code_cell_is_the_model_cards_quickstart_verbatim():
    first_code = _code_cells()[0]
    assert _source_of(first_code) == QUICKSTART.format(repo_id="acme/nanoplm-base")


def test_markdown_cell_immediately_before_the_first_code_cell_names_repo():
    cells = _cells()
    first_code_index = next(i for i, c in enumerate(cells) if c["cell_type"] == "code")
    assert first_code_index > 0
    before = cells[first_code_index - 1]
    assert before["cell_type"] == "markdown"
    assert "REPO" in _source_of(before)


# --- (b) no cell slices residues by length (R16-2) ---------------------------


def test_no_cell_slices_residues_by_raw_length():
    assert ": len(seq)]" not in _source()


def test_embed_helper_uses_the_special_tokens_mask():
    # Scoped to the embed cell itself (I3 fix-round-1 finding): the immutable
    # QUICKSTART cell already contains every one of these strings, so a
    # whole-notebook grep here would pass even if `embed()` were broken.
    src = _cell_source_containing("def embed(")
    assert "return_special_tokens_mask=True" in src
    assert "special_tokens_mask" in src
    assert "attention_mask" in src and "~special" in src.replace(" ", "")


def test_tokenize_markdown_documents_special_tokens_accurately():
    src = _source().lower()
    for marker in ("</s>", "register", "bos", "uzob", "normali"):
        assert marker in src, marker


def test_dataset_sequence_length_is_bounded_by_a_named_constant():
    src = _source()
    assert "MAX_SEQ_LEN" in src


# --- (c) outputs are cleared before commit (R16-5) ---------------------------


def test_every_code_cell_has_no_outputs_or_execution_count():
    for cell in _code_cells():
        assert cell["outputs"] == [], cell["source"]
        assert cell["execution_count"] is None, cell["source"]


# --- (d) the dataset cell pins a revision (R16-6) -----------------------------


def test_dataset_cell_pins_a_revision():
    network_cells = [
        c for c in _code_cells() if "network" in c.get("metadata", {}).get("tags", [])
    ]
    assert network_cells, "expected at least one cell tagged 'network'"
    dataset_cell_src = "\n".join(_source_of(c) for c in network_cells)
    assert "load_dataset(" in dataset_cell_src
    assert re.search(r"revision\s*=", dataset_cell_src)


def test_network_cells_cover_dataset_download_and_umap():
    network_cells = [
        c for c in _code_cells() if "network" in c.get("metadata", {}).get("tags", [])
    ]
    joined = "\n".join(_source_of(c) for c in network_cells)
    assert "load_dataset" in joined
    assert "umap" in joined.lower()


# --- (e) labels come from the dataset's own class, not a naive location split
# (fix round 1, I1): "Cytoplasm,Nucleus,U" rows carry the dataset's own
# label 0, the same id as every plain "Nucleus,*" row -- so `location`'s
# comma-separated compartments must not be resolved by just taking the first
# one, which silently misfiles every dual-localization row.


def test_labels_come_from_the_datasets_own_label_column():
    src = _dataset_cell_source()
    assert '["label"]' in src
    # the buggy fix-round-0 derivation (first comma-separated compartment,
    # wrong for dual-localization rows) must not reappear.
    assert '.split(",")[0]' not in src


def test_label_to_name_mapping_is_verified_unambiguous_before_use():
    src = _dataset_cell_source()
    assert "assert" in src
    assert "len(c) == 1" in src or "len(names) == 1" in src


def test_markdown_explains_multi_compartment_handling():
    src = _source().lower()
    assert "dual-localiz" in src or "multi-compartment" in src


# --- (f) the residue-length filter agrees with the tokenizer's truncation
# (fix round 1, I2): `MAX_SEQ_LEN` bounds TOKEN count (residues + EOS/BOS/
# register overhead), so filtering the dataset by raw residue count against
# the same number silently lets boundary-length sequences get truncated by
# `embed()` -- contradicting the "untruncated" claim in the markdown.


def test_length_filter_does_not_compare_raw_residues_to_the_token_cap():
    src = _source()
    assert 'len(ex["aa_seq"]) <= MAX_SEQ_LEN' not in src


def test_residue_cap_is_derived_once_and_reused_by_the_dataset_filter():
    src = _source()
    assert (
        src.count("RESIDUE_CAP =") == 1
    ), "RESIDUE_CAP must be derived in exactly one place"
    assert "RESIDUE_CAP" in _dataset_cell_source()


# --- the hermetic run: every untagged code cell actually executes ------------


def _script_for(repo_dir) -> str:
    """Every untagged code cell, in order, with REPO substituted."""
    script = "\n\n".join(_source_of(cell) for cell in _untagged_code_cells())
    # The quickstart cell defines REPO = "acme/nanoplm-base" on its own line;
    # point it at the tiny local export instead of rewriting the notebook.
    assert 'REPO = "acme/nanoplm-base"' in script
    script = script.replace('REPO = "acme/nanoplm-base"', f"REPO = {str(repo_dir)!r}")
    return script


def test_untagged_cells_run_end_to_end_with_registers(tmp_path):
    cfg = _build_export(tmp_path, num_register_tokens=2)
    script = _script_for(tmp_path)
    result = _run_quickstart(tmp_path, script)
    assert result.returncode == 0, result.stderr[-6000:]

    lines = [l for l in result.stdout.splitlines() if l.strip()]
    # QUICKSTART's own two prints (one per demo sequence).
    quickstart_sequences = ["MKALCLLLLPVLG", "MVLSPADKTNVKAAWGKV"]
    for seq, line in zip(quickstart_sequences, lines[:2]):
        assert line.startswith(seq)
        assert f"torch.Size([{len(seq)}, {cfg.hidden_size}])" in line

    # device print.
    assert any(line.startswith("device:") for line in lines)

    # tokenize-demo print: a 2D tensor of input ids.
    assert any(line.startswith("tensor([[") for line in lines)

    # embed-helper demo print: (n, hidden) shape for 2 literal sequences.
    demo_sequences = ["MKALCLLLLPVLG", "MVLSPADKTNVKAAWGKV"]
    assert f"torch.Size([{len(demo_sequences)}, {cfg.hidden_size}])" in result.stdout


# --- (g) embed() is checked numerically, not just by output shape (fix round
# 1, I3): a wrong mask (a dropped `~`, pooling over padding, ...) keeps
# `embed()`'s output shape correct while silently changing its VALUES, so the
# shape-only checks above cannot catch it. This runs `embed()` on sequences of
# DIFFERENT lengths (so the batched call actually pads) and compares each row,
# within a tight tolerance, against an independent reference: a single
# unpadded forward per sequence, pooled over exactly the positions the
# tokenizer's own special-tokens mask marks as real residues.

_MASK_CHECK_SNIPPET = """
# --- appended by tests/hf/test_notebook.py, not part of the notebook itself:
# an independent numeric check of embed()'s masking (I3) ---
_verify_sequences = ["MKALCLLLLPVLG", "MVLSPADKTNVKAAWGKV", "MSTNPKPQRKIQFAVLLLWMV"]
_batched = embed(_verify_sequences)

_reference_rows = []
for _seq in _verify_sequences:
    _single = tokenizer(
        [_seq], return_tensors="pt", return_special_tokens_mask=True
    ).to(device)
    _special = _single.pop("special_tokens_mask").bool()
    _residue_mask = _single["attention_mask"].bool() & ~_special
    with torch.no_grad():
        _hidden = model(**_single).last_hidden_state
    _reference_rows.append(_hidden[0][_residue_mask[0]].mean(dim=0).float().cpu())
_reference = torch.stack(_reference_rows)

_max_diff = (_batched - _reference).abs().max().item()
print("MASK_CHECK_MAX_DIFF:", _max_diff)
assert _max_diff < 1e-4, f"embed() disagrees with an independent unpadded reference: {_max_diff}"
"""


def test_embed_matches_an_independent_unpadded_reference(tmp_path):
    _build_export(tmp_path, num_register_tokens=2)
    script = _script_for(tmp_path) + "\n\n" + _MASK_CHECK_SNIPPET
    result = _run_quickstart(tmp_path, script)
    assert result.returncode == 0, result.stderr[-6000:]
    assert "MASK_CHECK_MAX_DIFF:" in result.stdout
