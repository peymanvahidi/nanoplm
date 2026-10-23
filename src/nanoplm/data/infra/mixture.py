"""The mixture manifest: several prepared pretraining datasets and their weights.

``nanoplm data from-yaml`` builds each source of a data.yaml with more than one source
into ``<output_dir>/<name>/`` (or reads a ``prepared`` folder as it is) and then writes
``<output_dir>/.data_manifest`` with ``write_mixture``. That file is a MixtureManifest:
per source the folder (relative to output_dir), the weight, the kind and counts, and the
sha256 of the source's own .data_manifest at the time the mix was written.

Training reads it back with ``load_sources``, which refuses a source whose manifest
changed since then: its counts in the mix would be stale.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

from nanoplm.data.infra.manifest import (
    MANIFEST_FILENAME, MixtureManifest, PretrainManifest, read_manifest, write_manifest,
)

WEIGHT_TOLERANCE = 1e-6
RERUN = "Rerun `nanoplm data from-yaml` to write the mixture again."


class MixtureError(ValueError):
    """A mixture cannot be written or read."""


def manifest_hash(folder: Union[str, Path]) -> str:
    """sha256 of a dataset folder's .data_manifest, as the mixture records it."""
    return hashlib.sha256((Path(folder) / MANIFEST_FILENAME).read_bytes()).hexdigest()


def weights_error(weights: dict) -> Optional[str]:
    """Why these weights cannot be used, or None. Each is a share of training sequences."""
    for name, w in weights.items():
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w):
            return f"sources.{name}.weight must be a number, got {w!r}"
        if w <= 0:
            return f"sources.{name}.weight must be above 0, got {w:g}"
    total = sum(weights.values())
    if abs(total - 1) > WEIGHT_TOLERANCE:
        parts = " + ".join(f"{w:g} ({n})" for n, w in weights.items())
        return (f"the weights must sum to 1 (they are shares of training sequences): "
                f"{parts} = {total:g}")
    return None


def _rel(path: Path, root: Path) -> str:
    """``path`` relative to ``root`` when it sits inside it, else absolute."""
    path, root = path.resolve(), root.resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _source_manifest(name: str, folder: Path) -> PretrainManifest:
    where = f"sources.{name}"
    try:
        m = read_manifest(folder)
    except (FileNotFoundError, ValueError) as e:
        raise MixtureError(f"{where}: {folder} is not a built dataset ({e})") from e
    if m.pipeline_mode != "pretrain":
        raise MixtureError(f"{where}: {folder} holds {m.pipeline_mode} data, not pretraining data")
    if m.is_mixture:
        raise MixtureError(f"{where}: {folder} is itself a mixture; list its sources instead")
    return m


def write_mixture(output_dir: Union[str, Path], entries: Sequence[tuple]) -> Path:
    """Write the mixture manifest. ``entries`` holds (name, folder, weight, fasta) per
    source; fasta is the URL or path it was built from, or the prepared folder."""
    out = Path(output_dir)
    if len(entries) < 2:
        raise MixtureError("a mixture needs at least two sources")
    names = [e[0] for e in entries]
    if len(set(names)) != len(names):
        raise MixtureError(f"duplicate source names: {', '.join(names)}")
    err = weights_error({e[0]: e[2] for e in entries})
    if err:
        raise MixtureError(err)
    rows, manifests = [], []
    for name, folder, weight, fasta in entries:
        folder = Path(folder)
        m = _source_manifest(name, folder)
        manifests.append(m)
        rows.append({
            "name": name,
            "dir": _rel(folder, out),
            "weight": float(weight),
            "kind": "cluster" if m.is_cluster_sampled else "plain",
            "fasta": None if fasta is None else str(fasta),
            "train_sequences": m.train_sequences,
            "val_sequences": m.val_sequences,
            "pool_sequences": m.pool_sequences,
            "total_train_tokens": m.total_train_tokens,
            "min_seq_len": m.min_seq_len,
            "max_seq_len": m.max_seq_len,
            "register_tokens": m.num_register_tokens,
            "manifest_sha256": manifest_hash(folder),
        })
    regs = {r["name"]: r["register_tokens"] for r in rows}
    if len(set(regs.values())) > 1:
        got = ", ".join(f"{n} {v}" for n, v in regs.items())
        raise MixtureError(f"the sources must use the same register_tokens; got {got}")
    mix = MixtureManifest(
        pipeline_mode="pretrain",
        seqs_num=sum(m.seqs_num for m in manifests),
        min_seq_len=min(m.min_seq_len for m in manifests),
        max_seq_len=max(m.max_seq_len for m in manifests),
        val_ratio=round(sum(r["val_sequences"] for r in rows)
                        / max(1, sum(r["train_sequences"] + r["val_sequences"] for r in rows)), 6),
        train_dir="",
        val_dir="",
        train_sequences=sum(r["train_sequences"] for r in rows),
        val_sequences=sum(r["val_sequences"] for r in rows),
        num_register_tokens=rows[0]["register_tokens"],
        samples_per_shard=max(m.samples_per_shard for m in manifests),
        total_train_tokens=sum(r["total_train_tokens"] for r in rows),
        sources=rows,
    )
    return write_manifest(out, mix)


@dataclass
class MixtureSource:
    """One source of a mixture as training opens it."""

    name: str
    dir: Path
    weight: float
    manifest: PretrainManifest
    row: dict

    @property
    def kind(self) -> str:
        return self.row["kind"]


def load_sources(dataset_dir: Union[str, Path], manifest: MixtureManifest) -> list:
    """The sources of a mixture, each checked against the manifest hash the mix recorded."""
    root = Path(dataset_dir)
    out = []
    for row in manifest.sources:
        folder = Path(row["dir"])
        folder = folder if folder.is_absolute() else root / folder
        where = f"mixture source {row['name']}"
        if not (folder / MANIFEST_FILENAME).is_file():
            raise MixtureError(f"{where}: {folder / MANIFEST_FILENAME} not found. {RERUN}")
        if manifest_hash(folder) != row["manifest_sha256"]:
            raise MixtureError(
                f"{where}: {folder / MANIFEST_FILENAME} changed after the mixture was "
                f"written (the source was rebuilt or edited). {RERUN}")
        m = _source_manifest(row["name"], folder)
        out.append(MixtureSource(name=row["name"], dir=folder, weight=float(row["weight"]),
                                 manifest=m, row=row))
    return out


def source_names(manifest) -> list:
    """The source names of a mixture manifest, [] for any other dataset."""
    return [r["name"] for r in manifest.sources] if getattr(manifest, "is_mixture", False) else []


def resolve_eval_sources(eval_sources, manifest) -> list:
    """The sources eval runs on: every one for ``all``, else the named ones in mixture
    order. Unknown names and a list on a dataset that is no mixture are errors."""
    names = source_names(manifest)
    if eval_sources in (None, "all"):
        return names
    if isinstance(eval_sources, str):
        eval_sources = [eval_sources]
    if not names:
        raise MixtureError(f"eval_sources: {list(eval_sources)} names sources, but the dataset "
                           "is not a mixture; set eval_sources: all")
    if not eval_sources:
        raise MixtureError("eval_sources is empty; use all or list one or more sources")
    unknown = [n for n in eval_sources if n not in names]
    if unknown:
        raise MixtureError(f"eval_sources: unknown source{'s' if len(unknown) > 1 else ''} "
                           f"{', '.join(map(str, unknown))} (the mixture has {', '.join(names)})")
    return [n for n in names if n in eval_sources]



# ------------------------------------------------------------------ the run plan

REPEAT_WARN = 10      # warn when a source would be read more than this many times


def run_plan(manifest: MixtureManifest, budget_tokens: int) -> list:
    """Per source: share, size, expected tokens used and epochs over ``budget_tokens``.

    The run draws ``budget_tokens / mean sample length`` sequences, ``share`` of them
    from each source (the share MixtureDataset gives it: weight rounded to whole
    positions per mixture epoch). Packing waste is ignored, as in the passes hint.
    """
    from nanoplm.data.datasets.mixture import allocate
    rows = manifest.sources
    sizes = [max(1, int(r["train_sequences"])) for r in rows]
    n = sum(sizes)
    shares = [k / n for k in allocate([float(r["weight"]) for r in rows], n)]
    per_sample = [r["total_train_tokens"] / s for r, s in zip(rows, sizes)]
    mean_len = sum(w * t for w, t in zip(shares, per_sample)) or 1.0
    samples = budget_tokens / mean_len
    out = []
    for r, size, share, t in zip(rows, sizes, shares, per_sample):
        used = samples * share
        out.append({"name": r["name"], "weight": float(r["weight"]), "share": share,
                    "kind": r["kind"], "size": size, "samples": used,
                    "tokens": used * t, "epochs": used / size})
    return out


def format_run_plan(plan: list, budget_tokens: int) -> list:
    """The training-start table as lines, then one warning line per source read more
    than REPEAT_WARN times."""
    head = ("source", "share", "size", "tokens used", "epochs")
    rows = [head] + [(p["name"], f"{p['share']:.3f}",
                      f"{p['size']:,} {'clusters' if p['kind'] == 'cluster' else 'seqs'}",
                      f"{p['tokens']:,.0f}", f"{p['epochs']:.2f}") for p in plan]
    widths = [max(len(r[i]) for r in rows) for i in range(len(head))]
    lines = [f"Data mixture over a budget of {budget_tokens:,} tokens "
             "(share = share of training sequences):"]
    lines += ["  " + "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                               for i, (c, w) in enumerate(zip(r, widths))) for r in rows]
    for p in plan:
        if p["epochs"] > REPEAT_WARN:
            lines.append(f"Warning: source {p['name']} is read {p['epochs']:.1f} times over "
                         f"the run (more than {REPEAT_WARN}x); lower its weight or add data")
    return lines


# ------------------------------------------------------------------ eval


def eval_split(val_sizes: dict, eval_names: list, micro_batch_seqs: int) -> dict:
    """Validation sequences per evaluated source. The eval keeps the batch count one
    loader over every source's val set would have, split evenly over ``eval_names``;
    a source with fewer val sequences than its part uses all it has."""
    total_batches = math.ceil(sum(val_sizes.values()) / micro_batch_seqs)
    per_source = math.ceil(total_batches / len(eval_names)) * micro_batch_seqs
    return {n: min(val_sizes[n], per_source) for n in eval_names}


def eval_workers(num_workers: int, n_loaders: int) -> int:
    """DataLoader workers per source's val loader: ``num_workers`` split over the
    loaders, at least one each when there are any. Eval workers start anew at every
    eval when persistent workers are off (packing), so K loaders of num_workers each
    would start K times as many processes as one val set does."""
    return 0 if num_workers <= 0 else max(1, math.ceil(num_workers / n_loaders))


def eval_payload(losses: dict, weights: dict, suffix: str = "") -> dict:
    """W&B keys of one mixture eval: ``eval/loss/<source>`` per source and ``eval/loss``,
    the mean of those weighted by each source's share (over the sources evaluated).
    ``suffix`` (e.g. "/r4") goes after the source name; eval/loss then gets none."""
    out = {f"eval/loss/{n}{suffix}": v for n, v in losses.items()}
    if not suffix:
        out["eval/loss"] = weighted_loss(losses, weights)
    return out


def eval_keys(by_r: dict, target_r, weights: dict) -> dict:
    """The loss keys of one mixture eval. ``by_r`` maps each loop count r evaluated to
    {source: loss}; without recycling it holds the one key None. With recycling every r
    gives eval/loss/<source>/r<r>, and the unsuffixed keys are those of ``target_r``,
    left out when the sweep does not hold it (as a single-source run leaves out
    eval/loss)."""
    if None in by_r:
        return eval_payload(by_r[None], weights)
    out = {}
    for r, losses in by_r.items():
        out.update(eval_payload(losses, weights, suffix=f"/r{r}"))
    if target_r in by_r:
        out.update(eval_payload(by_r[target_r], weights))
    return out


def weighted_loss(losses: dict, weights: dict) -> float:
    total = sum(weights[n] for n in losses)
    return sum(weights[n] * v for n, v in losses.items()) / total
