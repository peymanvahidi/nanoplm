#!/usr/bin/env python3
"""
nanoPLM CLI - Data subcommands for nanoPLM package
"""

import click
import os
from pathlib import Path
import subprocess
import sys
import time

from nanoplm.config.datasets import DATASET_URLS

from nanoplm.data.preparation.downloader import Downloader, DownloadError
from nanoplm.data.preparation.extractor import Extractor, ExtractionError
from nanoplm.data.preparation.shuffler import FastaShuffler, ShufflingError
from nanoplm.data.preparation.filterer import Filterer, FilterError
from nanoplm.data.preparation.splitor import Splitor, SplitError
from nanoplm.data.infra.manifest import PretrainManifest, DistillationManifest, write_manifest
from nanoplm.data.infra.validation import ValidationError
from nanoplm.data.sharding.distill_shard_writer import SaveKDDataset, shard_h5_file
from nanoplm.distillation.models.teacher import ProtT5
from nanoplm.data.sharding.pretrain_shard_writer import ShardWriter
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer

from nanoplm.utils import create_dirs, logger
from nanoplm.utils.common import inside_git_repo, is_git_subdir, read_yaml, is_flash_attention_available


# dvc is a declared dependency, so the interpreter running nanoplm always has
# it importable. A bare "dvc" on PATH does not: the documented cluster launch
# is "$REPO_ROOT/.venv/bin/nanoplm" with the venv unactivated, where PATH has
# no dvc at all.
_DVC = [sys.executable, "-m", "dvc"]


# Our own pipeline exceptions. Their messages are written for a user, so they
# stand alone as a one-line error.
from nanoplm.data.preparation.cluster_config import ClusterConfigError
from nanoplm.data.preparation.cluster_index import ClusterIndexError
from nanoplm.data.preparation.cluster_table import ClusterTableError
from nanoplm.data.preparation.uniref_hierarchy import HierarchyError

_PIPELINE_ERRORS = (
    DownloadError, ExtractionError, ShufflingError, FilterError, SplitError, ValidationError,
    ClusterConfigError, ClusterIndexError, ClusterTableError, HierarchyError,
)

# Third-party and builtin types the stages raise for real input problems: h5py
# and the native backends surface corrupt files as OSError and RuntimeError.
# These same types are also what a bug in our own code raises, and we cannot
# tell the two apart from the type alone, so their message carries a pointer to
# the traceback rather than pretending the cause is understood.
_OPAQUE_ERRORS = (OSError, EOFError, RuntimeError, ValueError)

_DEBUG_HINT = "Re-run with NANOPLM_DEBUG=1 for the full traceback."


class _GroupedDataCommand(click.Group):
    _GROUPS = [
        ("Main commands (most users only need these — they orchestrate the rest "
         "of the pipeline through DVC under the hood)",
         ["get-yaml", "from-yaml"]),
        ("Pipeline stages (invoked by 'from-yaml' via DVC; can also be run "
         "directly for debugging)",
         ["get-file", "extract", "check", "shuffle", "filter", "split"]),
        ("Pretraining-specific",
         ["pre-tokenize"]),
        ("Knowledge-distillation-specific",
         ["kd-embed", "kd-reshard"]),
    ]

    def format_commands(self, ctx, formatter):
        registered = {c: self.get_command(ctx, c) for c in self.list_commands(ctx)}
        seen = set()
        for header, names in self._GROUPS:
            rows = []
            for name in names:
                cmd = registered.get(name)
                if cmd is None or cmd.hidden:
                    continue
                rows.append((name, cmd.get_short_help_str() or ""))
                seen.add(name)
            if rows:
                with formatter.section(header):
                    formatter.write_dl(rows)
        # Any commands not in a group (shouldn't be visible normally)
        leftover = [(n, c) for n, c in registered.items()
                    if n not in seen and not (c and c.hidden)]
        if leftover:
            with formatter.section("Other"):
                formatter.write_dl([(n, c.get_short_help_str() or "") for n, c in leftover])

    def invoke(self, ctx):
        # Centralizes failure presentation for every 'data' subcommand: an
        # expected domain error becomes one clean line instead of each
        # subcommand carrying its own try/except (or none at all).
        try:
            return super().invoke(ctx)
        except (click.exceptions.Exit, click.Abort):
            # Both subclass RuntimeError (Click's own quirk), but are clean
            # exits (--help, ctx.exit(), Ctrl-C prompts), not domain errors.
            raise
        except _PIPELINE_ERRORS as e:
            if os.environ.get("NANOPLM_DEBUG"):
                raise
            raise click.ClickException(str(e)) from e
        except _OPAQUE_ERRORS as e:
            if os.environ.get("NANOPLM_DEBUG"):
                raise
            raise click.ClickException(
                f"{type(e).__name__}: {e}\n{_DEBUG_HINT}"
            ) from e


def _refuse_overwriting(input_path: Path, output_path: Path) -> None:
    """Stop when the output names the input: the stage would replace or delete it."""
    def entry(p: Path) -> Path:         # the directory entry; a symlink there is not followed
        return Path(os.path.realpath(p.parent)) / p.name
    if entry(output_path) == entry(input_path):
        raise click.ClickException(f"-o {output_path} is the input file; give -o another file "
                                   "name or a directory.")


def _output_file(input_path: Path, output_path: Path, suffix: str) -> Path:
    """``output_path``, or in that directory the input's name with ``suffix`` in place of
    its .fasta/.fa(.gz/.zst) ending (x.fa -> x_filtered.fasta); never the input itself."""
    from nanoplm.data.preparation.data_config import fasta_stem
    if output_path.is_dir():
        output_path = output_path / f"{fasta_stem(input_path.name)}{suffix}"
    _refuse_overwriting(input_path, output_path)
    return output_path


@click.group(name="data", cls=_GroupedDataCommand)
@click.help_option("--help", "-h")
def data():
    """Manage protein datasets for nanoPLM pretraining and distillation."""
    pass


@data.command("get-file")
@click.argument(
    "dataset",
    required=False,
    type=click.Choice(DATASET_URLS.keys(), case_sensitive=False),
    default=None,
)
@click.option(
    "--url",
    required=False,
    type=str,
    default="",
    help="URL to download (empty string treated as null).",
)
@click.option(
    "--local-fasta",
    required=False,
    type=str,
    default="",
    help="Local .fasta or .fa path, optionally .gz or .zst (empty string treated as null).",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Directory path or file path where the dataset will be saved.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Flag to force overwrite if output already exists.",
)
@click.help_option("--help", "-h")
def get_file(dataset, url, local_fasta, output, force):
    """Acquire a FASTA source. Pass either a dataset key, --url, or --local-fasta.

    Examples:

    \b
      nanoplm data get-file uniref50 -o data_directory
      nanoplm data get-file --url https://example.com/data.fasta.gz -o data_directory/mydata.fasta.gz
      nanoplm data get-file --local-fasta /path/to/local.fasta.gz -o data_directory/local.fasta.gz
    """
    url = url or None
    local_fasta = local_fasta or None
    sources = [bool(dataset), bool(url), bool(local_fasta)]
    if sum(sources) != 1:
        raise click.UsageError(
            "Provide exactly one of: dataset key, --url, --local-fasta"
        )
    output_path = Path(output)
    create_dirs(output_path)
    # If output is a directory, derive a filename from the source.
    if output_path.is_dir():
        if local_fasta:
            output_path = output_path / Path(local_fasta).name
        else:
            chosen_url = DATASET_URLS[dataset] if dataset else url
            output_path = output_path / Path(chosen_url).name
    if local_fasta:
        _refuse_overwriting(Path(local_fasta), output_path)
    if output_path.exists() and not force:
        click.echo(f"File already exists: {output_path}. Use --force to overwrite.")
        return
    if output_path.exists() and force:
        output_path.unlink()
    if local_fasta:
        if not Path(local_fasta).exists():
            raise click.UsageError(
                f"--local-fasta path does not exist: {local_fasta}"
            )
        Downloader.link_local(Path(local_fasta), output_path)
        click.echo(f"Linked local file: {local_fasta} -> {output_path}")
        return
    chosen_url = DATASET_URLS[dataset] if dataset else url
    Downloader(url=chosen_url, output_path=output_path).download()
    click.echo(f"Downloaded to: {output_path}")


@data.command("extract")
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input file path where the dataset is saved.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(exists=False),
    help="Output file path / directory where the dataset will be saved.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force extraction even if output file already exists",
    default=False,
)
@click.help_option("--help", "-h")
def extract(input: str, output: str, force: bool):
    """Extract a .gz or .zst file"""
    from nanoplm.data.preparation.data_config import is_compressed, strip_compressed

    input_path = Path(input)
    output_path = Path(output)
    if not is_compressed(input):
        raise click.ClickException(f"Input must be a .gz or .zst file, got: {input}")

    create_dirs(output_path)

    # Resolve output file path
    if output_path.is_dir():
        output_path = output_path / strip_compressed(input_path.name)
    _refuse_overwriting(input_path, output_path)

    # Check if output file already exists
    if output_path.exists():
        if force:
            output_path.unlink()
            click.echo(f"Removed existing file: {output_path}")
        else:
            click.echo(f"File already exists: {output_path} \nUse --force to overwrite")
            return

    # Extract dataset
    extractor = Extractor(input_path=input_path, output_path=output_path)
    extractor.extract()
    click.echo(f"Extraction completed successfully. Output: {output_path}")


@data.command("shuffle")
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input FASTA file path",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(exists=False),
    help="Output file path / directory for the shuffled FASTA",
)
@click.option(
    "--backend",
    type=click.Choice(["auto", "seqkit", "fast", "biopython"]),
    default="auto",
    help="Backend to use for shuffling (auto: seqkit if available, else fast)",
)
@click.option(
    "--seed", type=int, default=None, help="Random seed for shuffling (optional)"
)
@click.option(
    "--shuffle",
    "shuffle_flag",
    default="true",
    type=click.Choice(["true", "false"], case_sensitive=False),
    help="If false, skip shuffling and symlink (or copy) input to output.",
)
@click.help_option("--help", "-h")
def shuffle(input: str, output: str, backend: str, seed: int, shuffle_flag: str):
    """Shuffle sequences in a FASTA file."""
    # Check if the input file is a .fasta / .fa file
    if not input.endswith((".fasta", ".fa")):
        raise click.ClickException("Input file must be a .fasta or .fa file")

    input_path = Path(input)
    output_path = Path(output)

    create_dirs(output_path)

    output_path = _output_file(input_path, output_path, "_shuffled.fasta")

    do_shuffle = shuffle_flag.lower() == "true"

    shuffler = FastaShuffler(
        input_path=input_path,
        output_path=output_path,
        seed=seed,
        do_shuffle=do_shuffle,
    )
    shuffler.shuffle()


@data.command("filter")
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input FASTA file to filter",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(exists=False),
    help="Output FASTA path for filtered sequences",
)
@click.option(
    "--min-seq-len",
    default=20,
    type=int,
    help="Minimum sequence length",
)
@click.option(
    "--max-seq-len",
    default=1024,
    type=int,
    help="Maximum sequence length",
)
@click.option(
    "--seqs-num",
    default=-1,
    type=int,
    help="Number of sequences to retrieve (-1 for all)",
)
@click.option(
    "--skip-n",
    default=0,
    type=int,
    help="Number of sequences to skip from the beginning of the input FASTA file",
)
@click.option(
    "--exclude-ids",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Optional path to a text file with one sequence ID per line; matching sequences are excluded.",
)
@click.help_option("--help", "-h")
def filter(
    input: str,
    output: str,
    min_seq_len: int,
    max_seq_len: int,
    seqs_num: int,
    skip_n: int,
    exclude_ids: str | None,
):
    """Filter sequences by length and optional max count."""
    input_path = Path(input)
    output_path = Path(output)

    create_dirs(output_path)

    output_path = _output_file(input_path, output_path, "_filtered.fasta")

    filterer = Filterer(
        input_path=input_path,
        output_path=output_path,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        seqs_num=seqs_num,
        exclude_ids=Path(exclude_ids) if exclude_ids else None,
    )
    filterer.filter()
    click.echo("Filtering completed successfully")


@data.command("check")
@click.option("--input", "-i", required=True,
              type=click.Path(exists=True, dir_okay=False), help="Input FASTA file")
@click.option("--output", "-o", required=True,
              type=click.Path(file_okay=False),
              help="Output directory for validation sidecars")
@click.option("--drop-invalid", default="false",
              type=click.Choice(["true", "false"], case_sensitive=False),
              help="If true, exit 0 even with invalid sequences and write invalid_ids.txt")
@click.help_option("--help", "-h")
def check(input: str, output: str, drop_invalid: str):
    """Validate FASTA sequences. Writes validation.json, invalid.fasta, invalid_ids.txt
    into the output directory. Exits non-zero if invalid sequences are found AND
    --drop-invalid=false; exits zero otherwise.
    """
    from nanoplm.data.preparation.checker import Checker

    drop = drop_invalid.lower() == "true"
    result = Checker(
        input_path=Path(input),
        output_dir=Path(output),
        drop_invalid=drop,
    ).check()

    if result.invalid_count > 0:
        msg = (
            f"Validation found {result.invalid_count:,} problematic sequence(s).\n"
            f"  Inspect: {Path(output) / 'invalid.fasta'}\n"
            f"  Summary: {Path(output) / 'validation.json'}\n"
        )
        if not drop:
            msg += (
                "\nTo exclude these sequences and continue the pipeline:\n"
                "    nanoplm data from-yaml --fix\n"
            )
            raise click.ClickException(msg)
        click.echo(msg, err=True)


@data.command("split")
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Filtered FASTA input file",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(exists=False),
    help="Output directory for the train/val split (train.fasta and val.fasta are written here)",
)
@click.option(
    "--val-ratio",
    default=0.1,
    type=float,
    help="Validation ratio",
)
@click.option("--val-count-file", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Take the val count from this val_split file instead of --val-ratio.")
@click.help_option("--help", "-h")
def split(
    input: str,
    output: str,
    val_ratio: float,
    val_count_file: str = None,
):
    """Split a filtered FASTA into train/val sets."""
    input_path = Path(input)
    output_path = Path(output)

    create_dirs(output_path)

    if not output_path.is_dir():
        raise click.ClickException(f"--output must be a directory, got: {output_path}")

    train_file_path = output_path / "train.fasta"
    val_file_path = output_path / "val.fasta"

    want = _read_val_count(val_count_file) if val_count_file else None
    if want is not None:
        # the native split takes int(total * ratio) records; aim halfway past the count
        val_ratio = min(1.0, (want["val"] + 0.5) / max(1, want["units"]))
    splitor = Splitor(
        input_file=input_path,
        train_file=train_file_path,
        val_file=val_file_path,
        val_ratio=val_ratio,
    )
    train_size, val_size = splitor.split()
    if want is not None and (val_size, train_size + val_size) != (want["val"], want["units"]):
        raise click.ClickException(
            f"split: {input_path} has {train_size + val_size:,} sequences and gave {val_size:,} "
            f"val; {val_count_file} expects {want['units']:,} and {want['val']:,}. Run "
            "`nanoplm data from-yaml` again")
    click.echo(
        f"Splitting completed successfully.\nTrain file: {train_file_path}\nVal file: {val_file_path}\nTrain size: {train_size}, Val size: {val_size}"
    )


@data.command("kd-embed")
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input file path where the dataset is saved",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(exists=False),
    help="Output HDF5 file path where the processed dataset will be saved.",
)
@click.option(
    "--teacher-model",
    required=False,
    default="prott5",
    type=click.Choice(["prott5"]),
    help="Teacher model to use",
)
@click.option(
    "--max-seq-len",
    default=1024,
    type=int,
    help="Maximum sequence length used for teacher embedding calculation, all sequences would be padded / truncated to this length",
)
@click.option(
    "--samples-per-shard",
    default=-1,
    type=int,
    help="Number of samples per shard file. Use -1 for a single file (no sharding).",
)
@click.option(
    "--batch-size",
    default=4,
    type=int,
    help="Batch size for teacher embedding calculation",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force overwrite existing output file",
    default=False,
)
@click.option(
    "--device",
    default=lambda: os.environ.get("NANOPLM_DEVICE", "auto"),
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    show_default="$NANOPLM_DEVICE or auto",
    help="Device for teacher embedding computation.",
)
@click.option(
    "--skip-n",
    default=0,
    type=int,
    help="Number of sequences to skip from the beginning of the input FASTA file before processing",
)
@click.help_option("--help", "-h")
def kd_embed(
    input: str,
    output: str,
    teacher_model: str,
    max_seq_len: int,
    samples_per_shard: int,
    batch_size: int,
    force: bool,
    device: str,
    skip_n: int,
):
    """Generate knowledge distillation datasets with teacher embeddings."""
    if samples_per_shard < -1 or samples_per_shard == 0:
        raise click.ClickException("samples-per-shard must be -1 (no sharding) or a positive number (not 0)")
    if batch_size < 1:
        raise click.ClickException("batch-size must be at least 1")
    if max_seq_len < 1:
        raise click.ClickException("max-seq-len must be at least 1")
    if skip_n < 0:
        raise click.ClickException("skip-n must be at least 0")

    input_path = Path(input)
    output_path = Path(output)

    create_dirs(output_path)

    if not output_path.is_dir() and not str(output_path).endswith(".h5"):
        raise click.ClickException("Output must end with .h5 or be a directory")
    output_path = _output_file(input_path, output_path, "_kd_dataset.h5")

    if teacher_model == "prott5":
        selected_teacher_model = ProtT5()
    else:
        raise click.ClickException(f"Teacher model {teacher_model} not supported")

    kd_data = SaveKDDataset(
        input_fasta=input_path,
        output_path=output_path,
        teacher=selected_teacher_model,
        mode="get_embeddings",
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        device=device,
        skip_n=skip_n,
        samples_per_shard=samples_per_shard,
        force=force,
    )
    result = kd_data.process_dataset()

    if isinstance(result, list):
        click.echo(
            f"Knowledge distillation dataset generation completed successfully. Created {len(result)} shard files:"
        )
        for path in result:
            click.echo(f"  {path}")
    else:
        click.echo(
            f"Knowledge distillation dataset generation completed successfully. Output: {result}"
        )


@data.command("pre-tokenize")
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input FASTA file path where the dataset is saved",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(exists=False),
    help="Output directory path where the processed binary shards will be saved.",
)
@click.option(
    "--max-seq-len",
    default=1024,
    type=int,
    help="Maximum sequence length, all sequences would be padded / truncated to this length",
)
@click.option(
    "--max-workers",
    default=lambda: int(os.environ.get("NANOPLM_MAX_WORKERS", "2")),
    type=int,
    show_default="$NANOPLM_MAX_WORKERS or 2",
    help="Number of parallel worker threads.",
)
@click.option(
    "--samples-per-shard",
    default=10000,
    type=int,
    help="Number of samples to save per binary shard",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force overwrite existing output file",
    default=False,
)
@click.option(
    "--register-tokens",
    default=0,
    type=click.IntRange(0, 2),
    show_default=True,
    help="Number of register tokens (0, 1, or 2) to append after EOS. "
         "Register tokens act as learnable scratch space for the model.",
)
@click.help_option("--help", "-h")
def pre_tokenize(
    input: str,
    output: str,
    max_seq_len: int,
    max_workers: int,
    samples_per_shard: int,
    force: bool,
    register_tokens: int,
):

    if samples_per_shard < 1:
        raise click.ClickException("samples_per_shard must be at least 1")
    if max_seq_len < 1:
        raise click.ClickException("max-seq-len must be at least 1")
    if max_workers < 1 and max_workers != -1:
        raise click.ClickException("max_workers must be either -1 (use all available cores) or at least 1")

    input_path = Path(input)
    output_dir = Path(output)
    create_dirs(output_dir)

    tokenizer = ProtModernBertTokenizer(num_register_tokens=register_tokens)

    if register_tokens > 0:
        click.echo(f"Using {register_tokens} register token(s) after EOS")
    click.echo("Creating binary shards for pretraining")
    saver = ShardWriter(
        fasta_path=str(input_path),
        tokenizer=tokenizer,
        max_length=max_seq_len,
        output_dir=str(output_dir),
        samples_per_shard=samples_per_shard,
        max_workers=max_workers,
        force=force,
    )
    shards = saver.create_shards()

    click.echo(
        "Pretraining shard generation complete:\n"
        f"  shards: {len(shards)} -> {output_dir}\n"
    )

@data.command("kd-reshard")
@click.option(
    "--input-file",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input H5 file to shard",
)
@click.option(
    "--n-shards", "-n", required=True, type=int, help="Number of shard files to create"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=False),
    default=None,
    help="Output directory for shard files (defaults to same directory as input file)",
)
@click.option(
    "--total-sequences",
    "-t",
    type=int,
    default=None,
    help="Total number of sequences (if known, skips counting for faster start)",
)
@click.help_option("--help", "-h")
def kd_reshard(input_file: str, n_shards: int, output_dir: str, total_sequences: int):
    """Shard a large H5 file into smaller files for better performance.

    Examples:
        # Basic sharding (will count sequences first)
        nanoplm data kd-reshard --input-file train.h5 --n-shards 33

        # Fast sharding (skips counting if you know the sequence count)
        nanoplm data kd-reshard --input-file train.h5 --n-shards 33 --total-sequences 12500000

    This will create train_shard_0.h5, train_shard_1.h5, ..., train_shard_32.h5
    """

    shard_paths = shard_h5_file(
        input_h5_path=input_file,
        n_sharded_files=n_shards,
        output_dir=output_dir,
        total_sequences=total_sequences,
    )

    click.echo(f"Successfully created {len(shard_paths)} shard files:")
    for path in shard_paths:
        click.echo(f"  {path}")


# ---------------------------------------------------------------------------
# Cluster subsampling stages (hidden; run by from-yaml through DVC)
# ---------------------------------------------------------------------------

@data.command("get-cluster-xml", hidden=True)
@click.option("--url", required=True, type=str)
@click.option("--output", "-o", required=True, type=click.Path(dir_okay=False))
def get_cluster_xml(url: str, output: str):
    """Internal: fetch uniref50.xml.gz, or reuse a copy already at --output."""
    from nanoplm.data.preparation.cluster_pipeline import fetch_cluster_xml
    out = Path(output)
    if out.exists() and out.stat().st_size > 0:
        click.echo(f"Reusing {out}")
        return
    fetch_cluster_xml(out, url=url)
    click.echo(f"Downloaded to: {out}")


@data.command("cluster-members", hidden=True)
@click.option("--source", required=True,
              type=click.Choice(["uniref_xml", "mmseqs", "tsv", "headers"]))
@click.option("--fasta", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", required=True, type=click.Path(dir_okay=False))
@click.option("--level", type=click.Choice(["uniref90", "uniref100"]), default=None)
@click.option("--xml", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--tsv", type=click.Path(exists=True, dir_okay=False), default=None,
              help="The representative<TAB>member file (--source tsv).")
@click.option("--identity", type=float, default=0.5)
@click.option("--coverage", type=float, default=0.8)
@click.option("--tmp-dir", type=click.Path(file_okay=False), default=None)
@click.option("--threads", type=int,
              default=lambda: int(os.environ.get("NANOPLM_MAX_WORKERS", "2")),
              show_default="$NANOPLM_MAX_WORKERS or 2")
def cluster_members(source, fasta, output, level, xml, tsv, identity, coverage, tmp_dir,
                    threads):
    """Internal: write the canonical cluster table (cluster_id<TAB>sequence_id)."""
    if source in ("tsv", "headers"):
        from nanoplm.data.preparation.cluster_import import (ClusterImportError,
                                                             import_members_table)
        try:
            stats = import_members_table(source, Path(output), fasta=Path(fasta), tsv=tsv)
        except (ClusterImportError, OSError, EOFError) as e:
            raise click.ClickException(str(e)) from None
        click.echo(f"Wrote {output}: {stats['clusters']:,} clusters, "
                   f"{stats['members']:,} sequences")
        return
    from nanoplm.data.preparation.cluster_table import build_members_table
    stats = build_members_table(source, Path(output), fasta=Path(fasta), level=level,
                                xml=Path(xml) if xml else None, identity=identity,
                                coverage=coverage, tmp_dir=Path(tmp_dir) if tmp_dir else None,
                                threads=threads)
    click.echo(f"Wrote {output}: {stats['clusters']:,} clusters, {stats['members']:,} sequences")


@data.command("cluster-pool", hidden=True)
@click.option("--input", "-i", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", required=True, type=click.Path(file_okay=False))
@click.option("--max-seq-len", required=True, type=int)
@click.option("--samples-per-shard", required=True, type=int)
@click.option("--register-tokens", type=int, default=0)
@click.option("--max-workers", type=int,
              default=lambda: int(os.environ.get("NANOPLM_MAX_WORKERS", "2")),
              show_default="$NANOPLM_MAX_WORKERS or 2")
def cluster_pool(input, output, max_seq_len, samples_per_shard, register_tokens, max_workers):
    """Internal: pre-tokenize the filtered member FASTA into the pool (FASTA order)."""
    from nanoplm.data.preparation.cluster_pipeline import build_pool
    if max_workers < 1 and max_workers != -1:
        raise click.ClickException("max_workers must be either -1 (use all available cores) or at least 1")
    build_pool(Path(input), Path(output), max_seq_len, samples_per_shard, register_tokens,
               max_workers)
    click.echo(f"Wrote pool shards to {output}")


@data.command("cluster-index", hidden=True)
@click.option("--members-tsv", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--fasta", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--pool", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--index-dir", required=True, type=click.Path(file_okay=False))
@click.option("--val-dir", required=True, type=click.Path(file_okay=False))
@click.option("--max-seq-len", required=True, type=int)
@click.option("--val-ratio", type=float, default=None)
@click.option("--val-count-file", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Take the val cluster count from this val_split file instead of --val-ratio.")
@click.option("--samples-per-shard", required=True, type=int)
@click.option("--seqs-num", type=int, default=-1)
@click.option("--register-tokens", type=int, default=0)
@click.option("--cluster-xml", type=click.Path(dir_okay=False), default=None,
              help="The UniProt XML members.tsv came from, named in a release-mismatch error.")
def cluster_index_cmd(members_tsv, fasta, pool, index_dir, val_dir, max_seq_len, val_ratio,
                      val_count_file, samples_per_shard, seqs_num, register_tokens, cluster_xml):
    """Internal: build the cluster index and the validation gathers."""
    from nanoplm.data.preparation.cluster_pipeline import build_index_and_val
    if (val_ratio is None) == (val_count_file is None):
        raise click.ClickException("cluster-index: give --val-ratio or --val-count-file")
    want = _read_val_count(val_count_file) if val_count_file else None
    if want is not None:
        # draw_val_clusters holds out round(clusters * ratio): exactly the count
        val_ratio = want["val"] / want["units"]
    man = build_index_and_val(Path(fasta), Path(pool), Path(members_tsv), Path(index_dir),
                              Path(val_dir), max_seq_len, val_ratio, samples_per_shard,
                              seqs_num, register_tokens,
                              cluster_xml=Path(cluster_xml) if cluster_xml else None)
    c = man["counts"]
    if want is not None and (c["val_clusters"], c["clusters_in_universe"]) != (
            want["val"], want["units"]):
        raise click.ClickException(
            f"cluster-index: {c['clusters_in_universe']:,} clusters and {c['val_clusters']:,} "
            f"val; {val_count_file} expects {want['units']:,} and {want['val']:,}. Run "
            "`nanoplm data from-yaml` again")
    click.echo(f"Cluster index: {c['train_clusters']:,} train / {c['val_clusters']:,} val "
               f"clusters over {c['pool_rows']:,} pool sequences")


def _read_val_count(path) -> dict:
    from nanoplm.data.preparation.val_split import read_json
    got = read_json(path)
    if not (isinstance(got.get("val"), int) and isinstance(got.get("units"), int)
            and 0 <= got["val"] <= got["units"]):
        raise click.ClickException(f"{path}: not a val_split count file; run "
                                   "`nanoplm data from-yaml` again")
    return got


@data.command("count-units", hidden=True)
@click.option("--input", "-i", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", required=True, type=click.Path(dir_okay=False))
@click.option("--members-tsv", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Count the clusters of this table the FASTA holds, not its sequences.")
@click.option("--max-clusters", type=int, default=-1, help="Cap on the cluster count; -1 = none.")
def count_units_cmd(input, output, members_tsv, max_clusters):
    """Internal: count a source's sequences or clusters for the val split."""
    from nanoplm.data.preparation import val_split as vs
    if members_tsv:
        n, kind = vs.count_universe_clusters(input, members_tsv, max_clusters), vs.CLUSTER
    else:
        n, kind = vs.count_fasta_records(input), vs.PLAIN
    vs.write_units(output, n, kind)
    click.echo(f"{input}: {n:,} {kind}")


@data.command("val-split", hidden=True)
@click.option("--val-ratio", type=float, default=None)
@click.option("--val-size", type=int, default=None)
@click.option("--source", "sources", multiple=True, required=True, type=(str, str, float, str),
              help="NAME UNITS_JSON WEIGHT OUT_JSON, once per built source.")
def val_split_cmd(val_ratio, val_size, sources):
    """Internal: share the val size out over the sources by weight."""
    from nanoplm.data.preparation import val_split as vs
    pool = {}
    for name, units, weight, _ in sources:
        got = vs.read_json(units)
        pool[name] = (int(got["units"]), got["kind"], weight)
    try:
        counts = vs.val_counts(pool, val_ratio, val_size)
    except vs.ValSplitError as e:
        raise click.ClickException(str(e)) from None
    for name, units, _, out in sources:
        vs.write_val_count(out, counts[name], *pool[name][:2])
        click.echo(f"{name}: {counts[name]:,} of {pool[name][0]:,} {pool[name][1]} for val")


@data.command("get-yaml")
@click.argument("output", required=False,
                type=click.Path(file_okay=False, writable=True, resolve_path=True))
@click.option("--distillation", is_flag=True, default=False,
              help="Emit distillation_data.yaml instead of data.yaml")
@click.option("--subsampling", is_flag=True, default=False,
              help="Emit a data.yaml set up for cluster subsampling")
@click.option("--force", is_flag=True, default=False,
              help="Overwrite existing files")
@click.help_option("--help", "-h")
def get_yaml(output, distillation, subsampling, force):
    """Generate template data.yaml + dvc.yaml for the data pipeline."""
    from nanoplm.cli.templates import load_template
    from nanoplm.cli._dvc_yaml_gen import generate_dvc_yaml
    import yaml

    if distillation and subsampling:
        raise click.UsageError("--subsampling and --distillation cannot be used together: "
                               "cluster subsampling is for pretraining data. Pick one")
    output_path = Path(output) if output else Path.cwd()
    create_dirs(output_path)

    if distillation:
        params_filename = "distillation_data.yaml"
        params_content = load_template("data_distill")
        kind = "distillation"
    else:
        params_filename = "data.yaml"
        params_content = load_template("data_pretrain_subsampling" if subsampling
                                       else "data_pretrain")
        kind = "pretrain"

    params_path = output_path / params_filename
    dvc_path = output_path / "dvc.yaml"

    if not force:
        existing = [p for p in (params_path, dvc_path) if p.exists()]
        if existing:
            raise click.ClickException(
                "File(s) already exist: " + ", ".join(str(p) for p in existing)
                + ". Use --force to overwrite."
            )

    params_path.write_text(params_content, encoding="utf-8")
    parsed = yaml.safe_load(params_content)
    dvc_content = generate_dvc_yaml(
        config=parsed,
        kind=kind,
        params_filename=params_filename,
        base_dir=output_path,
    )
    dvc_path.write_text(dvc_content, encoding="utf-8")
    click.echo(
        f"Wrote {params_filename} and dvc.yaml in {output_path}.\n"
        f"Edit {params_filename} and run `nanoplm data from-yaml` to execute."
    )


@data.command("from-yaml")
@click.help_option("--help", "-h")
@click.argument(
    "config",
    default=None,
    required=False,
    type=click.Path(dir_okay=False, readable=True),
)
@click.option(
    "--target",
    type=str,
    required=False,
    default=None,
    help="DVC stage to run, as named in dvc.yaml (e.g. split, or cluster_index with "
         "cluster subsampling; in a mix <source>-<step>, e.g. swissprot-split). Defaults "
         "to the full pipeline.",
)
@click.option(
    "--no-auto-init",
    is_flag=True,
    default=False,
    help="Disable automatic 'dvc init' if the repo is not initialized.",
)
@click.option(
    "--force",
    "force_repro",
    is_flag=True,
    default=False,
    help="Pass -f/--force to 'dvc repro' (ignore unchanged).",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Run 'dvc repro' with -v for verbose output.",
)
@click.option(
    "--fix",
    is_flag=True,
    default=False,
    help="Set drop_invalid_sequences=true in config before running.",
)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help="Print the DVC plan without executing (dvc repro --dry --allow-missing).",
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default="info",
    help="Log level for nanoplm output.",
)
def from_yaml(
    config: str | None,
    target: str | None,
    no_auto_init: bool,
    force_repro: bool,
    verbose: bool,
    fix: bool,
    dry_run: bool,
    log_level: str,
):
    """Regenerate dvc.yaml from config and run the DVC data pipeline.

    Reads data.yaml (or distillation_data.yaml), validates the config,
    regenerates dvc.yaml to match the current settings, then runs dvc repro.
    """
    import yaml as _yaml
    from nanoplm.utils import set_log_level
    from nanoplm.cli._dvc_yaml_gen import generate_dvc_yaml

    set_log_level(log_level)

    config_path = _resolve_config_path(config)
    cwd = config_path.parent

    try:
        params = read_yaml(str(config_path))
    except Exception as e:
        raise click.ClickException(f"Failed to read {config_path}: {e}") from e

    data_params = params.get("data_params") or {}
    kind = _detect_kind(data_params, config_path.name)
    _reject_legacy_pipeline_mode(data_params)

    cfg = None
    if kind == "pretrain":
        cfg = _read_pretrain_config(config_path)
        params = cfg.raw
        plans = _validate_sources(cfg, cwd, config_path)
    else:
        _validate_source_fields(params.get("data_dirs", {}))
        _validate_pipeline_fields(params.get("data_dirs", {}), data_params, kind, config_path)

    if fix:
        _flip_drop_invalid(config_path, True)
        click.echo(f"Set {config_path.name}: drop_invalid_sequences=true")
        # Re-read after flip so dvc.yaml is generated with updated value
        if kind == "pretrain":
            cfg = _read_pretrain_config(config_path, quiet=True)
            params = cfg.raw
        else:
            params = read_yaml(str(config_path))
            data_params = params.get("data_params", {})

    if kind == "pretrain":
        _print_plan(cfg, plans, cwd, config_path)
        # one value for every source: a mix refuses a per-source max_workers
        max_workers = next(iter(cfg.sources.values())).params.get("max_workers")
    else:
        max_workers = data_params.get("max_workers")

    # Regenerate dvc.yaml BEFORE any DVC subprocess call
    dvc_yaml_path = cwd / "dvc.yaml"
    dvc_content = generate_dvc_yaml(
        config=params,
        kind=kind,
        params_filename=config_path.name,
        base_dir=cwd,
    )
    _warn_orphaned_outputs(cwd, dvc_content, params)
    dvc_yaml_path.write_text(dvc_content, encoding="utf-8")
    # click.echo, not logger.info: this must still print at --log-level error.
    click.echo(f"Regenerated {dvc_yaml_path}")

    dvc_dir = cwd / ".dvc"
    # --dry-run is a read-only preview and must not mutate the directory by
    # running 'dvc init' on its own; if that turns out to be required, the
    # 'dvc repro' call below fails with an explanation instead.
    if not dvc_dir.exists() and not dry_run:
        if no_auto_init:
            raise click.ClickException(
                f"DVC is not initialized in {cwd}. Run 'dvc init' first or omit --no-auto-init."
            )
        try:
            init_cmd = _DVC + ["init", "-q"]
            if not inside_git_repo(cwd):
                init_cmd.append("--no-scm")
            elif is_git_subdir(cwd):
                init_cmd.append("--subdir")
            subprocess.run(init_cmd, cwd=str(cwd), check=True)
            # Link outputs into the cache instead of copying them: a UniRef90 cluster
            # run would otherwise keep a second ~160 GB copy. Linked outputs are
            # read-only; edit one only after `dvc unprotect <path>`. Not for the
            # data_dirs layout: its get_file symlinks local_fasta into the outputs, and
            # a linked cache would make the user's own file read-only.
            if cfg is not None and not cfg.legacy:
                subprocess.run(_DVC + ["config", "cache.type", _CACHE_TYPE], cwd=str(cwd),
                               check=True)
                click.echo(f"Initialized DVC repository in: {cwd} (cache.type {_CACHE_TYPE})")
            else:
                click.echo(f"Initialized DVC repository in: {cwd}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise click.ClickException(
                f"Failed to initialize DVC in {cwd}. Details: {e}"
            )

    # Build environment for repro
    device = data_params.get("device")
    repro_env = dict(os.environ)
    if max_workers is not None:
        repro_env["NANOPLM_MAX_WORKERS"] = str(max_workers)
    if device is not None:
        repro_env["NANOPLM_DEVICE"] = str(device)

    # dvc repro --dry needs an initialized repo, and --dry-run will not create
    # one, so say that here rather than letting dvc say it less helpfully first.
    if dry_run and not dvc_dir.exists():
        raise click.ClickException(
            f"--dry-run needs an initialized DVC repo and will not create one. "
            f"{cwd} has no .dvc/. Run 'nanoplm data from-yaml' once without "
            f"--dry-run, or 'dvc init', then retry."
        )

    if kind == "pretrain":
        _print_rebuild_line(cwd, dvc_content, repro_env, params,
                            list(cfg.sources) if cfg.is_mixture else None)

    # dvc repro --dry fails at a stage whose input an earlier stage has not written yet
    # (a pending download), so a dry run previews up to the stages that write it
    pending = _pending_stages(cwd, dvc_content, params) if dry_run and not target else []

    # Build repro command
    if dry_run:
        # --allow-missing would skip exactly the pending stages, whose outputs are missing
        cmd: list[str] = _DVC + ["repro", "--dry"] + ([] if pending else ["--allow-missing"])
    else:
        cmd = _DVC + ["repro"]
    if verbose:
        cmd.append("-v")
    if force_repro:
        cmd.append("-f")
    cmd += [target] if target else pending

    repro_target_desc = target or (f"up to {', '.join(pending)}" if pending else "full pipeline")
    click.echo(f"Running dvc repro ({repro_target_desc}){'  [dry-run]' if dry_run else ''}...")
    try:
        result = subprocess.run(cmd, cwd=str(cwd), env=repro_env)
        if result.returncode != 0:
            raise click.ClickException(f"dvc repro exited with code {result.returncode}")
    except FileNotFoundError:
        raise click.ClickException(
            "'dvc' command not found. Please install DVC (pip install dvc)."
        )

    if pending:
        click.echo(f"The dry run stops after {', '.join(pending)}: DVC cannot preview the "
                   "stages that read their outputs until those exist.")
    if not dry_run:
        _print_summary(cwd, params, kind, config_path, cfg)


# ---------------------------------------------------------------------------
# Private helpers for from-yaml
# ---------------------------------------------------------------------------

# reflink is left out: on GPFS / Lustre DVC would try it on every file first
_CACHE_TYPE = "hardlink,symlink"


def _resolve_config_path(config: str | None) -> Path:
    """Resolve the config argument to an absolute Path.

    Resolution order:
    1. If *config* is given and absolute → use as-is.
    2. If *config* is given and relative → resolve against cwd.
    3. If *config* is None → look for data.yaml then distillation_data.yaml in cwd.
    """
    cwd = Path.cwd()
    if config is not None:
        p = Path(config)
        if not p.is_absolute():
            p = (cwd / p).resolve()
        if not p.exists():
            raise click.ClickException(f"Config file not found: {p}")
        return p
    for candidate in ("data.yaml", "distillation_data.yaml"):
        p = cwd / candidate
        if p.exists():
            return p
    raise click.ClickException(
        "No config file found in current directory. "
        "Pass the path explicitly or run 'nanoplm data get-yaml' first."
    )


def _reject_legacy_pipeline_mode(data_params: dict) -> None:
    """Raise ClickException if the legacy pipeline_mode key is present."""
    if "pipeline_mode" in data_params:
        raise click.ClickException(
            "The 'pipeline_mode' field is no longer supported. "
            "Remove 'data_params.pipeline_mode' from your config file."
        )


def _validate_source_fields(data_dirs: dict) -> None:
    """Validate data_dirs source/output fields.

    - Exactly one of url / local_fasta must be set.
    - The chosen source must end with .fasta or .fa, optionally .gz or .zst.
    - extracted_fasta is always required.
    - compressed_fasta is required only when the source is compressed, and then ends with
      the source's .gz / .zst (extract picks the decompressor by that name).
      In plain-.fasta mode it is optional; warn if it's set anyway.
    """
    from nanoplm.data.preparation.data_config import (compression_suffix, is_compressed,
                                                      is_fasta_name, strip_compressed)

    url = data_dirs.get("url") or None
    local_fasta = data_dirs.get("local_fasta") or None
    if url and local_fasta:
        raise click.ClickException(
            "Set either data_dirs.url or data_dirs.local_fasta, not both."
        )
    if not url and not local_fasta:
        raise click.ClickException(
            "Either data_dirs.url or data_dirs.local_fasta must be set."
        )

    source = url if url else local_fasta
    compressed_source = is_compressed(source)
    # Extension whitelist (after stripping URL query/fragment).
    if not is_fasta_name(source):
        raise click.ClickException(
            "url/local_fasta must end with .fasta or .fa (optionally .gz or .zst), "
            f"got: {source}"
        )

    extracted = data_dirs.get("extracted_fasta") or None
    if not extracted:
        raise click.ClickException("data_dirs.extracted_fasta is required.")

    compressed = data_dirs.get("compressed_fasta") or None
    if compressed_source and not compressed:
        raise click.ClickException(
            "data_dirs.compressed_fasta is required when source is a .gz or .zst file."
        )
    suffix = compression_suffix(source)
    if compressed_source and not str(compressed).endswith(suffix):
        key = "url" if url else "local_fasta"
        raise click.ClickException(
            f"data_dirs.compressed_fasta must end with {suffix} like data_dirs.{key} "
            f"({source}): the extract stage picks the decompressor by this name. Set "
            f'compressed_fasta: "{strip_compressed(str(compressed)) + suffix}"'
        )
    if (not compressed_source) and compressed:
        click.echo("Warning: data_dirs.compressed_fasta is unused when the source is a plain "
                   ".fasta file; it can be removed.", err=True)


# data_dirs path keys that generate_dvc_yaml (_dvc_yaml_gen.py) substitutes as
# ${data_dirs.X} into every pipeline stage, for both kinds. url/local_fasta,
# extracted_fasta and compressed_fasta are covered by _validate_source_fields.
_DATA_DIRS_REQUIRED = [
    "validation_dir", "shuffled_fasta", "filtered_fasta", "splitted_fasta_dir", "output_dir",
]


def _expects(text: str):
    """Attach what a validator accepts, for the invalid-value error."""
    def wrap(fn):
        fn.expects = text
        return fn
    return wrap


def _int_at_least(lo: int):
    return _expects(f"an integer >= {lo}")(
        lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= lo)


def _int_in_range(lo: int, hi: int):
    return _expects(f"an integer from {lo} to {hi}")(
        lambda v: isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi)


@_expects("-1 (no sharding) or an integer >= 1")
def _shard_count(v) -> bool:
    """-1 (no sharding) or a positive shard size, matching kd-embed's own check."""
    return isinstance(v, int) and not isinstance(v, bool) and (v == -1 or v > 0)


@_expects("a number between 0 and 1, both excluded")
def _unit_float(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < 1


@_expects("an integer or null")
def _optional_int(v) -> bool:
    return v is None or (isinstance(v, int) and not isinstance(v, bool))


@_expects("true or false")
def _is_bool(v) -> bool:
    return isinstance(v, bool)


@_expects("a non-empty string")
def _nonempty_str(v) -> bool:
    return isinstance(v, str) and bool(v)


_PRETRAIN = frozenset({"pretrain"})
_DISTILLATION = frozenset({"distillation"})
_BOTH = _PRETRAIN | _DISTILLATION

# data_params keys generate_dvc_yaml substitutes as ${data_params.X}, one row
# per (field, kinds-that-need-it, validator). A field needed by both kinds with
# a kind-specific rule (samples_per_shard) gets one row per kind.
_DATA_PARAMS_TABLE = [
    ("drop_invalid_sequences", _BOTH, _is_bool),
    ("shuffle", _BOTH, _is_bool),
    ("shuffle_seed", _BOTH, _optional_int),
    ("min_seq_len", _BOTH, _int_at_least(0)),
    ("max_seq_len", _BOTH, _int_at_least(0)),
    ("max_seqs", _PRETRAIN, _int_at_least(-1)),
    ("seqs_num", _DISTILLATION, _int_at_least(-1)),
    ("val_ratio", _BOTH, _unit_float),
    ("samples_per_shard", _PRETRAIN, _int_at_least(1)),
    ("samples_per_shard", _DISTILLATION, _shard_count),
    ("register_tokens", _PRETRAIN, _int_in_range(0, 2)),
    ("teacher_model", _DISTILLATION, _nonempty_str),
    ("embed_calc_batch_size", _DISTILLATION, _int_at_least(1)),
]


def _validate_pipeline_fields(data_dirs: dict, data_params: dict, kind: str, config_path: Path) -> None:
    """Check every data_dirs/data_params key generate_dvc_yaml substitutes for *kind*.

    A missing or mistyped key here reaches DVC only as an unresolved
    ${...} template, so a typo would otherwise surface mid-`dvc repro`
    instead of before it runs.
    """
    for key in _DATA_DIRS_REQUIRED:
        if not data_dirs.get(key):
            raise click.ClickException(f"{config_path.name}: data_dirs.{key} is required.")

    for field, kinds, is_valid in _DATA_PARAMS_TABLE:
        if kind not in kinds:
            continue
        if field not in data_params:
            raise click.ClickException(f"{config_path.name}: data_params.{field} is required.")
        value = data_params[field]
        if not is_valid(value):
            raise click.ClickException(
                f"{config_path.name}: data_params.{field} has an invalid value: {value!r} "
                f"(expected {is_valid.expects})"
            )


def _validate_source_params(src, config_path: Path) -> None:
    """The pretraining data_params of one source, each named where data.yaml sets it."""
    for field, kinds, is_valid in _DATA_PARAMS_TABLE:
        if "pretrain" not in kinds:
            continue
        if field not in src.params:
            raise click.ClickException(f"{config_path.name}: data_params.{field} is required.")
        value = src.params[field]
        if field == "val_ratio" and value is None:      # val_size is set instead
            continue
        if not is_valid(value):
            raise click.ClickException(
                f"{config_path.name}: {src.key(field)} has an invalid value: {value!r} "
                f"(expected {is_valid.expects})")


def _read_pretrain_config(config_path: Path, quiet: bool = False):
    """data.yaml read strictly (duplicate and unknown keys are errors) into one source."""
    from nanoplm.data.preparation.data_config import (
        DataConfigError, load_strict, parse_data_config,
    )
    try:
        raw = load_strict(config_path)
        _reject_legacy_pipeline_mode(raw.get("data_params") or {})
        cfg = parse_data_config(raw, config_path.parent, config_path.name)
    except DataConfigError as e:
        msg = str(e)
        raise click.ClickException(msg if msg.startswith(config_path.name)
                                   else f"{config_path.name}: {msg}") from e
    cfg.raw = raw
    if not quiet:
        for w in cfg.warnings:
            click.echo(f"Warning: {w}", err=True)
    if cfg.legacy:
        _validate_source_fields(raw.get("data_dirs", {}))
        _validate_pipeline_fields(raw.get("data_dirs", {}), {}, "none", config_path)
    for src in cfg.sources.values():
        _validate_source_params(src, config_path)
        if not cfg.legacy and not src.is_url:
            local = _resolve_path(src.fasta, config_path.parent)
            if not local.is_file():
                raise click.ClickException(
                    f"{config_path.name}: sources.{src.name}.fasta: {src.fasta} not found (a "
                    "relative path is relative to data.yaml's directory)")
    return cfg


def orphaned_outputs(cwd: Path, dvc_content: str, params: dict) -> list:
    """Outputs of an earlier build that this dvc.yaml no longer writes or reads, still on
    disk: what dvc.lock records (DVC keeps a removed stage's entry) minus every path of
    the new stages. Covers a removed or renamed source, a dropped cluster block, a
    members switch and the move between one source and a mix. Nothing is deleted."""
    import yaml as _yaml
    lock = cwd / "dvc.lock"
    if not lock.is_file():
        return []
    try:
        stages = (_yaml.safe_load(lock.read_text(encoding="utf-8")) or {}).get("stages") or {}
    except _yaml.YAMLError:
        return []
    graph = _stage_graph(dvc_content, params)
    used = [str(_resolve_path(p, cwd)) for deps, outs in graph.values() for p in deps + outs]
    found = []
    for stage in stages.values():
        for out in (stage or {}).get("outs") or []:
            path = _resolve_path(os.path.normpath(out["path"]), cwd)
            if (path.exists() and path not in found
                    and not any(_overlaps(str(path), u) for u in used)):
                found.append(path)
    return found


def _warn_orphaned_outputs(cwd: Path, dvc_content: str, params: dict) -> None:
    paths = orphaned_outputs(cwd, dvc_content, params)
    if not paths:
        return
    # the .gitignore lines DVC wrote for these paths (one per output, "/<name>")
    gitignores = sorted({str(p.parent / ".gitignore") for p in paths
                         if (p.parent / ".gitignore").is_file() and f"/{p.name}" in
                         (p.parent / ".gitignore").read_text().split()})
    click.echo("Warning: an earlier build left these outputs, which the new dvc.yaml no "
               "longer uses; nothing deletes them:", err=True)
    for p in paths:
        click.echo(f"  {p}", err=True)
    if gitignores:
        click.echo("and their lines in " + ", ".join(gitignores), err=True)
    # a linked cache keeps the data on disk until DVC drops its copy too
    click.echo("To free the disk, delete them and run `dvc gc --workspace` (it removes every "
               "cached file the current dvc.lock does not use).", err=True)


def _validate_sources(cfg, cwd: Path, config_path: Path) -> dict:
    """Check every source before DVC runs: {name: its cluster plan, or None}."""
    out = {src.name: _validate_cluster_block(src, config_path) for src in cfg.sources.values()}
    regs = {s.name: s.params["register_tokens"] for s in cfg.sources.values()}
    if cfg.is_mixture and len(set(regs.values())) > 1:
        got = ", ".join(f"{n} {v}" for n, v in regs.items())
        raise click.ClickException(f"{config_path.name}: the sources of a mix must use the "
                                   f"same register_tokens; got {got}")
    return out


def _validate_cluster_block(src, config_path: Path):
    """Resolve the source's cluster block now, so a bad block fails before DVC runs."""
    from nanoplm.data.preparation.cluster_config import ClusterConfigError, resolve_cluster_plan
    from nanoplm.data.preparation.cluster_table import MMSEQS_INSTALL_HINT, find_mmseqs
    if src.legacy:
        return None
    try:
        plan = resolve_cluster_plan(src, config_path.parent)
    except ClusterConfigError as e:
        raise click.ClickException(f"{config_path.name}: {e}") from e
    if plan is None:
        return None
    workers = src.params.get("max_workers")
    if not (isinstance(workers, int) and not isinstance(workers, bool)
            and (workers == -1 or workers >= 1)):
        raise click.ClickException(
            f"{config_path.name}: {src.key('max_workers')} must be -1 (all cores) or at "
            f"least 1 with cluster_subsampling (sources.{src.name}), got {workers!r}")
    for n in plan.notes:
        click.echo(n)
    for w in plan.warnings:
        click.echo(f"Warning: {w}", err=True)
    if plan.cluster_source == "mmseqs" and find_mmseqs() is None:
        uniprot = (f" Or set sources.{src.name}.cluster_subsampling.clusters: uniprot to "
                   "use UniProt's clusters, which need no MMseqs2."
                   if plan.members_level in ("uniref90", "uniref100") else
                   " Or give the clusters yourself: clusters: <file>.tsv of "
                   "representative<TAB>member lines.")
        raise click.ClickException(
            f"{config_path.name}: sources.{src.name}: these members are clustered with "
            f"MMseqs2, but `mmseqs` was not found. {MMSEQS_INSTALL_HINT}{uniprot}")
    return plan


def _detect_kind(data_params: dict, filename: str) -> str:
    """Detect pipeline kind from the config filename or data_params keys."""
    if "distillation" in filename:
        return "distillation"
    if "teacher_model" in data_params or "embed_calc_batch_size" in data_params:
        return "distillation"
    return "pretrain"


def _flip_drop_invalid(config_path: Path, value: bool) -> None:
    """Edit drop_invalid_sequences in place (every occurrence, so a per-source
    override flips too), keeping the file's comments and layout. A data.yaml that
    leaves it out (it defaults to false) gets it added under data_params."""
    import re
    text = config_path.read_text()
    new_val = "true" if value else "false"
    pattern = re.compile(r"^(\s*drop_invalid_sequences\s*:\s*)(true|false)(\s*(?:#.*)?)$", re.M)
    new_text, n = pattern.subn(rf"\g<1>{new_val}\g<3>", text)
    if n == 0 and not re.search(r"^\s*drop_invalid_sequences\s*:", text, re.M):
        line = f"drop_invalid_sequences: {new_val}"
        section = re.search(r"^data_params:[ \t]*(#.*)?$", text, re.M)
        if section:
            # the indent of data_params' first field, so the mapping stays valid
            first = re.compile(r"^([ \t]+)[A-Za-z_]", re.M).search(text, section.end())
            indent = first.group(1) if first else "  "
            new_text = text[:section.end()] + f"\n{indent}{line}" + text[section.end():]
            n = 1
        elif not re.search(r"^data_params\s*:", text, re.M):
            new_text = text.rstrip("\n") + f"\n\ndata_params:\n  {line}\n"
            n = 1
    if n == 0:
        raise click.ClickException(
            "Couldn't find drop_invalid_sequences in config to flip."
        )
    config_path.write_text(new_text)


def _size(path: Path) -> str:
    n = path.stat().st_size
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"


def _date(path: Path) -> str:
    """When a file was written (a download: which release it is)."""
    import datetime
    return datetime.date.fromtimestamp(path.stat().st_mtime).isoformat()


def _file_cell(value: str, download, cwd: Path, shared=None, owner: str = "") -> str:
    """Where one input file comes from: a URL to download (``download`` is where it
    goes), a copy already downloaded, or a local file read in place. ``shared`` maps
    each download already shown to its source, so a shared one is listed once."""
    from nanoplm.data.preparation.data_config import is_url
    if is_url(value):
        on_disk = _resolve_path(download, cwd) if download else None
        name = Path(download).name if download else value
        if shared is not None and download:
            if download in shared:
                return f"{name} (same download as {shared[download]})"
            shared[download] = owner
        if on_disk is not None and on_disk.is_file():
            return f"{name} (reuse, {_size(on_disk)}, {_date(on_disk)})"
        return f"{name} (url)"
    on_disk = _resolve_path(value, cwd)
    return f"{value} (local, {_size(on_disk)})" if on_disk.is_file() else f"{value} (local)"


def _fasta_cell(src, plan, cwd: Path, shared=None) -> str:
    """The files the stages read: the fasta, or the members FASTA when the members
    differ from it (the fasta is then never fetched), plus the cluster XML."""
    if plan is None or plan.members_from_source:
        cell = _file_cell(src.fasta, src.paths.get(
            "compressed_fasta" if src.is_compressed else "extracted_fasta"), cwd,
            shared, src.name)
    else:
        cell = _file_cell(plan.members, str(plan.members_compressed or plan.members_fasta),
                          cwd, shared, src.name)
    if plan is not None and plan.xml_url:
        cell += " + " + _file_cell(plan.xml_url, str(plan.xml_path), cwd, shared, src.name)
    if plan is not None and plan.tsv_path:
        cell += " + " + _file_cell(plan.tsv_url or plan.clusters,
                                   str(plan.tsv_path) if plan.tsv_url else None, cwd, shared,
                                   src.name)
    return cell


def _max_seqs_cell(max_seqs: int, clusters: bool) -> str:
    """The max_seqs cap; it counts clusters for a cluster-subsampled source."""
    cap = "all" if max_seqs == -1 else f"{max_seqs:,}"
    return f"{cap} clusters" if clusters else cap


def plan_table(cfg, plans: dict, cwd: Path, config_path: Path) -> str:
    """The pre-run summary: one row per source, before any DVC call. ``plans`` maps
    each source to its cluster plan (None = off). A mix adds the weight column and lists
    a download shared by sources once."""
    out = _resolve_path(cfg.output_dir, cwd)
    n = len(cfg.sources)
    mix = cfg.is_mixture
    lines = [f"{config_path.name}: {n} source{'s' if n > 1 else ''} -> {out}"]
    if mix:
        lines.append("weight = share of training sequences")
    rows = [("source",) + (("weight",) if mix else ())
            + ("fasta", "max seqs", "len", "val", "cluster subsampling")]
    starred = False
    shared = {} if mix else None
    for src in cfg.sources.values():
        plan = plans.get(src.name)
        over = set(src.overrides)
        starred |= bool(over & {"max_seqs", "min_seq_len", "max_seq_len"})

        def mark(text, *fields):
            return text + ("*" if over & set(fields) else "")

        p = src.params
        seqs = _max_seqs_cell(p["max_seqs"], plan is not None)
        rows.append((src.name,) + ((f"{src.weight:g}",) if mix else ()) + (
            _fasta_cell(src, plan, cwd, shared), mark(seqs, "max_seqs"),
            mark(f"{p['min_seq_len']}-{p['max_seq_len']}", "min_seq_len", "max_seq_len"),
            "by weight" if src.val_from_split else f"{p['val_ratio']:g}",
            plan.source_line() if plan is not None else "off",
        ))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]) - 1)]
    for r in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(r, widths)) + "  " + r[-1])
    if starred:
        lines.append("* = set on this source")
    if cfg.val_from_split:
        field, value = cfg.val
        what = (f"{value:g} of the pool" if field == "val_ratio"
                else f"{value:,} sequences / clusters")
        lines.append(f"val = data_params.{field}: {what}, shared out by weight")
    return "\n".join(lines)


def _print_plan(cfg, plans: dict, cwd: Path, config_path: Path) -> None:
    click.echo(plan_table(cfg, plans, cwd, config_path))


def _stage_graph(dvc_content: str, params: dict) -> dict:
    """{stage: (deps, outs)} in dvc.yaml order, a foreach stage expanded per item
    (``pre-tokenize@train``). ${...} references are filled in from data.yaml."""
    import os.path
    import re
    import yaml as _yaml

    def lookup(ref: str, item: dict):
        node = {"item": item, **params}
        for part in ref.split("."):
            if not isinstance(node, dict) or part not in node:
                return "${" + ref + "}"
            node = node[part]
        return str(node)

    def paths(entries, item):
        out = []
        for e in entries or []:
            for p in (e if isinstance(e, dict) else [e]):
                p = str(p)
                for _ in range(3):      # an item value may hold ${data_dirs.X} itself
                    p = re.sub(r"\$\{([^}]+)\}", lambda m: lookup(m.group(1), item), p)
                out.append(os.path.normpath(p))
        return out

    graph = {}
    for name, stage in (_yaml.safe_load(dvc_content).get("stages") or {}).items():
        if "foreach" in stage:
            for key, item in stage["foreach"].items():
                graph[f"{name}@{key}"] = (paths(stage["do"].get("deps"), item),
                                          paths(stage["do"].get("outs"), item))
        else:
            graph[name] = (paths(stage.get("deps"), {}), paths(stage.get("outs"), {}))
    return graph


def _overlaps(a: str, b: str) -> bool:
    """One path is the other or inside it (``split/train.fasta`` in ``split``)."""
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def _with_downstream(changed: set, graph: dict) -> set:
    """``changed`` plus every stage that reads, directly or not, what one of them writes."""
    out = set(changed)
    todo = list(changed)
    while todo:
        outs = graph.get(todo.pop(), ((), ()))[1]
        for name, (deps, _) in graph.items():
            if name not in out and any(_overlaps(d, o) for d in deps for o in outs):
                out.add(name)
                todo.append(name)
    return out


def _pending_stages(cwd: Path, dvc_content: str, params: dict) -> list:
    """Stages whose inputs are all there but which have not written an output another
    stage reads (e.g. a download not fetched yet)."""
    graph = _stage_graph(dvc_content, params)
    reads = [d for deps, _ in graph.values() for d in deps]

    def missing(p: str) -> bool:
        return not _resolve_path(p, cwd).exists()

    return [name for name, (deps, outs) in graph.items()
            if not any(missing(d) for d in deps)
            and any(missing(o) and any(_overlaps(o, d) for d in reads) for o in outs)]


def _print_rebuild_line(cwd: Path, dvc_content: str, env: dict, params: dict,
                        sources=None) -> None:
    """Which stages dvc repro will run. Without a dvc.lock that is all of them; otherwise
    the stages `dvc status` reports changed (it reuses DVC's file-hash cache after the
    first run) and every stage downstream of one, since their inputs will change too.
    ``sources`` (the names of a mix) turns the stage list into one summary per source."""
    import json
    graph = _stage_graph(dvc_content, params)
    names = list(graph)
    if not (cwd / "dvc.lock").exists():
        click.echo("Rebuild: all (first run)")
        return
    try:
        r = subprocess.run(_DVC + ["status", "--json"], cwd=str(cwd), env=env,
                           capture_output=True, text=True, timeout=_STATUS_TIMEOUT)
        changed = set(json.loads(getattr(r, "stdout", "") or "{}"))
        if getattr(r, "returncode", 1) != 0:
            return
    except (subprocess.TimeoutExpired, ValueError, OSError):
        click.echo("Rebuild: unknown (dvc status was slow; dvc repro decides)")
        return
    changed = _with_downstream(changed & set(names), graph)
    rebuild = [n for n in names if n in changed]
    fresh = [n for n in names if n not in changed]
    if not rebuild:
        click.echo("Up to date: all stages")
        return
    if sources:
        click.echo(_rebuild_summary(names, changed, sources))
        return
    # dvc status cannot tell which of these DVC's run cache already holds (repro then
    # restores the outputs instead of running the stage), hence "may"
    click.echo(f"May rerun (DVC skips a stage its run cache already has): "
               f"{', '.join(rebuild)}." + (f" Up to date: {', '.join(fresh)}." if fresh else ""))


_STATUS_TIMEOUT = 120
_RUN_CACHE_NOTE = "(DVC skips a stage its run cache already has.)"


def _rebuild_summary(names: list, changed: set, sources) -> str:
    """'Rebuild: swissprot: up to date; uniref50: 6 of 7 stages may rerun; mixture: may
    rerun.' A mix's stages are <source>-<step>; the shared download, extract and check
    stages are grouped as downloads."""
    groups: dict = {}
    for n in names:
        head = n.split("-", 1)[0]
        group = n if n in ("val_split", "mixture") else head if head in sources else "downloads"
        groups.setdefault(group, []).append(n)
    order = (["downloads"] if "downloads" in groups else []) + \
        [s for s in sources if s in groups] + ["val_split", "mixture"]
    parts = []
    for g in order:
        stages = groups.get(g, [])
        k = sum(n in changed for n in stages)
        if not stages:
            continue
        if k == 0:
            parts.append(f"{g}: up to date")
        elif len(stages) == 1:
            parts.append(f"{g}: may rerun")
        else:
            parts.append(f"{g}: {k} of {len(stages)} stages may rerun")
    return f"Rebuild: {'; '.join(parts)}. {_RUN_CACHE_NOTE}"


def _print_summary(cwd: Path, params: dict, kind: str, config_path: Path, cfg=None) -> None:
    """Print a human-readable summary using .data_manifest if present."""
    import yaml as _yaml
    if cfg is not None and cfg.is_mixture:
        _print_mixture_summary(cwd, cfg)
        return
    if cfg is not None:
        output_dir_value, src_value = cfg.output_dir, cfg.source.fasta
    else:
        data_dirs = params.get("data_dirs") or {}
        output_dir_value = data_dirs.get("output_dir")
        src_value = data_dirs.get("url") or data_dirs.get("local_fasta")
    if not output_dir_value:
        raise click.ClickException(
            f"{config_path.name}: output_dir is missing; cannot locate the pipeline output."
        )
    output_dir = Path(output_dir_value)
    manifest_path = (cwd / output_dir / ".data_manifest") if not output_dir.is_absolute() \
        else (output_dir / ".data_manifest")
    if not manifest_path.exists():
        click.echo("Pipeline completed (no manifest found).")
        return
    try:
        m = _yaml.safe_load(manifest_path.read_text()) or {}
    except _yaml.YAMLError as e:
        raise click.ClickException(f"{manifest_path}: failed to parse manifest: {e}")
    title = "Pretraining" if kind == "pretrain" else "Distillation"
    if m.get("cluster_index_dir"):
        from nanoplm.data.preparation.cluster_config import resolve_cluster_plan
        plan = resolve_cluster_plan(cfg.source, cwd) if cfg is not None else None
        source = f"  Cluster source:  {plan.clusters_text()}\n" if plan else ""
        click.echo(
            f"\n{title} data pipeline complete (cluster subsampling).\n\n"
            f"  Members:         {plan.members if plan else src_value}\n"
            f"{source}"
            f"  Pool sequences:  {m.get('pool_sequences')}\n"
            f"  {_clusters_line(m)}\n"
            f"    Train:         {m.get('train_sequences')}\n"
            f"    Val:           {m.get('val_sequences')}\n"
            f"  Manifest:        {manifest_path}\n"
            f"{_dataset_dir_line(manifest_path, kind)}"
        )
        return
    click.echo(
        f"\n{title} data pipeline complete.\n\n"
        f"  Source:          {src_value}\n"
        f"  Total sequences: {m.get('seqs_num')}\n"
        f"    Train:         {m.get('train_sequences')}\n"
        f"    Val:           {m.get('val_sequences')}\n"
        f"  Manifest:        {manifest_path}\n"
        f"{_dataset_dir_line(manifest_path, kind)}"
    )


def _clusters_line(m: dict) -> str:
    """'UniRef50 clusters: 265 (members from UniRef90)', 'MMseqs2 clusters:  40', or
    'Clusters:           40 (from the CL: tags)': every label padded to one width."""
    level, n = m.get("cluster_level"), m.get("seqs_num")
    if level in ("uniref90", "uniref100"):
        label, note = "UniRef50 clusters:", f" (members from UniRef{level[len('uniref'):]})"
    elif level in ("tsv", "headers"):
        label = "Clusters:"
        note = " (from the cluster file)" if level == "tsv" else " (from the CL: tags)"
    else:
        label, note = "MMseqs2 clusters:", ""
    return f"{label:<19}{n}{note}"


def _dataset_dir_line(manifest_path: Path, kind: str) -> str:
    if kind != "pretrain":
        return ""
    return f"  Set pretrain.yaml's dataset_dir to {manifest_path.parent.resolve()}.\n"


def _print_mixture_summary(cwd: Path, cfg) -> None:
    from nanoplm.data.infra.manifest import read_manifest
    folder = _resolve_path(cfg.output_dir, cwd)
    try:
        m = read_manifest(folder)
    except (FileNotFoundError, ValueError):
        click.echo("Pipeline completed (no mixture manifest found).")
        return
    rows = [("source", "weight", "kind", "train", "val")] + [
        (r["name"], f"{r['weight']:g}", r["kind"], f"{r['train_sequences']:,}",
         f"{r['val_sequences']:,}") for r in m.sources]
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    click.echo(f"\nPretraining data mixture complete ({len(m.sources)} sources; train and "
               "val count clusters for a cluster source).\n")
    for r in rows:
        click.echo("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
    click.echo(f"\n  Manifest:  {folder / '.data_manifest'}\n"
               f"  Set pretrain.yaml's dataset_dir to {folder}.\n")


@data.command("write-manifest", hidden=True)
@click.option("--kind", required=True, type=click.Choice(["pretrain", "distillation"]))
@click.option("--output", "-o", required=True, type=click.Path(file_okay=False))
@click.option("--min-seq-len", required=True, type=int)
@click.option("--max-seq-len", required=True, type=int)
@click.option("--val-ratio", type=float, default=None,
              help="The val ratio to record; left out, the built val share.")
@click.option("--samples-per-shard", required=True, type=int)
@click.option("--register-tokens", type=int, default=0)
@click.option("--teacher-model", type=str, default=None)
@click.option("--cluster-index", is_flag=True, default=False,
              help="Describe a cluster-subsampled dataset (one member per cluster per pass).")
@click.help_option("--help", "-h")
def write_manifest_cmd(kind, output, min_seq_len, max_seq_len, val_ratio,
                       samples_per_shard, register_tokens, teacher_model, cluster_index):
    """Internal: write .data_manifest after shard generation."""
    output_dir = Path(output)
    if cluster_index:
        from nanoplm.data.preparation.cluster_pipeline import write_cluster_manifest
        if val_ratio is None:
            import json
            from nanoplm.data.datasets.cluster_sampled import INDEX_MANIFEST
            from nanoplm.data.preparation.cluster_config import CLUSTER_INDEX_DIR
            c = json.loads((output_dir / CLUSTER_INDEX_DIR / INDEX_MANIFEST).read_text())["counts"]
            val_ratio = round(c["val_clusters"] / max(1, c["clusters_in_universe"]), 6)
        write_cluster_manifest(output_dir, min_seq_len, max_seq_len, val_ratio,
                               samples_per_shard, register_tokens)
        click.echo(f"Wrote manifest: {output_dir / '.data_manifest'}")
        return
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    for d, name in [(train_dir, "train"), (val_dir, "val")]:
        if not d.is_dir():
            raise click.ClickException(
                f"write-manifest: expected shard directory not found: {d}. "
                f"Run the shard-writing stage before write-manifest."
            )
    if kind == "pretrain":
        train_count = _count_pretrain_shard_sequences(train_dir)
        val_count = _count_pretrain_shard_sequences(val_dir)
        train_tokens = _sum_pretrain_shard_tokens(train_dir)
    else:
        train_count = _count_distill_h5_sequences(train_dir)
        val_count = _count_distill_h5_sequences(val_dir)
    if val_ratio is None:
        val_ratio = round(val_count / max(1, train_count + val_count), 6)

    if kind == "pretrain":
        manifest = PretrainManifest(
            pipeline_mode="pretrain",
            seqs_num=train_count + val_count,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            val_ratio=val_ratio,
            train_dir="train",
            val_dir="val",
            train_sequences=train_count,
            val_sequences=val_count,
            num_register_tokens=register_tokens,
            sharded=True,
            samples_per_shard=samples_per_shard,
            total_train_tokens=train_tokens,
        )
    else:
        # Pre-computed distillation mode requires train_h5_prefix and val_h5_prefix.
        # Derive them from files present in the shard dirs.
        train_h5_files = list(train_dir.glob("*.h5"))
        val_h5_files = list(val_dir.glob("*.h5"))
        train_h5_prefix = _get_h5_prefix(train_h5_files) if train_h5_files else "train_kd_dataset.h5"
        val_h5_prefix = _get_h5_prefix(val_h5_files) if val_h5_files else "val_kd_dataset.h5"
        manifest = DistillationManifest(
            pipeline_mode="distillation",
            seqs_num=train_count + val_count,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            val_ratio=val_ratio,
            train_dir="train",
            val_dir="val",
            train_sequences=train_count,
            val_sequences=val_count,
            num_register_tokens=register_tokens,
            sharded=samples_per_shard > 0,
            samples_per_shard=samples_per_shard,
            teacher_model=teacher_model or "",
            train_h5_prefix=train_h5_prefix,
            val_h5_prefix=val_h5_prefix,
            on_the_fly=False,
        )
    write_manifest(output_dir, manifest)
    click.echo(f"Wrote manifest: {output_dir / '.data_manifest'}")


@data.command("write-mixture", hidden=True)
@click.option("--output", "-o", required=True, type=click.Path(file_okay=False))
@click.option("--source", "sources", multiple=True, required=True,
              type=(str, click.Path(file_okay=False), float, str),
              help="NAME FOLDER WEIGHT FASTA, once per source.")
@click.help_option("--help", "-h")
def write_mixture_cmd(output, sources):
    """Internal: write the mixture manifest of a data.yaml with several sources."""
    from nanoplm.data.infra.mixture import MixtureError, write_mixture
    entries = [(n, Path(d), w, f) for n, d, w, f in sources]
    try:
        path = write_mixture(Path(output), entries)
    except MixtureError as e:
        raise click.ClickException(f"write-mixture: {e}") from e
    click.echo(f"Wrote mixture manifest: {path}")


def _count_pretrain_shard_sequences(shard_dir: Path) -> int:
    """Sum sequence counts from each .idx.npy in the dir.

    Each shard's .idx.npy is an int32 array of per-sequence sizes (length N
    for N sequences), per the C writer (fast_fasta_shard_writer.c) and the
    ShardedDataset loader.
    """
    import numpy as np
    total = 0
    for idx_path in sorted(shard_dir.glob("shard_*.idx.npy")):
        arr = np.load(idx_path)
        total += len(arr)
    return total


def _sum_pretrain_shard_tokens(shard_dir: Path) -> int:
    """Sum all per-sequence encoded token lengths across a dir's .idx.npy shards.

    The .idx.npy arrays (already loaded to count sequences) are int32 arrays of
    per-sequence encoded lengths from the C writer's encoded_length() — i.e.
    used residues plus BOS/EOS/register tokens. Summing them is therefore the
    exact total encoded token count (the correct unit for a training token
    budget) — no extra I/O or extra pass over the data.
    """
    import numpy as np
    total = 0
    for idx_path in sorted(shard_dir.glob("shard_*.idx.npy")):
        total += int(np.load(idx_path).sum())
    return total


def _count_distill_h5_sequences(shard_dir: Path) -> int:
    """Sum group counts across all H5 files in the dir."""
    import h5py
    total = 0
    for h5_path in sorted(shard_dir.glob("*.h5")):
        with h5py.File(h5_path, "r") as f:
            total += sum(1 for k in f.keys() if k.isdigit())
    return total


def _resolve_path(path_value: str | Path, cwd: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (cwd / path).resolve()
    return path



def _get_h5_prefix(h5_files: list) -> str:
    """Get the base H5 prefix from a list of H5 files.

    For sharded files like "train_kd_dataset_shard_0000.h5", returns "train_kd_dataset.h5".
    For single files like "train_kd_dataset.h5", returns the filename as-is.
    """
    if not h5_files:
        return ""

    first_file = Path(h5_files[0]).name

    # Check if it's a sharded file
    if "_shard_" in first_file:
        # Extract prefix before "_shard_"
        prefix = first_file.split("_shard_")[0]
        return f"{prefix}.h5"
    else:
        return first_file


