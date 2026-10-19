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
    help="Local .fasta or .fasta.gz path (empty string treated as null).",
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
    """Extract compressed dataset files"""
    input_path = Path(input)
    output_path = Path(output)

    create_dirs(output_path)

    # Resolve output file path
    if output_path.is_dir():
        output_path = output_path / input_path.name.replace(".gz", "")

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
    # Check if the input file is a .fasta file
    if not input.endswith(".fasta"):
        raise click.ClickException("Input file must be a .fasta file")

    input_path = Path(input)
    output_path = Path(output)

    create_dirs(output_path)

    if output_path.is_dir():
        # replace .fasta with _shuffled.fasta
        output_path = output_path / input_path.name.replace(".fasta", "_shuffled.fasta")

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

    if output_path.is_dir():
        output_path = output_path / input_path.name.replace(".fasta", "_filtered.fasta")

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
@click.help_option("--help", "-h")
def split(
    input: str,
    output: str,
    val_ratio: float,
):
    """Split a filtered FASTA into train/val sets."""
    input_path = Path(input)
    output_path = Path(output)

    create_dirs(output_path)

    if not output_path.is_dir():
        raise click.ClickException(f"--output must be a directory, got: {output_path}")

    train_file_path = output_path / "train.fasta"
    val_file_path = output_path / "val.fasta"

    splitor = Splitor(
        input_file=input_path,
        train_file=train_file_path,
        val_file=val_file_path,
        val_ratio=val_ratio,
    )
    train_size, val_size = splitor.split()
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

    if output_path.is_dir():
        output_path = output_path / input_path.name.replace(".fasta", "_kd_dataset.h5")
    else:
        if not str(output_path).endswith(".h5"):
            raise click.ClickException("Output must end with .h5 or be a directory")

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
@click.option("--source", required=True, type=click.Choice(["uniref_xml", "mmseqs"]))
@click.option("--fasta", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", required=True, type=click.Path(dir_okay=False))
@click.option("--level", type=click.Choice(["uniref90", "uniref100"]), default=None)
@click.option("--xml", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--identity", type=float, default=0.5)
@click.option("--coverage", type=float, default=0.8)
@click.option("--tmp-dir", type=click.Path(file_okay=False), default=None)
@click.option("--threads", type=int, default=None)
def cluster_members(source, fasta, output, level, xml, identity, coverage, tmp_dir, threads):
    """Internal: write the canonical cluster table (cluster_id<TAB>sequence_id)."""
    from nanoplm.data.preparation.cluster_pipeline import build_members_table
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
@click.option("--val-ratio", required=True, type=float)
@click.option("--samples-per-shard", required=True, type=int)
@click.option("--seqs-num", type=int, default=-1)
@click.option("--register-tokens", type=int, default=0)
def cluster_index_cmd(members_tsv, fasta, pool, index_dir, val_dir, max_seq_len, val_ratio,
                      samples_per_shard, seqs_num, register_tokens):
    """Internal: build the cluster index and the validation gathers."""
    from nanoplm.data.preparation.cluster_pipeline import build_index_and_val
    man = build_index_and_val(Path(fasta), Path(pool), Path(members_tsv), Path(index_dir),
                              Path(val_dir), max_seq_len, val_ratio, samples_per_shard,
                              seqs_num, register_tokens)
    c = man["counts"]
    click.echo(f"Cluster index: {c['train_clusters']:,} train / {c['val_clusters']:,} val "
               f"clusters over {c['pool_rows']:,} pool sequences")


@data.command("get-yaml")
@click.argument("output", required=False,
                type=click.Path(file_okay=False, writable=True, resolve_path=True))
@click.option("--distillation", is_flag=True, default=False,
              help="Emit distillation_data.yaml instead of data.yaml")
@click.option("--force", is_flag=True, default=False,
              help="Overwrite existing files")
@click.help_option("--help", "-h")
def get_yaml(output, distillation, force):
    """Generate template data.yaml + dvc.yaml for the data pipeline."""
    from nanoplm.cli.templates import load_template
    from nanoplm.cli._dvc_yaml_gen import generate_dvc_yaml
    import yaml

    output_path = Path(output) if output else Path.cwd()
    create_dirs(output_path)

    if distillation:
        params_filename = "distillation_data.yaml"
        params_content = load_template("data_distill")
        kind = "distillation"
    else:
        params_filename = "data.yaml"
        params_content = load_template("data_pretrain")
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
    help="DVC stage to run (e.g., split). Defaults to full pipeline.",
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

    data_params = params.get("data_params", {})
    kind = _detect_kind(data_params, config_path.name)

    _reject_legacy_pipeline_mode(data_params)
    _validate_source_fields(params.get("data_dirs", {}))
    _validate_pipeline_fields(params.get("data_dirs", {}), data_params, kind, config_path)
    if kind == "pretrain":
        _validate_cluster_block(params, config_path)

    if fix:
        _flip_drop_invalid(config_path, True)
        click.echo(f"Set {config_path.name}: data_params.drop_invalid_sequences=true")
        # Re-read after flip so dvc.yaml is generated with updated value
        params = read_yaml(str(config_path))
        data_params = params.get("data_params", {})

    # Regenerate dvc.yaml BEFORE any DVC subprocess call
    dvc_yaml_path = cwd / "dvc.yaml"
    dvc_content = generate_dvc_yaml(
        config=params,
        kind=kind,
        params_filename=config_path.name,
        base_dir=cwd,
    )
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
            click.echo(f"Initialized DVC repository in: {cwd}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise click.ClickException(
                f"Failed to initialize DVC in {cwd}. Details: {e}"
            )

    # Build environment for repro
    max_workers = data_params.get("max_workers")
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

    # Build repro command
    if dry_run:
        cmd: list[str] = _DVC + ["repro", "--dry", "--allow-missing"]
    else:
        cmd = _DVC + ["repro"]
    if verbose:
        cmd.append("-v")
    if force_repro:
        cmd.append("-f")
    if target:
        cmd.append(target)

    repro_target_desc = target if target else "full pipeline"
    click.echo(f"Running dvc repro ({repro_target_desc}){'  [dry-run]' if dry_run else ''}...")
    try:
        result = subprocess.run(cmd, cwd=str(cwd), env=repro_env)
        if result.returncode != 0:
            raise click.ClickException(f"dvc repro exited with code {result.returncode}")
    except FileNotFoundError:
        raise click.ClickException(
            "'dvc' command not found. Please install DVC (pip install dvc)."
        )

    if not dry_run:
        _print_summary(cwd, params, kind, config_path)


# ---------------------------------------------------------------------------
# Private helpers for from-yaml
# ---------------------------------------------------------------------------

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
    - The chosen source must end with .fasta or .fasta.gz.
    - extracted_fasta is always required.
    - compressed_fasta is required only when the source is gzipped.
      In plain-.fasta mode it is optional; warn if it's set anyway.
    """
    from nanoplm.cli._dvc_yaml_gen import _source_is_compressed

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
    is_compressed = _source_is_compressed(source)
    # Extension whitelist: must end in .fasta or .fasta.gz (after stripping URL query/fragment).
    from urllib.parse import urlparse
    suffix_target = urlparse(source).path if "://" in source else source
    if not (suffix_target.endswith(".fasta") or suffix_target.endswith(".fasta.gz")):
        raise click.ClickException(
            f"url/local_fasta must end with .fasta or .fasta.gz, got: {source}"
        )

    extracted = data_dirs.get("extracted_fasta") or None
    if not extracted:
        raise click.ClickException("data_dirs.extracted_fasta is required.")

    compressed = data_dirs.get("compressed_fasta") or None
    if is_compressed and not compressed:
        raise click.ClickException(
            "data_dirs.compressed_fasta is required when source is a .gz file."
        )
    if (not is_compressed) and compressed:
        logger.warning(
            "data_dirs.compressed_fasta is unused when source is a plain .fasta file; "
            "it can be removed."
        )


# data_dirs path keys that generate_dvc_yaml (_dvc_yaml_gen.py) substitutes as
# ${data_dirs.X} into every pipeline stage, for both kinds. url/local_fasta,
# extracted_fasta and compressed_fasta are covered by _validate_source_fields.
_DATA_DIRS_REQUIRED = [
    "validation_dir", "shuffled_fasta", "filtered_fasta", "splitted_fasta_dir", "output_dir",
]


def _int_at_least(lo: int):
    return lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= lo


def _int_in_range(lo: int, hi: int):
    return lambda v: isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


def _shard_count(v) -> bool:
    """-1 (no sharding) or a positive shard size, matching kd-embed's own check."""
    return isinstance(v, int) and not isinstance(v, bool) and (v == -1 or v > 0)


def _unit_float(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < 1


def _optional_int(v) -> bool:
    return v is None or (isinstance(v, int) and not isinstance(v, bool))


def _is_bool(v) -> bool:
    return isinstance(v, bool)


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
    ("seqs_num", _BOTH, _int_at_least(-1)),
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
                f"{config_path.name}: data_params.{field} has an invalid value: {value!r}"
            )


def _validate_cluster_block(params: dict, config_path: Path) -> None:
    """Resolve cluster_subsampling now, so a bad block fails before DVC runs."""
    from nanoplm.data.preparation.cluster_config import ClusterConfigError, resolve_cluster_plan
    from nanoplm.data.preparation.cluster_table import MMSEQS_INSTALL_HINT, find_mmseqs
    try:
        plan = resolve_cluster_plan(params, config_path.parent)
    except ClusterConfigError as e:
        raise click.ClickException(f"{config_path.name}: {e}") from e
    if plan is None:
        return
    workers = (params.get("data_params") or {}).get("max_workers")
    if not (isinstance(workers, int) and not isinstance(workers, bool)
            and (workers == -1 or workers >= 1)):
        raise click.ClickException(
            f"{config_path.name}: data_params.max_workers must be -1 (all cores) or at "
            f"least 1 with cluster_subsampling, got {workers!r}")
    click.echo(f"Cluster source: {plan.source_line()}")
    for n in plan.notes:
        click.echo(n)
    for w in plan.warnings:
        logger.warning(w)
        click.echo(f"Warning: {w}", err=True)
    if plan.cluster_source == "mmseqs" and find_mmseqs() is None:
        raise click.ClickException(
            f"{config_path.name}: these members are clustered with MMseqs2, but `mmseqs` "
            f"was not found. {MMSEQS_INSTALL_HINT}")


def _detect_kind(data_params: dict, filename: str) -> str:
    """Detect pipeline kind from the config filename or data_params keys."""
    if "distillation" in filename:
        return "distillation"
    if "teacher_model" in data_params or "embed_calc_batch_size" in data_params:
        return "distillation"
    return "pretrain"


def _flip_drop_invalid(config_path: Path, value: bool) -> None:
    """Edit a single key in YAML preserving file structure as best as we can."""
    import re
    text = config_path.read_text()
    new_val = "true" if value else "false"
    pattern = re.compile(r"^(\s*drop_invalid_sequences\s*:\s*)(true|false)(\s*(?:#.*)?)$", re.M)
    new_text, n = pattern.subn(rf"\g<1>{new_val}\g<3>", text)
    if n == 0:
        raise click.ClickException(
            "Couldn't find drop_invalid_sequences in config to flip."
        )
    config_path.write_text(new_text)


def _print_summary(cwd: Path, params: dict, kind: str, config_path: Path) -> None:
    """Print a human-readable summary using .data_manifest if present."""
    import yaml as _yaml
    data_dirs = params.get("data_dirs") or {}
    output_dir_value = data_dirs.get("output_dir")
    if not output_dir_value:
        raise click.ClickException(
            f"{config_path.name}: data_dirs.output_dir is missing; cannot locate the pipeline output."
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
    src = data_dirs.get("url") or data_dirs.get("local_fasta")
    if m.get("cluster_index_dir"):
        from nanoplm.data.preparation.cluster_config import resolve_cluster_plan
        plan = resolve_cluster_plan(params, cwd)
        source = f"  Cluster source:  {plan.source_line()}\n" if plan else ""
        click.echo(
            f"\n{title} data pipeline complete (cluster subsampling).\n\n"
            f"  Members:         {plan.members if plan else src}\n"
            f"{source}"
            f"  Pool sequences:  {m.get('pool_sequences')}\n"
            f"  Clusters:        {m.get('seqs_num')} ({m.get('cluster_level')})\n"
            f"    Train:         {m.get('train_sequences')}\n"
            f"    Val:           {m.get('val_sequences')}\n"
            f"  Manifest:        {manifest_path}\n"
        )
        return
    click.echo(
        f"\n{title} data pipeline complete.\n\n"
        f"  Source:          {src}\n"
        f"  Total sequences: {m.get('seqs_num')}\n"
        f"    Train:         {m.get('train_sequences')}\n"
        f"    Val:           {m.get('val_sequences')}\n"
        f"  Manifest:        {manifest_path}\n"
    )


@data.command("write-manifest", hidden=True)
@click.option("--kind", required=True, type=click.Choice(["pretrain", "distillation"]))
@click.option("--output", "-o", required=True, type=click.Path(file_okay=False))
@click.option("--min-seq-len", required=True, type=int)
@click.option("--max-seq-len", required=True, type=int)
@click.option("--val-ratio", required=True, type=float)
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


