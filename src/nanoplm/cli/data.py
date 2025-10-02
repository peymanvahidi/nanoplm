#!/usr/bin/env python3
"""
nanoPLM CLI - Data subcommands for nanoPLM package
"""

import click
from pathlib import Path
import shutil
import subprocess

from nanoplm.config.datasets import DATASET_URLS

from nanoplm.data.downloader import Downloader, DownloadError
from nanoplm.data.extractor import Extractor, ExtractionError
from nanoplm.data.shuffler import FastaShuffler, ShufflingError
from nanoplm.data.filterer import Filterer, FilterError
from nanoplm.data.splitor import Splitor, SplitError
from nanoplm.data.dataset import SaveKDDataset, shard_h5_file
from nanoplm.models.teacher import ProtT5

from nanoplm.utils import create_dirs
from nanoplm.utils.common import inside_git_repo, is_git_subdir


@click.group(name="data")
@click.help_option("--help", "-h")
def data():
    """Group of commands for managing datasets."""
    pass


@data.command("download")
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
    help="Explicit URL to download from. (can be used if dataset key is not provided).",
    default=None,
)
@click.option(
    "--output",
    "-o",
    required=False,
    type=click.Path(exists=False),
    help="Directory path or file path where the dataset will be saved. If not provided the file name will be taken from the dataset url.",
    default="downloaded_dataset",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Flag to force download even if file already exists.",
    default=False,
)
@click.help_option("--help", "-h")
def download(dataset: str | None, output: str | None, url: str | None, force: bool):
    """Download datasets using a dataset key (e.g., uniref50) or --url.

    Examples:

    \b
      nanoplm data download uniref50 -o data_directory
      nanoplm data download --url https://example.com/data.fasta.gz -o data_directory/mydata.fasta.gz
    """

    output_path = Path(output)

    # Validate arguments
    if not dataset and not url:
        raise click.UsageError("Either a dataset key or --url must be provided.")

    if dataset and url:
        raise click.UsageError("Both a dataset key and --url cannot be provided.")

    # Get dataset URL
    if dataset:
        dataset_url = DATASET_URLS[dataset]
    else:
        dataset_url = url

    create_dirs(output_path)

    # Get output  path
    if output_path.is_dir():
        output_path = output_path / Path(dataset_url).name

    # Check if output path already exists
    if output_path.exists():
        if force:
            output_path.unlink()
            click.echo(f"Removed existing file: {output_path}")
        else:
            click.echo(
                f"File already exists: {output_path} \nUse --force to overwrite."
            )
            return

    # Download dataset
    try:
        downloader = Downloader(url=dataset_url, output_path=output_path)
        downloader.download()
        click.echo(f"Dataset downloaded to: {output_path}")

    except DownloadError as e:
        raise click.ClickException(f"Download failed: {e}")
    except Exception as e:
        raise click.ClickException(f"An unexpected error occurred during download: {e}")


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
    try:
        extractor = Extractor(input_path=input_path, output_path=output_path)
        extractor.extract()
        click.echo(f"Extraction completed successfully. Output: {output_path}")

    except (ExtractionError, EOFError, OSError) as e:
        if isinstance(e, EOFError):
            raise click.ClickException(
                "Extraction failed: Corrupted or incomplete gzip file"
            )
        elif isinstance(e, OSError):
            raise click.ClickException(f"Extraction failed: {e}")
        else:
            raise click.ClickException(f"Extraction failed: {e}")
    except Exception as e:
        raise click.ClickException(
            f"An unexpected error occurred during extraction: {e}"
        )


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
    type=click.Choice(["seqkit", "biopython"]),
    default="biopython",
    help="Backend to use for shuffling",
)
@click.option(
    "--seed", type=int, default=None, help="Random seed for shuffling (optional)"
)
@click.help_option("--help", "-h")
def shuffle(input: str, output: str, backend: str, seed: int):
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

    try:
        shuffler = FastaShuffler(
            input_path=input_path,
            output_path=output_path,
            backend=backend,
            seed=seed,
        )
        shuffler.shuffle()
    except ShufflingError as e:
        raise click.ClickException(f"FASTA shuffling failed: {e}")
    except Exception as e:
        raise click.ClickException(
            f"An unexpected error occurred during shuffling: {e}"
        )


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
@click.help_option("--help", "-h")
def filter(
    input: str,
    output: str,
    min_seq_len: int,
    max_seq_len: int,
    seqs_num: int,
    skip_n: int,
):
    """Filter sequences by length and optional max count."""
    input_path = Path(input)
    output_path = Path(output)

    create_dirs(output_path)

    if output_path.is_dir():
        output_path = output_path / input_path.name.replace(".fasta", "_filtered.fasta")

    try:
        filterer = Filterer(
            input_path=input_path,
            output_path=output_path,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            seqs_num=seqs_num,
            skip_n=skip_n,
        )
        filterer.filter()
        click.echo("Filtering completed successfully")
    except FilterError as e:
        raise click.ClickException(f"Filtering failed: {e}")
    except Exception as e:
        raise click.ClickException(
            f"An unexpected error occurred during filtering: {e}"
        )


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
    help="Output file path / directory for the train split",
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
        click.echo("Output must be a directory")
    else:
        train_file_path = output_path / "train.fasta"
        val_file_path = output_path / "val.fasta"

    try:
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

    except SplitError as e:
        raise click.ClickException(f"Splitting failed: {e}")
    except Exception as e:
        raise click.ClickException(
            f"An unexpected error occurred during splitting: {e}"
        )


@data.command("save-kd-dataset")
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
    "--n-files",
    default=1,
    type=int,
    help="Number of shard files to create (1 for single file, >1 for sharded files).",
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
    default="auto",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    help="Device to use",
)
@click.option(
    "--skip-n",
    default=0,
    type=int,
    help="Number of sequences to skip from the beginning of the input FASTA file before processing",
)
@click.help_option("--help", "-h")
def save_kd_dataset(
    input: str,
    output: str,
    teacher_model: str,
    max_seq_len: int,
    n_files: int,
    batch_size: int,
    force: bool,
    device: str,
    skip_n: int,
):

    if n_files < 1:
        raise click.ClickException("n-files must be at least 1")
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

    """Generate knowledge distillation datasets with teacher embeddings"""
    if teacher_model == "prott5":
        selected_teacher_model = ProtT5()
    else:
        click.ClickException(f"Teacher model {teacher_model} not supported")

    kd_data = SaveKDDataset(
        input_fasta=input_path,
        output_path=output_path,
        teacher=selected_teacher_model,
        mode="get_embeddings",
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        device=device,
        skip_n=skip_n,
        n_files=n_files,
        force=force,
    )
    result = kd_data.process_dataset()

    if n_files > 1:
        click.echo(
            f"Knowledge distillation dataset generation completed successfully. Created {len(result)} shard files:"
        )
        for path in result:
            click.echo(f"  {path}")
    else:
        click.echo(
            f"Knowledge distillation dataset generation completed successfully. Output: {result}"
        )


@data.command("shard")
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
def shard(input_file: str, n_shards: int, output_dir: str, total_sequences: int):
    """Shard a large H5 file into smaller files for better performance.

    Examples:
        # Basic sharding (will count sequences first)
        protx data shard --input-file train.h5 --n-shards 33

        # Fast sharding (skips counting if you know the sequence count)
        protx data shard --input-file train.h5 --n-shards 33 --total-sequences 12500000

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


@data.command("get-yaml")
@click.help_option("--help", "-h")
@click.argument(
    "output",
    required=False,
    type=click.Path(dir_okay=True, writable=True, resolve_path=True),
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite if file exists",
)
def get_yaml(output: str | None, force: bool):
    """Generate a DVC params.yaml and dvc.yaml template for data pipeline.

    If OUTPUT is omitted, writes ./params.yaml and ./dvc.yaml in the current directory.
    If OUTPUT is a directory, writes params.yaml and dvc.yaml inside it.
    """

    if output is None:
        output_path = Path.cwd()
    else:
        output_path = Path(output)

    create_dirs(output_path)

    if not output_path.is_dir():
        raise click.ClickException(
            f"OUTPUT must be a directory, not a file: {output_path}"
        )

    params_path = output_path / "params.yaml"
    dvc_path = output_path / "dvc.yaml"

    if not force:
        existing = []
        if params_path.exists():
            existing.append(str(params_path))
        if dvc_path.exists():
            existing.append(str(dvc_path))
        if existing:
            raise click.ClickException(
                "File(s) already exist: " + ", ".join(existing) + ". Use --force to overwrite."
            )

    template = (
        "data_params:\n"
        "  seqs_num: 20000\n"
        "  min_seq_len: 20\n"
        "  max_seq_len: 512\n"
        "  val_ratio: 0.1\n"
        "\n"
        "  teacher_model: \"prott5\"\n"
        "  embed_calc_batch_size: 4\n"
        "\n"
        "  device: \"auto\"\n"
        "  \n"
        "  shuffle: true\n"
        "  shuffle_seed: 24\n"
        "  # If you want to skip some sequences\n"
        "  filter_skip_n: 0\n"
        "  # For sharding the train and val datasets for KNOWLEDGE DISTILLATION\n"
        "  train_shards: 5\n"
        "  val_shards: 2\n"
        "\n"
        "# Data directories\n"
        "data_dirs:\n"
        "  url: \"https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz\"\n"
        "  # swissprot: \"https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz\"\n"
        "  # trembl: \"https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_trembl.fasta.gz\"\n"
        "  # uniref50: \"https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz\"\n"
        "  # uniref90: \"https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz\"\n"
        "  # uniref100: \"https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz\"\n"
        "  compressed_fasta: \"output/data/raw/uniref50.fasta.gz\"\n"
        "  extracted_fasta: \"output/data/raw/uniref50.fasta\"\n"
        "\n"
        "  shuffled_fasta: \"output/data/raw/uniref50_shuffled.fasta\"\n"
        "\n"
        "  filtered_fasta: \"output/data/filter/uniref50_filtered.fasta\"\n"
        "  splitted_fasta_dir: \"output/data/split\"\n"
        "\n"
        "  kd_train_dir: \"output/data/kd_dataset/train\"\n"
        "  kd_val_dir: \"output/data/kd_dataset/val\"\n"
    )

    if force and params_path.exists():
        params_path.unlink()

    params_path.write_text(template, encoding="utf-8")

    dvc_yaml_template = (
        "# This is a template for a DVC pipeline for the data pipeline.\n"
        "# DO NOT EDIT THIS FILE, IT IS AUTO-GENERATED.\n"
        "stages:\n"
        "\n"
        "  download:\n"
        "    cmd: nanoplm data download --url ${data_dirs.url} -o ${data_dirs.compressed_fasta}\n"
        "    params:\n"
        "      - data_dirs.url\n"
        "      - data_dirs.compressed_fasta\n"
        "    outs:\n"
        "      - ${data_dirs.compressed_fasta}\n"
        "\n"
        "  extract:\n"
        "    cmd: nanoplm data extract -i ${data_dirs.compressed_fasta} -o ${data_dirs.extracted_fasta}\n"
        "    deps:\n"
        "      - ${data_dirs.compressed_fasta}\n"
        "    params:\n"
        "      - data_dirs.compressed_fasta\n"
        "      - data_dirs.extracted_fasta\n"
        "    outs:\n"
        "      - ${data_dirs.extracted_fasta}\n"
        "\n"
        "  shuffle:\n"
        "    cmd: nanoplm data shuffle -i ${data_dirs.extracted_fasta} -o ${data_dirs.shuffled_fasta} --seed ${data_params.shuffle_seed}\n"
        "    deps:\n"
        "      - ${data_dirs.extracted_fasta}\n"
        "    params:\n"
        "      - data_dirs.shuffled_fasta\n"
        "      - data_params.shuffle_seed\n"
        "    outs:\n"
        "      - ${data_dirs.shuffled_fasta}\n"
        "\n"
        "  filter:\n"
        "    cmd: nanoplm data filter -i ${data_dirs.shuffled_fasta} -o ${data_dirs.filtered_fasta} --min-seq-len ${data_params.min_seq_len} --max-seq-len ${data_params.max_seq_len} --seqs-num ${data_params.seqs_num} --skip-n ${data_params.filter_skip_n}\n"
        "    deps:\n"
        "      - ${data_dirs.shuffled_fasta}\n"
        "    params:\n"
        "      - data_dirs.shuffled_fasta\n"
        "      - data_params.min_seq_len\n"
        "      - data_params.max_seq_len\n"
        "      - data_params.seqs_num\n"
        "      - data_params.filter_skip_n\n"
        "    outs:\n"
        "      - ${data_dirs.filtered_fasta}\n"
        "\n"
        "  split:\n"
        "    cmd: nanoplm data split -i ${data_dirs.filtered_fasta} -o ${data_dirs.splitted_fasta_dir} --val-ratio ${data_params.val_ratio}\n"
        "    deps:\n"
        "      - ${data_dirs.filtered_fasta}\n"
        "    params:\n"
        "      - data_dirs.filtered_fasta\n"
        "      - data_dirs.splitted_fasta_dir\n"
        "      - data_params.val_ratio\n"
        "    outs:\n"
        "      - ${data_dirs.splitted_fasta_dir}\n"
        "\n"
        "  save_kd_train_dataset:\n"
        "    cmd: nanoplm data save-kd-dataset -i ${data_dirs.splitted_fasta_dir}/train.fasta -o ${data_dirs.kd_train_dir} --teacher-model ${data_params.teacher_model} --max-seq-len ${data_params.max_seq_len} --n-files ${data_params.train_shards} --batch-size ${data_params.embed_calc_batch_size} --device ${data_params.device} --skip-n ${data_params.filter_skip_n}\n"
        "    deps:\n"
        "      - ${data_dirs.splitted_fasta_dir}/train.fasta\n"
        "    params:\n"
        "      - data_dirs.splitted_fasta_dir\n"
        "      - data_dirs.kd_train_dir\n"
        "      - data_params.max_seq_len\n"
        "      - data_params.train_shards\n"
        "      - data_params.embed_calc_batch_size\n"
        "      - data_params.filter_skip_n\n"
        "    outs:\n"
        "      - ${data_dirs.kd_train_dir}\n"
        "\n"
        "  save_kd_val_dataset:\n"
        "    cmd: nanoplm data save-kd-dataset -i ${data_dirs.splitted_fasta_dir}/val.fasta -o ${data_dirs.kd_val_dir} --teacher-model ${data_params.teacher_model} --max-seq-len ${data_params.max_seq_len} --n-files ${data_params.val_shards} --batch-size ${data_params.embed_calc_batch_size} --device ${data_params.device} --skip-n ${data_params.filter_skip_n}\n"
        "    deps:\n"
        "      - ${data_dirs.splitted_fasta_dir}/val.fasta\n"
        "    params:\n"
        "      - data_dirs.splitted_fasta_dir\n"
        "      - data_dirs.kd_val_dir\n"
        "      - data_params.max_seq_len\n"
        "      - data_params.val_shards\n"
        "      - data_params.embed_calc_batch_size\n"
        "      - data_params.filter_skip_n\n"
        "    outs:\n"
        "      - ${data_dirs.kd_val_dir}\n"
    )

    if force and dvc_path.exists():
        dvc_path.unlink()

    dvc_path.write_text(dvc_yaml_template, encoding="utf-8")

    click.echo(f"Both files are written to: {output_path} directory.\nEdit the params.yaml file and use `nanoplm data repro` to run the pipeline.")


@data.command("repro")
@click.help_option("--help", "-h")
@click.option(
    "--params",
    "params_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
    required=False,
    help="Path to a params.yaml to copy into the working directory before running.",
)
@click.option(
    "--distillation",
    is_flag=True,
    default=False,
    help="Run full pipeline including knowledge distillation dataset preparation. By default, only runs up to split stage.",
)
@click.option(
    "--target",
    type=str,
    required=False,
    help="DVC stage to reproduce (e.g., split). Overrides --distillation flag if specified.",
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
    "verbose",
    is_flag=True,
    default=False,
    help="Run 'dvc repro' with -v for verbose output.",
)
def repro(
    params_path: str | None,
    distillation: bool,
    target: str | None,
    no_auto_init: bool,
    force_repro: bool,
    verbose: bool,
):
    """Run the DVC data pipeline, optionally with a specific params file.

    By default, runs pipeline up to the 'split' stage, which is sufficient for pretraining.
    Use --distillation to include knowledge distillation dataset preparation stages.

    This will:
    - Optionally initialize DVC in the current directory.
    - If --params is given, copy it to params.yaml in the working directory.
    - Run `dvc repro` for the specified stages.
    """

    cwd = Path.cwd()

    work_dir = Path(params_path).parent if params_path else cwd

    dvc_dir = work_dir / ".dvc"
    if not dvc_dir.exists():
        if not no_auto_init:
            try:
                init_cmd = ["dvc", "init", "-q"]
                if not inside_git_repo(work_dir):
                    init_cmd.append("--no-scm")
                elif is_git_subdir(work_dir):
                    init_cmd.append("--subdir")
                subprocess.run(init_cmd, cwd=str(work_dir), check=True)
                click.echo(f"Initialized DVC repository in: {work_dir}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise click.ClickException(
                    f"Failed to initialize DVC in {work_dir}. Ensure DVC is installed and on PATH. Details: {e}"
                )
        else:
            raise click.ClickException(
                f"DVC is not initialized in {work_dir}. Run 'dvc init' first."
            )

    dvc_yaml = work_dir / "dvc.yaml"
    if not dvc_yaml.exists():
        raise click.ClickException(
            f"dvc.yaml not found in {work_dir}. Create a pipeline first or copy the sample from the nanoPLM repo."
        )

    # Determine the target stage
    if target is None and not distillation:
        target = "split"
        click.echo("Running pipeline up to 'split' stage (use --distillation to include KD dataset preparation)")

    # Build repro command
    cmd: list[str] = ["dvc", "repro"]
    if verbose:
        cmd.append("-v")
    if force_repro:
        cmd.append("-f")
    if target:
        cmd.append(target)

    params_dst = work_dir / "params.yaml"

    if params_path is not None:
        src = Path(params_path)
        try:
            if src.resolve() != params_dst.resolve():
                shutil.copy2(src, params_dst)
        except FileNotFoundError:
            raise click.ClickException(f"Params file not found: {src}")
    
    try:
        subprocess.run(cmd, cwd=str(work_dir), check=True)
    except FileNotFoundError:
        raise click.ClickException("'dvc' command not found. Please install DVC (pip install dvc).")
    except subprocess.CalledProcessError as e:
        raise click.ClickException(f"dvc repro failed with exit code {e.returncode}")
