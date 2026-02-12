#!/usr/bin/env python3
"""
nanoPLM CLI - Data subcommands for nanoPLM package
"""

import click
from pathlib import Path
import subprocess

from nanoplm.config.datasets import DATASET_URLS

from nanoplm.data.downloader import Downloader, DownloadError
from nanoplm.data.extractor import Extractor, ExtractionError
from nanoplm.data.shuffler import FastaShuffler, ShufflingError
from nanoplm.data.filterer import Filterer, FilterError
from nanoplm.data.splitor import Splitor, SplitError
from nanoplm.data.dataset import SaveKDDataset, shard_h5_file
from nanoplm.models.teacher import ProtT5
from nanoplm.pretraining.dataset import ShardWriter
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer

from nanoplm.utils import create_dirs
from nanoplm.utils.common import inside_git_repo, is_git_subdir, read_yaml, is_flash_attention_available


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


@data.command("save-pretrain-dataset")
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
    default=2,
    type=int,
    help="Number of CPUs. Use --max-workers -1 to use all available CPUs",
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
@click.help_option("--help", "-h")
def save_pretrain_dataset(
    input: str,
    output: str,
    max_seq_len: int,
    max_workers: int,
    samples_per_shard: int,
    force: bool,

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

    tokenizer = ProtModernBertTokenizer()

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
                "File(s) already exist: "
                + ", ".join(existing)
                + ". Use --force to overwrite."
            )

    template = (
        "data_params:\n"
        "  # Pipeline mode: 'pretrain', 'distillation', or 'none'\n"
        "  # - 'pretrain': Generate HDF5 shards for MLM pretraining\n"
        "  # - 'distillation': Generate teacher embeddings for knowledge distillation\n"
        "  # - 'none': Only run data preparation (download, filter, split)\n"
        '  pipeline_mode: "pretrain"\n'
        "\n"
        "  seqs_num: 20000\n"
        "  min_seq_len: 20\n"
        "  max_seq_len: 512\n"
        "  val_ratio: 0.1\n"
        '  device: "auto"\n'
        "\n"
        '  shuffle_backend: "biopython"  # or "seqkit" (faster, requires installation)\n'
        "  shuffle: true\n"
        "  shuffle_seed: 24\n"
        "  filter_skip_n: 0\n"
        "\n"
        "# Pretrain config (used when pipeline_mode: 'pretrain')\n"
        "# A .data_manifest file will be created in output_dir for use by pretrain pipeline\n"
        "pretrain_config:\n"
        '  output_dir: "output/data/pretrain_data"  # Will contain train/ and val/ subdirs\n'
        "  samples_per_shard: 2000\n"
        "  max_workers: 2  # -1 to use all available CPUs\n"
        "  force: false\n"
        "\n"
        "# Distillation config (used when pipeline_mode: 'distillation')\n"
        "# A .data_manifest file will be created in output_dir for use by distill pipeline\n"
        "distillation_config:\n"
        '  output_dir: "output/data/distillation_data"  # Will contain train/ and val/ subdirs\n'
        "  on_the_fly: false  # If true, skip embedding generation (embeddings computed during training)\n"
        "  samples_per_shard: 2000  # -1 for single file (no sharding)\n"
        '  teacher_model: "prott5"\n'
        "  embed_calc_batch_size: 4\n"
        "\n"
        "# Data directories\n"
        "data_dirs:\n"
        '  url: "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz"\n'
        '  # swissprot: "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz"\n'
        '  # trembl: "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_trembl.fasta.gz"\n'
        '  # uniref50: "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"\n'
        '  # uniref90: "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz"\n'
        '  # uniref100: "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz"\n'
        '  compressed_fasta: "output/data/raw/uniref50.fasta.gz"\n'
        '  extracted_fasta: "output/data/raw/uniref50.fasta"\n'
        '  shuffled_fasta: "output/data/raw/uniref50_shuffled.fasta"\n'
        '  filtered_fasta: "output/data/filter/uniref50_filtered.fasta"\n'
        '  splitted_fasta_dir: "output/data/split"\n'
    )

    if force and params_path.exists():
        params_path.unlink()

    params_path.write_text(template, encoding="utf-8")

    dvc_yaml_template = (
        "# This is a template for a DVC pipeline for the data pipeline.\n"
        "# DO NOT EDIT THIS FILE, IT IS AUTO-GENERATED.\n"
        "# The KD stages are only run when pipeline_mode is 'distillation'.\n"
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
        "    cmd: nanoplm data shuffle -i ${data_dirs.extracted_fasta} -o ${data_dirs.shuffled_fasta} --backend ${data_params.shuffle_backend} --seed ${data_params.shuffle_seed}\n"
        "    deps:\n"
        "      - ${data_dirs.extracted_fasta}\n"
        "    params:\n"
        "      - data_dirs.shuffled_fasta\n"
        "      - data_params.shuffle_backend\n"
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
        "    cmd: nanoplm data save-kd-dataset -i ${data_dirs.splitted_fasta_dir}/train.fasta -o ${distillation_config.output_dir}/train --teacher-model ${distillation_config.teacher_model} --max-seq-len ${data_params.max_seq_len} --samples-per-shard ${distillation_config.samples_per_shard} --batch-size ${distillation_config.embed_calc_batch_size} --device ${data_params.device} --skip-n ${data_params.filter_skip_n}\n"
        "    deps:\n"
        "      - ${data_dirs.splitted_fasta_dir}/train.fasta\n"
        "    params:\n"
        "      - data_dirs.splitted_fasta_dir\n"
        "      - distillation_config.output_dir\n"
        "      - distillation_config.teacher_model\n"
        "      - distillation_config.samples_per_shard\n"
        "      - distillation_config.embed_calc_batch_size\n"
        "      - data_params.max_seq_len\n"
        "      - data_params.filter_skip_n\n"
        "    outs:\n"
        "      - ${distillation_config.output_dir}/train\n"
        "\n"
        "  save_kd_val_dataset:\n"
        "    cmd: nanoplm data save-kd-dataset -i ${data_dirs.splitted_fasta_dir}/val.fasta -o ${distillation_config.output_dir}/val --teacher-model ${distillation_config.teacher_model} --max-seq-len ${data_params.max_seq_len} --samples-per-shard ${distillation_config.samples_per_shard} --batch-size ${distillation_config.embed_calc_batch_size} --device ${data_params.device} --skip-n ${data_params.filter_skip_n}\n"
        "    deps:\n"
        "      - ${data_dirs.splitted_fasta_dir}/val.fasta\n"
        "    params:\n"
        "      - data_dirs.splitted_fasta_dir\n"
        "      - distillation_config.output_dir\n"
        "      - distillation_config.teacher_model\n"
        "      - distillation_config.samples_per_shard\n"
        "      - distillation_config.embed_calc_batch_size\n"
        "      - data_params.max_seq_len\n"
        "      - data_params.filter_skip_n\n"
        "    outs:\n"
        "      - ${distillation_config.output_dir}/val\n"
    )

    if force and dvc_path.exists():
        dvc_path.unlink()

    dvc_path.write_text(dvc_yaml_template, encoding="utf-8")

    click.echo(
        f"Both files are written to: {output_path} directory.\nEdit the params.yaml file and use `nanoplm data repro` to run the pipeline."
    )


@data.command("from-yaml")
@click.help_option("--help", "-h")
@click.argument(
    "config",
    default="params.yaml",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--target",
    type=str,
    required=False,
    help="DVC stage to reproduce (e.g., split). Overrides pipeline_mode if specified.",
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
def from_yaml(
    config: str,
    target: str | None,
    no_auto_init: bool,
    force_repro: bool,
    verbose: bool,
):
    """Run the DVC data pipeline from a YAML file.

    The pipeline_mode in params.yaml controls what dataset is generated:
    - 'none': Only run data preparation (download, filter, split)
    - 'pretrain': Generate HDF5 shards for MLM pretraining
    - 'distillation': Generate teacher embeddings for knowledge distillation

    This will:
    - Optionally initialize DVC in the current directory.
    - Run `dvc repro` for the appropriate stages based on pipeline_mode.
    """

    config = Path(config)

    if config.is_absolute():
        cwd = config.parent
        params_yaml = config
    else:
        params_yaml = Path.cwd() / config
        cwd = params_yaml.parent

    dvc_yaml = cwd / "dvc.yaml"
    dvc_dir = cwd / ".dvc"

    if not dvc_yaml.exists():
        raise click.ClickException(
            f"dvc.yaml not found in {cwd}."
            "\nUse `nanoplm data get-yaml` to generate a dvc.yaml file."
        )

    if not params_yaml.exists():
        raise click.ClickException(
            f"params.yaml not found in {cwd}."
            "\nUse `nanoplm data get-yaml` to generate a params.yaml file."
        )

    # Read params early to get pipeline_mode
    try:
        params = read_yaml(str(params_yaml))
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    except Exception as e:
        raise click.ClickException(f"Failed to read params.yaml: {e}") from e

    data_params = params.get("data_params", {})
    pipeline_mode = data_params.get("pipeline_mode", "pretrain")

    # Validate pipeline_mode
    valid_modes = {"none", "pretrain", "distillation"}
    if pipeline_mode not in valid_modes:
        raise click.ClickException(
            f"Invalid pipeline_mode: '{pipeline_mode}'. Must be one of: {', '.join(sorted(valid_modes))}"
        )

    if not dvc_dir.exists():
        if not no_auto_init:
            try:
                init_cmd = ["dvc", "init", "-q"]
                if not inside_git_repo(cwd):
                    init_cmd.append("--no-scm")
                elif is_git_subdir(cwd):
                    init_cmd.append("--subdir")
                subprocess.run(init_cmd, cwd=str(cwd), check=True)
                click.echo(f"Initialized DVC repository in: {cwd}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise click.ClickException(
                    f"Failed to initialize DVC in {cwd}. Ensure DVC is installed and on PATH. Details: {e}"
                )
        else:
            raise click.ClickException(
                f"DVC is not initialized in {cwd}. Run 'dvc init' first."
            )

    # Check if on_the_fly mode is enabled for distillation
    distillation_config = params.get("distillation_config", {})
    on_the_fly = distillation_config.get("on_the_fly", False)

    # Determine the target stage based on pipeline_mode
    if target is None:
        if pipeline_mode == "distillation" and not on_the_fly:
            # Run full DVC pipeline including KD stages
            target = None  # No target means run all stages
            click.echo("Running full pipeline for distillation (pipeline_mode: 'distillation', on_the_fly: false)")
        else:
            # For 'none', 'pretrain', and 'distillation' with on_the_fly, only run up to split stage
            target = "split"
            if pipeline_mode == "distillation" and on_the_fly:
                click.echo(f"Running pipeline up to 'split' stage (pipeline_mode: 'distillation', on_the_fly: true)")
            else:
                click.echo(f"Running pipeline up to 'split' stage (pipeline_mode: '{pipeline_mode}')")

    # Build repro command
    cmd: list[str] = ["dvc", "repro"]
    if verbose:
        cmd.append("-v")
    if force_repro:
        cmd.append("-f")
    if target:
        cmd.append(target)

    try:
        subprocess.run(cmd, cwd=str(cwd), check=True)
    except FileNotFoundError:
        raise click.ClickException(
            "'dvc' command not found. Please install DVC (pip install dvc)."
        )
    except subprocess.CalledProcessError as e:
        raise click.ClickException(f"dvc repro failed with exit code {e.returncode}")

    # Generate pretraining shards if pipeline_mode is 'pretrain'
    if pipeline_mode == "pretrain":
        data_dirs = params.get("data_dirs", {})
        pretrain_config = params.get("pretrain_config", {})

        splitted_dir = data_dirs.get("splitted_fasta_dir")
        if not splitted_dir:
            raise click.ClickException("data_dirs.splitted_fasta_dir is required for pretrain mode")

        train_fasta = _resolve_path(Path(splitted_dir) / "train.fasta", cwd)
        val_fasta = _resolve_path(Path(splitted_dir) / "val.fasta", cwd)

        # Use output_dir with train/ and val/ subdirectories
        output_dir = _resolve_path(pretrain_config.get("output_dir", "output/data/pretrain_shards"), cwd)
        train_hdf5_dir = output_dir / "train"
        val_hdf5_dir = output_dir / "val"
        create_dirs(train_hdf5_dir)
        create_dirs(val_hdf5_dir)

        max_seq_len = data_params.get("max_seq_len")
        samples_per_shard = pretrain_config.get("samples_per_shard")
        max_workers = pretrain_config.get("max_workers")
        force = pretrain_config.get("force")

        tokenizer = ProtModernBertTokenizer()

        click.echo("Creating binary shards for training dataset...")
        train_saver = ShardWriter(
            fasta_path=str(train_fasta),
            tokenizer=tokenizer,
            max_length=max_seq_len,
            output_dir=str(train_data_dir),
            samples_per_shard=samples_per_shard,
            max_workers=max_workers,
            force=force,
        )
        train_shards = train_saver.create_shards()
        train_sequences = len(train_saver._keys)

        click.echo("Creating binary shards for validation dataset...")
        val_saver = ShardWriter(
            fasta_path=str(val_fasta),
            tokenizer=tokenizer,
            max_length=max_seq_len,
            output_dir=str(val_data_dir),
            samples_per_shard=samples_per_shard,
            max_workers=max_workers,
            force=force,
        )
        val_shards = val_saver.create_shards()
        val_sequences = len(val_saver._keys)

        # Write manifest
        manifest = PretrainManifest(
            pipeline_mode="pretrain",
            seqs_num=data_params.get("seqs_num"),
            min_seq_len=data_params.get("min_seq_len"),
            max_seq_len=max_seq_len,
            val_ratio=data_params.get("val_ratio"),
            train_dir="train",
            val_dir="val",
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            sharded=True,
            samples_per_shard=samples_per_shard,
        )
        manifest_path = write_manifest(output_dir, manifest)

        click.echo(
            "Pretraining shard generation complete:\n"
            f"  Train shards: {len(train_shards)} ({train_sequences} sequences) -> {train_hdf5_dir}\n"
            f"  Val shards:   {len(val_shards)} ({val_sequences} sequences) -> {val_hdf5_dir}\n"
            f"  Manifest: {manifest_path}"
        )
    elif pipeline_mode == "distillation":
        # Write manifest for distillation
        distillation_config = params.get("distillation_config", {})
        output_dir = _resolve_path(distillation_config.get("output_dir", "output/data/kd_dataset"), cwd)
        on_the_fly = distillation_config.get("on_the_fly", False)

        if on_the_fly:
            # On-the-fly mode: Create manifest with FASTA paths (no embedding generation)
            data_dirs = params.get("data_dirs", {})
            splitted_dir = data_dirs.get("splitted_fasta_dir")
            if not splitted_dir:
                raise click.ClickException("data_dirs.splitted_fasta_dir is required for distillation mode")

            train_fasta = _resolve_path(Path(splitted_dir) / "train.fasta", cwd)
            val_fasta = _resolve_path(Path(splitted_dir) / "val.fasta", cwd)

            # Count sequences from FASTA files
            train_sequences = _count_sequences_in_fasta(train_fasta)
            val_sequences = _count_sequences_in_fasta(val_fasta)

            manifest = DistillationManifest(
                pipeline_mode="distillation",
                seqs_num=data_params.get("seqs_num"),
                min_seq_len=data_params.get("min_seq_len"),
                max_seq_len=data_params.get("max_seq_len"),
                val_ratio=data_params.get("val_ratio"),
                train_dir="train",
                val_dir="val",
                train_sequences=train_sequences,
                val_sequences=val_sequences,
                teacher_model=distillation_config.get("teacher_model"),
                on_the_fly=True,
                train_fasta=str(train_fasta),
                val_fasta=str(val_fasta),
            )
            manifest_path = write_manifest(output_dir, manifest)

            click.echo(
                f"Distillation manifest created (on-the-fly mode):\n"
                f"  Train FASTA: {train_fasta} ({train_sequences} sequences)\n"
                f"  Val FASTA: {val_fasta} ({val_sequences} sequences)\n"
                f"  Manifest: {manifest_path}\n"
                f"  Note: Teacher embeddings will be generated during training"
            )
        else:
            # Pre-computed mode: Create manifest with H5 paths (after DVC pipeline completes)
            # Count sequences from generated files
            train_sequences = _count_sequences_in_dir(output_dir / "train")
            val_sequences = _count_sequences_in_dir(output_dir / "val")

            # Determine H5 prefix names
            train_h5_files = list((output_dir / "train").glob("*.h5"))
            val_h5_files = list((output_dir / "val").glob("*.h5"))

            # Get the base prefix (e.g., "train_kd_dataset.h5" from "train_kd_dataset_shard_0000.h5")
            train_h5_prefix = _get_h5_prefix(train_h5_files) if train_h5_files else "train_kd_dataset.h5"
            val_h5_prefix = _get_h5_prefix(val_h5_files) if val_h5_files else "val_kd_dataset.h5"

            samples_per_shard = distillation_config.get("samples_per_shard")
            sharded = samples_per_shard > 0

            manifest = DistillationManifest(
                pipeline_mode="distillation",
                seqs_num=data_params.get("seqs_num"),
                min_seq_len=data_params.get("min_seq_len"),
                max_seq_len=data_params.get("max_seq_len"),
                val_ratio=data_params.get("val_ratio"),
                train_dir="train",
                val_dir="val",
                train_sequences=train_sequences,
                val_sequences=val_sequences,
                sharded=sharded,
                samples_per_shard=samples_per_shard,
                teacher_model=distillation_config.get("teacher_model"),
                train_h5_prefix=train_h5_prefix,
                val_h5_prefix=val_h5_prefix,
                on_the_fly=False,
            )
            manifest_path = write_manifest(output_dir, manifest)

            click.echo(
                f"Distillation dataset generation complete:\n"
                f"  Train: {train_sequences} sequences -> {output_dir / 'train'}\n"
                f"  Val: {val_sequences} sequences -> {output_dir / 'val'}\n"
                f"  Manifest: {manifest_path}"
            )
    else:
        click.echo("Data preparation complete (pipeline_mode: 'none')")


def _resolve_path(path_value: str | Path, cwd: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (cwd / path).resolve()
    return path


def _count_sequences_in_dir(directory: Path) -> int:
    """Count total sequences in all H5 files in a directory.
    This is to handle these two scenarios (otherwise we could have calcualted these using max_seq_num and val_ratio)
    1. If you request 20,000 sequences but the file ends after finding only 15,000 valid ones, it stops there. In this case, using the calculation would be wrong (you'd mistakenly think you have 20,000).
    2. If you use seqs_num: -1 (the default for "use everything"), you don't know the final count until the process finishes.

    Supports two formats:
    1. KD dataset format: each sequence is a numbered group (e.g., "0", "1", "2")
    2. Array format: sequences stored as arrays in datasets like "input_ids", "embeddings", etc.
    """
    import h5py

    total = 0
    h5_files = list(directory.glob("*.h5"))

    for h5_file in h5_files:
        try:
            with h5py.File(h5_file, "r") as f:
                # Try to count groups first (KD dataset format)
                # Groups are named with string integers like "0", "1", "2"
                group_count = 0
                for key in f.keys():
                    if key.isdigit():
                        group_count += 1

                if group_count > 0:
                    total += group_count
                else:
                    # Fall back to common dataset names (array format)
                    if "input_ids" in f:
                        total += len(f["input_ids"])
                    elif "sequences" in f:
                        total += len(f["sequences"])
                    elif "embeddings" in f:
                        total += len(f["embeddings"])
                    elif "teacher_embeddings" in f:
                        total += len(f["teacher_embeddings"])
        except Exception:
            pass

    return total


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


def _count_sequences_in_fasta(fasta_path: Path) -> int:
    """Count sequences in a FASTA file."""
    from Bio import SeqIO

    count = 0
    try:
        with open(fasta_path, "r") as f:
            for _ in SeqIO.parse(f, "fasta"):
                count += 1
    except Exception as e:
        click.echo(f"Warning: Could not count sequences in {fasta_path}: {e}", err=True)
        return 0

    return count
