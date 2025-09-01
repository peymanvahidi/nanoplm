#!/usr/bin/env python3
"""
MypLM CLI - Data subcommands for MypLM package
"""

import click
from pathlib import Path

from myplm.data.downloader import Downloader
from myplm.data.extractor import Extractor
from myplm.data.shuffler import FastaShuffler
from myplm.data.filter_splitor import FilterSplitor
from myplm.data.dataset import ProtXDataProcessor, shard_h5_file
from myplm.models.teacher import ProtT5
from myplm.utils import logger, create_dirs
from myplm.config.datasets import DATASET_URLS


# Build dataset key list once for help text/metavar.
DATASET_KEYS = DATASET_URLS.keys()
DATASET_KEYS_TEXT = ", ".join(DATASET_KEYS)


@click.group(name="data")
@click.help_option('--help', '-h')
def data():
    """Data-related commands: download, extract, shuffle, filter-split, save-dataset, shard."""
    pass

@data.command(
    "download"
)
@click.help_option('--help', '-h')
@click.argument(
    'dataset',
    required=False,
    metavar=f"[DATASET_KEY: {DATASET_KEYS_TEXT}]",
    type=click.Choice(DATASET_KEYS, case_sensitive=False),
    default=None,
)
@click.option(
    '--url', 
    required=False,
    type=str,
    help='Explicit URL to download from. (can be used if dataset key is not provided)',
    default=None,
)
@click.option(
    '--output', 
    '-o',
    required=False,
    type=click.Path(exists=False),
    help='Directory path or file path where the dataset will be saved. If not provided the file name will be taken from the dataset url',
    default="downloaded_dataset",
)
@click.option(
    '--force',
    '-f',
    is_flag=True,
    help='Force download even if file already exists',
    default=False
)
def download(
    dataset: str | None,
    output: str | None,
    url: str | None,
    force: bool 
):
    """Download datasets using a dataset key (e.g., uniref50) or --url.

    Examples:
        protx data download uniref50 -o output/data/raw
        protx data download --url https://example.com/data.fasta.gz -o out.fasta.gz

    Supported keys: {keys}
    """.format(keys=", ".join(sorted(DATASET_URLS.keys())))

    output = Path(output)
    create_dirs(output)

    # Validate arguments
    if not dataset and not url:
        logger.error("Either a dataset key or --url must be provided")
        raise click.Abort()
    
    if dataset and url:
        logger.error("Both a dataset key and --url cannot be provided")
        raise click.Abort()
    
    # Get dataset URL
    if dataset:
        dataset_url = DATASET_URLS[dataset]
    else:
        dataset_url = url
    
    # Get output path
    if output.is_dir():
        output_path = output / Path(dataset_url).name
    else:
        output_path = output

    # Check if output path already exists
    if output_path.exists():
        if force:
            output_path.unlink()
            click.echo(f"Removed existing file: {output_path}")
        else:
            click.echo(f"File already exists: {output_path} \nUse --force to overwrite.")
            return

    # Download dataset
    try:
        downloader = Downloader(url=dataset_url, output_file=output_path)
        downloader.download()
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        click.echo(f"Download failed: {e}", err=True)
        raise click.Abort()
    

@data.command("extract")
@click.help_option('--help', '-h')
@click.option(
    '--input',
    '-i',
    required=True,
    type=click.Path(exists=True),
    help='Input file path where the dataset is saved'
)
@click.option(
    '--output',
    '-o',
    required=False,
    type=click.Path(exists=False),
    help='Output file path where the dataset will be saved. If not provided it will be saved in the same directory as the input file'
)
@click.option(
    '--force',
    '-f',
    is_flag=True,
    help='Force extraction even if output file already exists',
    default=False
)
def extract(input: str, output: str, force: bool):
    """Extract compressed dataset files"""

    # Resolve output path
    if not output:
        output = Path(input).parent / Path(input).name.replace(".gz", "")
    else:
        output = Path(output)

    # Create output directory
    create_dirs(output)
    
    # Check if output file already exists
    if output.exists():
        if force:
            output.unlink()
            click.echo(f"Removed existing file: {output}")
        else:
            click.echo(f"File already exists: {output} \nUse --force to overwrite")
            return

    # Extract dataset
    try:
        extractor = Extractor(
            input_file=input,
            output_file=output
        )
        extractor.extract()
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        click.echo(f"Extraction failed: {e}", err=True)
        raise click.Abort()


@data.command("shuffle")
@click.help_option('--help', '-h')
@click.option(
    '--input',
    '-i',
    required=True,
    type=click.Path(exists=True),
    help='Input FASTA file path'
)
@click.option(
    '--output',
    '-o',
    required=True,
    type=click.Path(exists=False),
    help='Output file path for the shuffled FASTA'
)
@click.option(
    '--seed',
    type=int,
    default=None,
    help='Random seed for shuffling (optional)'
)
def shuffle(input: str, output: str, seed: int):
    """Shuffle sequences in a FASTA file."""
    try:
        shuffler = FastaShuffler(
            input_file=input,
            output_file=output,
            seed=seed
        )
        shuffler.shuffle()
        click.echo(f"FASTA file shuffled successfully. Output: {output}")
    except Exception as e:
        logger.error(f"FASTA shuffling failed: {e}")
        click.echo(f"FASTA shuffling failed: {e}", err=True)
        raise click.Abort()


@data.command("filter-split")
@click.help_option('--help', '-h')
@click.option(
    '--input',
    '-i',
    required=True,
    type=click.Path(exists=True),
    help='Input file path where the dataset is saved'
)
@click.option(
    '--filtered-seqs',
    '-o',
    required=True,
    type=click.Path(exists=False),
    help='Output file path where the filtered dataset will be saved'
)
@click.option(
    '--train-file',
    required=True,
    type=click.Path(exists=False),
    help='Output file path where the splitted dataset will be saved'
)
@click.option(
    '--val-file',
    required=True,
    type=click.Path(exists=False),
    help='Output file path where the splitted dataset will be saved'
)
@click.option(
    '--min-seq-len',
    default=20,
    type=int,
    help='Minimum sequence length'
)
@click.option(
    '--max-seq-len',
    default=1024,
    type=int,
    help='Maximum sequence length'
)
@click.option(
    '--max-seqs-num',
    default=-1,
    type=int,
    help='Maximum number of sequences (-1 for all)'
)
@click.option(
    '--val-ratio',
    default=0.1,
    type=float,
    help='Validation ratio'
)
@click.option(
    '--info-file',
    required=True,
    type=click.Path(exists=False),
    help='Info file path where the dataset will be saved'
)
@click.option(
    '--skip-n',
    default=0,
    type=int,
    help='Number of sequences to skip from the beginning of the input FASTA file'
)
def filter_split(
    input: str,
    filtered_seqs: str,
    train_file: str,
    val_file: str,
    min_seq_len: int,
    max_seq_len: int,
    max_seqs_num: int,
    val_ratio: float,
    info_file: str,
    skip_n: int
):
    """Filter sequences by length, optionally skip N sequences, and split into train/validation sets"""
    try:
        filter_splitor = FilterSplitor(
            input_file=input,
            output_file=filtered_seqs,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            max_seqs_num=max_seqs_num,
            val_ratio=val_ratio,
            info_file=info_file,
            skip_n=skip_n
        )
        filter_splitor.filter()
        filter_splitor.split(
            train_file=train_file,
            val_file=val_file
        )
        click.echo("Data filtering and splitting completed successfully")
        
    except Exception as e:
        logger.error(f"Filter and split failed: {e}")
        click.echo(f"Filter and split failed: {e}", err=True)
        raise click.Abort()

@data.command("save-dataset")
@click.help_option('--help', '-h')
@click.option(
    '--input-file',
    '-i',
    required=True,
    type=click.Path(exists=True),
    help='Input file path where the dataset is saved'
)
@click.option(
    '--teacher-model',
    required=False,
    default='prott5',
    type=click.Choice(['prott5']),
    help='Teacher model to use'
)
@click.option(
    '--output-file',
    '-o',
    required=True,
    type=click.Path(exists=False),
    help='Output HDF5 file path where the processed dataset will be saved.'
)
@click.option(
    '--max-seq-len',
    default=1024,
    type=int,
    help='Maximum sequence length'
)
@click.option(
    '--batch-size',
    default=64,
    type=int,
    help='Batch size for embedding calculation'
)
@click.option(
    '--device',
    default='auto',
    type=click.Choice(['auto', 'cuda', 'mps', 'cpu']),
    help='Device to use'
)
@click.option(
    '--skip-n',
    default=0,
    type=int,
    help='Number of sequences to skip from the beginning of the input FASTA file before processing'
)
@click.option(
    '--n-files',
    default=1,
    type=int,
    help='Number of shard files to create (1 for single file, >1 for sharded files). Compatible with --sharded flag in training.'
)
def save_dataset(
    input_file: str,
    teacher_model: str,
    output_file: str,
    max_seq_len: int, 
    batch_size: int, 
    device: str,
    skip_n: int,
    n_files: int
):
    """Generate ProtX datasets with teacher embeddings"""
    if teacher_model == "prott5":
        selected_teacher_model = ProtT5()
    else:
        logger.error(f"Teacher model {teacher_model} not supported")
        raise click.Abort()
    
    protx_data = ProtXDataProcessor(
        data_path=input_file,
        teacher_model=selected_teacher_model,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        device=device,
        skip_n=skip_n,
        n_files=n_files
    )
    result = protx_data.process_dataset(
        save_path=Path(output_file)
    )
    
    if n_files > 1:
        click.echo(f"ProtX dataset generation completed successfully. Created {len(result)} shard files:")
        for path in result:
            click.echo(f"  {path}")
    else:
        click.echo(f"ProtX dataset generation completed successfully. Output: {result}")

@data.command("shard")
@click.help_option('--help', '-h')
@click.option(
    '--input-file',
    '-i',
    required=True,
    type=click.Path(exists=True),
    help='Input H5 file to shard'
)
@click.option(
    '--n-shards',
    '-n',
    required=True,
    type=int,
    help='Number of shard files to create'
)
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(exists=False),
    default=None,
    help='Output directory for shard files (defaults to same directory as input file)'
)
@click.option(
    '--total-sequences',
    '-t',
    type=int,
    default=None,
    help='Total number of sequences (if known, skips counting for faster start)'
)
def shard(
    input_file: str,
    n_shards: int,
    output_dir: str,
    total_sequences: int
):
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
        total_sequences=total_sequences
    )
    
    click.echo(f"Successfully created {len(shard_paths)} shard files:")
    for path in shard_paths:
        click.echo(f"  {path}")

