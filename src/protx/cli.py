#!/usr/bin/env python3
"""
ProtX CLI - Command Line Interface for ProtX package
"""

import click
from pathlib import Path

from .data.downloader import Downloader
from .data.extractor import Extractor
from .data.shuffler import FastaShuffler
from .data.filter_splitor import FilterSplitor
from .data.dataset import ProtXDataProcessor
from .data.pipeline import DataPipeline
from .distillation.pipeline import DistillationPipeline
from .models.teacher import ProtT5
from .config import DataConfig, DistillConfig
from .utils import logger

@click.group()
@click.version_option()
@click.help_option(
    '--help',
    '-h',
)
def cli():
    """ProtX - Knowledge distillation of ProtT5"""
    pass


@cli.command("download-data")
@click.help_option('--help', '-h')
@click.option(
    '--url', 
    required=True,
    type=str,
    help='URL to download the dataset from'
)
@click.option(
    '--output', 
    '-o',
    required=True,
    type=click.Path(exists=False),
    help='Output file path where the dataset will be saved'
)
@click.option(
    '--force',
    '-f',
    is_flag=True,
    help='Force download even if file already exists'
)
def download_data(url: str, output: str, force: bool):
    """Download datasets using the Downloader class"""
    try:
        output_path = Path(output)
        
        if force and output_path.exists():
            output_path.unlink()
            logger.info(f"Removed existing file: {output_path}")
        
        downloader = Downloader(url=url, output_file=str(output_path))
        downloader.download()
        
        click.echo(f"Successfully downloaded to: {output_path}")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        click.echo(f"Download failed: {e}", err=True)
        raise click.Abort()


@cli.command("extract-data")
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
    required=True,
    type=click.Path(exists=False),
    help='Output file path where the dataset will be saved'
)
def extract_data(input: str, output: str):
    """Extract compressed dataset files"""
    try:
        extractor = Extractor(
            input_file=input,
            output_file=output
        )
        extractor.extract()
        click.echo("Data extraction completed successfully")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        click.echo(f"Extraction failed: {e}", err=True)
        raise click.Abort()


@cli.command("shuffle-fasta")
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
def shuffle_fasta_cli(input: str, output: str, seed: int):
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


@cli.command("filter-split-data")
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
def filter_split_data(
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


@cli.command("save-protx-dataset")
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
def save_protx_dataset(
    input_file: str,
    teacher_model: str,
    output_file: str,
    max_seq_len: int, 
    batch_size: int, 
    device: str,
    skip_n: int
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
        skip_n=skip_n
    )
    protx_data.process_dataset(
        save_path=Path(output_file)
    )
    click.echo(f"ProtX dataset generation completed successfully. Output: {output_file}")


@cli.command("run-pipeline")
@click.help_option('--help', '-h')
@click.option(
    '--config-file',
    type=click.Path(exists=True),
    default=None,
    help='Path to custom YAML config file (defaults to loading params.yaml if present)'
)
@click.option(
    '--save-protx-dataset',
    is_flag=True,
    help='Also generate ProtX datasets with teacher embeddings'
)
def run_pipeline(
    config_file: str,
    save_protx_dataset: bool
):
    """Run the complete data pipeline from download to dataset creation using config file"""
    try:
        config = DataConfig(config_file if config_file else "params.yaml")
        
        pipeline = DataPipeline(config)
        pipeline.run_all(save_protx_dataset=save_protx_dataset)
        
        click.echo("Complete pipeline execution finished successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        click.echo(f"Pipeline failed: {e}", err=True)
        raise click.Abort()

@cli.command("train-student")
@click.help_option('--help', '-h')
def train_student(
    data_config: str = DataConfig(),
    distill_config: str = DistillConfig()
):
    """Train the student model"""
    pipeline = DistillationPipeline(
        data_config=data_config,
        distill_config=distill_config
    )
    pipeline.train()

if __name__ == '__main__':
    cli()
