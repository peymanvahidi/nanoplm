#!/usr/bin/env python3
"""
ProtX CLI - Command Line Interface for ProtX package
"""

import click
from pathlib import Path

from .data.downloader import Downloader
from .data.extractor import Extractor
from .data.filter_splitor import FilterSplitor
from .data.dataset import ProtXDataProcessor
from .data.pipeline import DataPipeline
from .models.teacher import ProtT5
from .config import DataConfig
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
    '--shuffle',
    default=True,
    help='Shuffle the dataset'
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
    shuffle: bool
):
    """Filter sequences by length and split into train/validation sets"""
    try:
        filter_splitor = FilterSplitor(
            input_file=input,
            output_file=filtered_seqs,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            max_seqs_num=max_seqs_num,
            val_ratio=val_ratio,
            info_file=info_file,
            shuffle=shuffle
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
    help='Output file prefix where the dataset will be saved, since several files will be created based on the number of sequences in each file.'
)
@click.option(
    '--max-seq-len',
    default=1024,
    type=int,
    help='Maximum sequence length'
)
@click.option(
    '--seqs-num-per-file',
    default=10_000,
    type=int,
    help='Number of sequences per file'
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
def save_protx_dataset(
    input_file: str,
    teacher_model: str,
    output_file: str,
    max_seq_len: int, 
    seqs_num_per_file: int, 
    batch_size: int, 
    device: str
):
    """Generate ProtX datasets with teacher embeddings"""
    if teacher_model == "prott5":
        teacher_model = ProtT5()
    else:
        raise ValueError(f"Teacher model {teacher_model} not supported")
    
    protx_data = ProtXDataProcessor(
        data_path=input_file,
        teacher_model=teacher_model,
        max_seq_len=max_seq_len,
        seqs_num_per_file=seqs_num_per_file,
        batch_size=batch_size,
        device=device
    )
    protx_data.process_dataset(
        save_path=Path(output_file)
    )
    click.echo(f"ProtX dataset generation completed successfully")


@cli.command("run-pipeline")
@click.help_option('--help', '-h')
@click.option(
    '--config-file',
    type=click.Path(exists=True),
    help='Path to custom config file (defaults to params.yaml)'
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
        if config_file:
            config = DataConfig(config_file)
        else:
            config = DataConfig()
        
        pipeline = DataPipeline(config)
        pipeline.run_all(save_protx_dataset=save_protx_dataset)
        
        click.echo("Complete pipeline execution finished successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        click.echo(f"Pipeline failed: {e}", err=True)
        raise click.Abort()

if __name__ == '__main__':
    cli()
