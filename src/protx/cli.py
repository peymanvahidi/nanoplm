#!/usr/bin/env python3
"""
ProtX CLI - Command Line Interface for ProtX package
"""

import click
from pathlib import Path
import json

from .data.downloader import Downloader
from .data.extractor import Extractor
from .data.shuffler import FastaShuffler
from .data.filter_splitor import FilterSplitor
from .data.dataset import ProtXDataProcessor
from .distillation.pipeline_builder import DistillationPipelineBuilder
from .models.teacher import ProtT5
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

@cli.command("train-student")
@click.help_option('--help', '-h')
@click.option(
    '--train-file',
    type=str,
    required=True,
    help='Path to the training file'
)
@click.option(
    '--val-file',
    type=str,
    required=True,
    help='Path to the validation file'
)
@click.option(
    '--protx-train-prefix',
    type=str,
    required=True,
    help='Prefix of the training ProtX dataset'
)
@click.option(
    '--protx-val-prefix',
    type=str,
    required=True,
    help='Prefix of the validation ProtX dataset'
)
@click.option(
    '--student-embed-dim',
    type=int,
    default=512,
    help='Embedding dimension of the student model'
)
@click.option(
    '--student-num-layers',
    type=int,
    default=6,
    help='Number of layers of the student model'
)
@click.option(
    '--student-num-heads',
    type=int,
    default=8,
    help='Number of heads of the student model'
)
@click.option(
    '--on-the-fly',
    is_flag=True,
    help='Whether to use on-the-fly teacher embeddings'
)
@click.option(
    '--multi-gpu',
    is_flag=True,
    help='Whether to use multiple GPUs for training'
)
@click.option(
    '--num-epochs',
    type=int,
    default=10,
    help='Number of epochs to train the student model'
)
@click.option(
    '--batch-size',
    type=int,
    default=64,
    help='Batch size for training'
)
@click.option(
    '--max-lr',
    type=float,
    default=1e-3,
    help='Maximum learning rate'
)
@click.option(
    '--max-grad-norm',
    type=float,
    default=1.0,
    help='Maximum gradient norm for gradient clipping'
)
@click.option(
    '--max-seqs-num',
    type=int,
    default=1000000,
    help='Maximum number of sequences to use for training'
)
@click.option(
    '--max-seq-len',
    type=int,
    default=1024,
    help='Maximum sequence length'
)
@click.option(
    '--val-ratio',
    type=float,
    default=0.1,
    help='Ratio of validation set'
)
@click.option(
    '--num-workers',
    type=int,
    default=4,
    help='Number of workers to use for training'
)
@click.option(
    '--project-name',
    type=str,
    default="protx_distillation",
    help='Name of the project'
)
# @click.option(
#     '--checkpoint-dir',
#     type=str,
#     default=None,
#     help='Path to the checkpoint directory to resume training from'
# )
@click.option(
    '--wandb-dir',
    type=str,
    default=None,
    help='Path to the directory to save the wandb logs'
)
@click.option(
    '--device',
    default='cuda',
    type=click.Choice(['cuda', 'mps', 'cpu']),
    help='Device to use'
)
@click.option(
    '--lr-scheduler',
    type=click.Choice(['cosine', 'linear', 'polynomial', 'constant']),
    default='cosine',
    help='Learning rate scheduler type'
)
@click.option(
    '--lr-scheduler-kwargs',
    type=str,
    default=None,
    help='JSON string of additional kwargs for the learning rate scheduler (optional). ' +
         'Example: \'{"num_cycles": 1.0, "power": 1.0}\' for cosine/polynomial schedulers'
)
def train_student(
    train_file: str,
    val_file: str,
    protx_train_prefix: str,
    protx_val_prefix: str,
    student_embed_dim: int,
    student_num_layers: int,
    student_num_heads: int,
    on_the_fly: bool,
    multi_gpu: bool,
    num_epochs: int,
    batch_size: int,
    max_lr: float,
    max_grad_norm: float,
    max_seqs_num: int,
    max_seq_len: int,
    val_ratio: float,
    num_workers: int,
    project_name: str,
    # checkpoint_dir: str,
    wandb_dir: str,
    device: str,
    lr_scheduler: str,
    lr_scheduler_kwargs: str,
):
    """Train the student model"""
    # Parse lr_scheduler_kwargs if provided
    parsed_lr_kwargs = {}
    if lr_scheduler_kwargs:
        try:
            parsed_lr_kwargs = json.loads(lr_scheduler_kwargs)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON for lr-scheduler-kwargs: {e}")
            click.echo(f"Invalid JSON for lr-scheduler-kwargs: {e}", err=True)
            raise click.Abort()
    
    builder = DistillationPipelineBuilder()

    pipeline = (
        builder
            .with_training_files(
                train_file=train_file,
                val_file=val_file,
                protx_train_prefix=protx_train_prefix,
                protx_val_prefix=protx_val_prefix
            )
            .with_model_config(
                student_embed_dim=student_embed_dim,
                student_num_layers=student_num_layers,
                student_num_heads=student_num_heads,
            )
            .with_training_config(
                num_epochs=num_epochs,
                batch_size=batch_size,
                max_lr=max_lr,
                max_seqs_num=max_seqs_num,
                max_seq_len=max_seq_len,
                val_ratio=val_ratio,
                num_workers=num_workers,
                lr_scheduler=lr_scheduler,
                lr_scheduler_kwargs=parsed_lr_kwargs,
                max_grad_norm=max_grad_norm,
            )
            .with_experiment_config(
                project_name=project_name,
                wandb_dir=wandb_dir,
                device=device,
                on_the_fly=on_the_fly,
                multi_gpu=multi_gpu,
            )
        .build()
    )

    pipeline.train()

@cli.command("resume-training")
@click.help_option('--help', '-h')
@click.option(
    '--checkpoint-dir',
    type=str,
    required=True,
    help='Path to the checkpoint directory to resume training from'
)
@click.option(
    '--num-epochs',
    type=int,
    required=True,
    help='Number of epochs to train the student model'
)
@click.option(
    '--lr',
    type=float,
    default=None,
    help='Override learning rate for resumed training (optional)'
)
@click.option(
    '--lr-scheduler',
    type=click.Choice(['cosine', 'linear', 'polynomial', 'constant']),
    default=None,
    help='Learning rate scheduler type (optional, defaults to cosine)'
)
@click.option(
    '--lr-scheduler-kwargs',
    type=str,
    default=None,
    help='JSON string of additional kwargs for the learning rate scheduler (optional). ' +
         'Example: \'{"num_cycles": 1.0, "power": 1.0}\' for cosine/polynomial schedulers'
)
@click.option(
    '--max-grad-norm',
    type=float,
    default=None,
    help='Override gradient clipping norm for resumed training (optional)'
)
def resume_training(
    checkpoint_dir: str,
    num_epochs: int,
    lr: float,
    lr_scheduler: str,
    lr_scheduler_kwargs: str,
    max_grad_norm: float
):
    """Resume training from a checkpoint with optional learning rate and scheduler overrides.
    
    Examples:
        # Resume with new learning rate:
        protx resume-training --checkpoint-dir ./run-123/checkpoint-1500 --num-epochs 20 --lr 5e-4
        
        # Resume with linear scheduler:
        protx resume-training --checkpoint-dir ./run-123/checkpoint-1500 --num-epochs 20 --lr-scheduler linear
        
        # Resume with constant learning rate:
        protx resume-training --checkpoint-dir ./run-123/checkpoint-1500 --num-epochs 20 --lr-scheduler constant
    """
    # Parse lr_scheduler_kwargs if provided
    parsed_lr_kwargs = {}
    if lr_scheduler_kwargs:
        try:
            parsed_lr_kwargs = json.loads(lr_scheduler_kwargs)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON for lr-scheduler-kwargs: {e}")
            click.echo(f"Invalid JSON for lr-scheduler-kwargs: {e}", err=True)
            raise click.Abort()
    
    builder = DistillationPipelineBuilder()
    
    # Build overrides dict
    overrides = {"num_epochs": num_epochs}
    if lr is not None:
        overrides["max_lr"] = lr
    if max_grad_norm is not None:
        overrides["max_grad_norm"] = max_grad_norm
    if lr_scheduler is not None:
        overrides["lr_scheduler"] = lr_scheduler
    if parsed_lr_kwargs:
        overrides["lr_scheduler_kwargs"] = parsed_lr_kwargs
    
    pipeline = builder.resume_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        **overrides
    )
    pipeline.train()

if __name__ == '__main__':
    cli()
