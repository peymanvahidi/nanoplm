"""
Pre-flight validation for dataset manifests and shard files.

Provides comprehensive validation checks before training to catch data issues
early with clear, actionable error messages.
"""

from pathlib import Path
from typing import Union, List, Optional, Dict
import yaml
import h5py

from nanoplm.utils import logger


class ValidationError(Exception):
    """
    Custom exception for dataset validation failures.

    Provides clear, actionable error messages that guide users to fix
    data preparation issues before training.
    """
    pass


def validate_dataset_manifest(
    manifest_path: Union[str, Path],
    expected_mode: Optional[str] = None
) -> dict:
    """
    Validate dataset manifest file and return parsed contents.

    Checks that manifest file exists, is properly formatted, and contains
    all required fields based on the pipeline mode (pretrain or distillation).

    Args:
        manifest_path: Path to .data_manifest file
        expected_mode: Expected pipeline_mode ("pretrain" or "distillation").
                      If provided, validates that manifest matches this mode.

    Returns:
        dict: Parsed and validated manifest

    Raises:
        FileNotFoundError: If manifest file doesn't exist
        ValidationError: If manifest is malformed or missing required fields

    Example:
        >>> manifest = validate_dataset_manifest(
        ...     "output/data/pretrain_shards/.data_manifest",
        ...     expected_mode="pretrain"
        ... )
        >>> print(f"Max seq len: {manifest['max_seq_len']}")
    """
    manifest_path = Path(manifest_path)

    # Check manifest exists
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {manifest_path}\n\n"
            f"The manifest file is missing. This file is created by 'nanoplm data from-yaml'.\n"
            f"To create your dataset, run:\n"
            f"  nanoplm data from-yaml\n\n"
            f"Make sure you have a params.yaml file configured with your dataset settings."
        )

    # Parse YAML
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValidationError(
            f"Failed to parse manifest file {manifest_path}:\n{e}\n\n"
            f"The manifest file appears to be corrupted.\n"
            f"Try regenerating it by running: nanoplm data from-yaml"
        )
    except Exception as e:
        raise ValidationError(f"Error reading manifest file {manifest_path}: {e}")

    if manifest is None:
        raise ValidationError(
            f"Manifest file is empty: {manifest_path}\n\n"
            f"The manifest file exists but contains no data.\n"
            f"Regenerate it by running: nanoplm data from-yaml"
        )

    # Validate common required fields
    common_required = ['pipeline_mode', 'train_dir', 'val_dir', 'max_seq_len']
    missing_common = [f for f in common_required if f not in manifest]
    if missing_common:
        raise ValidationError(
            f"Manifest missing required fields: {missing_common}\n"
            f"File: {manifest_path}\n\n"
            f"The manifest may be corrupted or created by an older version.\n"
            f"Regenerate it by running: nanoplm data from-yaml"
        )

    pipeline_mode = manifest['pipeline_mode']

    # Validate pipeline_mode value
    if pipeline_mode not in ['pretrain', 'distillation']:
        raise ValidationError(
            f"Invalid pipeline_mode '{pipeline_mode}' in manifest.\n"
            f"Expected 'pretrain' or 'distillation'.\n"
            f"File: {manifest_path}"
        )

    # Check expected mode if provided
    if expected_mode is not None and pipeline_mode != expected_mode:
        raise ValidationError(
            f"Dataset was prepared for '{pipeline_mode}' pipeline, "
            f"but '{expected_mode}' was expected.\n\n"
            f"Please use a dataset prepared with pipeline_mode: '{expected_mode}' in params.yaml.\n"
            f"To create the correct dataset, update params.yaml and run:\n"
            f"  nanoplm data from-yaml"
        )

    # Validate mode-specific fields
    if pipeline_mode == 'pretrain':
        if 'samples_per_shard' not in manifest or manifest['samples_per_shard'] <= 0:
            raise ValidationError(
                f"Pretraining manifest must have samples_per_shard > 0.\n"
                f"Found: {manifest.get('samples_per_shard')}\n"
                f"File: {manifest_path}"
            )

    elif pipeline_mode == 'distillation':
        # Check for teacher_model
        if 'teacher_model' not in manifest or not manifest['teacher_model']:
            raise ValidationError(
                f"Distillation manifest must specify teacher_model.\n"
                f"File: {manifest_path}\n\n"
                f"Update params.yaml with distillation_config.teacher_model and regenerate."
            )

        # Validate based on on_the_fly mode
        on_the_fly = manifest.get('on_the_fly', False)

        if on_the_fly:
            # On-the-fly mode: must have FASTA paths
            required_fasta = ['train_fasta', 'val_fasta']
            missing_fasta = [f for f in required_fasta if not manifest.get(f)]
            if missing_fasta:
                raise ValidationError(
                    f"On-the-fly distillation mode requires {missing_fasta}.\n"
                    f"File: {manifest_path}\n\n"
                    f"The manifest indicates on_the_fly mode but is missing FASTA paths."
                )
        else:
            # Pre-computed mode: must have shard info
            if not manifest.get('sharded'):
                raise ValidationError(
                    f"Pre-computed distillation mode requires sharded=True.\n"
                    f"File: {manifest_path}"
                )
            if not manifest.get('samples_per_shard') or manifest['samples_per_shard'] <= 0:
                raise ValidationError(
                    f"Pre-computed distillation mode requires samples_per_shard > 0.\n"
                    f"Found: {manifest.get('samples_per_shard')}\n"
                    f"File: {manifest_path}"
                )
            required_h5 = ['train_h5_prefix', 'val_h5_prefix']
            missing_h5 = [f for f in required_h5 if not manifest.get(f)]
            if missing_h5:
                raise ValidationError(
                    f"Pre-computed distillation mode requires {missing_h5}.\n"
                    f"File: {manifest_path}"
                )

    logger.info(f"✓ Manifest validation passed: {manifest_path.name}")
    logger.info(f"  Pipeline mode: {pipeline_mode}")
    logger.info(f"  Max seq len: {manifest['max_seq_len']}")

    return manifest


def validate_shard_files(
    shard_dir: Union[str, Path],
    expected_count: Optional[int] = None,
    check_contents: bool = True
) -> List[Path]:
    """
    Validate HDF5 shard files in a directory.

    Checks that shard files exist, are readable, and contain expected datasets.
    Optionally validates that the expected number of shards are present.

    Args:
        shard_dir: Directory containing *.h5 shard files
        expected_count: Expected number of shard files (optional)
        check_contents: Whether to open files and validate contents

    Returns:
        List[Path]: List of validated shard file paths

    Raises:
        ValidationError: If shards are missing, corrupt, or empty

    Example:
        >>> shard_paths = validate_shard_files(
        ...     "output/data/pretrain_shards/train",
        ...     expected_count=10,
        ...     check_contents=True
        ... )
        >>> print(f"Validated {len(shard_paths)} shards")
    """
    shard_dir = Path(shard_dir)

    # Check directory exists
    if not shard_dir.exists():
        raise ValidationError(
            f"Shard directory does not exist: {shard_dir}\n\n"
            f"The data preparation may have failed or the path is incorrect.\n"
            f"Run: nanoplm data from-yaml"
        )

    if not shard_dir.is_dir():
        raise ValidationError(
            f"Path is not a directory: {shard_dir}\n\n"
            f"Expected a directory containing HDF5 shard files."
        )

    # Find all .h5 files
    shard_files = sorted(shard_dir.glob("*.h5"))

    if not shard_files:
        raise ValidationError(
            f"No HDF5 shard files (*.h5) found in: {shard_dir}\n\n"
            f"The directory exists but contains no shard files.\n"
            f"Data preparation may have failed. Check logs and re-run:\n"
            f"  nanoplm data from-yaml"
        )

    # Check expected count
    if expected_count is not None and len(shard_files) != expected_count:
        raise ValidationError(
            f"Expected {expected_count} shard files but found {len(shard_files)} in: {shard_dir}\n\n"
            f"Found files:\n" + "\n".join(f"  - {f.name}" for f in shard_files[:5]) +
            (f"\n  ... and {len(shard_files) - 5} more" if len(shard_files) > 5 else "") + "\n\n"
            f"Some shard files may be missing. Re-run data preparation:\n"
            f"  nanoplm data from-yaml"
        )

    # Validate contents if requested
    if check_contents:
        logger.info(f"Validating {len(shard_files)} shard files in {shard_dir.name}...")

        for shard_path in shard_files:
            # Check file is readable
            if not shard_path.exists():
                raise ValidationError(
                    f"Shard file missing: {shard_path}\n"
                    f"File was listed but no longer exists. Check filesystem."
                )

            # Try to open and validate contents
            try:
                with h5py.File(shard_path, 'r') as f:
                    # Check for required dataset
                    if 'input_ids' not in f:
                        raise ValidationError(
                            f"Shard missing 'input_ids' dataset: {shard_path}\n\n"
                            f"This shard file is incomplete or corrupted.\n"
                            f"Delete the shard and regenerate:\n"
                            f"  rm {shard_path}\n"
                            f"  nanoplm data from-yaml"
                        )

                    # Check dataset is not empty
                    # For HDF5 files with group structure (distillation), check key count
                    # For direct dataset (pretraining), check dataset length
                    if isinstance(f['input_ids'], h5py.Group):
                        # Group-based structure (distillation dataset)
                        num_samples = len(f.keys())
                    else:
                        # Direct dataset (pretraining dataset)
                        num_samples = len(f['input_ids'])

                    if num_samples == 0:
                        raise ValidationError(
                            f"Shard is empty (0 sequences): {shard_path}\n\n"
                            f"This should not happen during normal data preparation.\n"
                            f"Delete the shard and regenerate:\n"
                            f"  rm {shard_path}\n"
                            f"  nanoplm data from-yaml"
                        )

            except OSError as e:
                raise ValidationError(
                    f"Cannot open shard file {shard_path.name}: {e}\n\n"
                    f"The file may be corrupted or partially written.\n"
                    f"Delete the shard and regenerate:\n"
                    f"  rm {shard_path}\n"
                    f"  nanoplm data from-yaml"
                )
            except ValidationError:
                # Re-raise our own validation errors
                raise
            except Exception as e:
                raise ValidationError(
                    f"Error validating shard {shard_path.name}: {e}\n\n"
                    f"Unexpected error during validation. The file may be corrupted."
                )

        logger.info(f"✓ All {len(shard_files)} shards validated successfully")

    return shard_files


def validate_pretrain_dataset(dataset_dir: Union[str, Path]) -> dict:
    """
    Validate complete pretraining dataset (manifest + shards).

    Convenience function that validates both the manifest and shard files
    for a pretraining dataset.

    Args:
        dataset_dir: Root directory containing .data_manifest and train/val subdirs

    Returns:
        dict: Validation results including:
            - manifest: Parsed manifest dict
            - train_shards: List of validated training shard paths
            - val_shards: List of validated validation shard paths

    Raises:
        FileNotFoundError: If manifest or directories missing
        ValidationError: If validation fails

    Example:
        >>> result = validate_pretrain_dataset("output/data/pretrain_shards")
        >>> print(f"Train shards: {len(result['train_shards'])}")
        >>> print(f"Val shards: {len(result['val_shards'])}")
    """
    dataset_dir = Path(dataset_dir)

    logger.info(f"Validating pretraining dataset: {dataset_dir}")

    # Validate manifest
    manifest_path = dataset_dir / ".data_manifest"
    manifest = validate_dataset_manifest(manifest_path, expected_mode="pretrain")

    # Get shard directories
    train_dir = dataset_dir / manifest['train_dir']
    val_dir = dataset_dir / manifest['val_dir']

    # Validate train shards
    logger.info("Validating training shards...")
    train_shards = validate_shard_files(train_dir, check_contents=True)

    # Validate validation shards
    logger.info("Validating validation shards...")
    val_shards = validate_shard_files(val_dir, check_contents=True)

    logger.info(f"✓ Pretraining dataset validation complete")
    logger.info(f"  Train shards: {len(train_shards)}")
    logger.info(f"  Val shards: {len(val_shards)}")
    logger.info(f"  Train sequences: {manifest.get('train_sequences', 'unknown')}")
    logger.info(f"  Val sequences: {manifest.get('val_sequences', 'unknown')}")

    return {
        'manifest': manifest,
        'train_shards': train_shards,
        'val_shards': val_shards,
    }


def validate_distillation_dataset(dataset_dir: Union[str, Path]) -> dict:
    """
    Validate complete distillation dataset (manifest + data files).

    Handles both on-the-fly mode (FASTA files) and pre-computed mode (HDF5 shards).
    Convenience function that validates both the manifest and data files.

    Args:
        dataset_dir: Root directory containing .data_manifest and data files

    Returns:
        dict: Validation results including:
            - manifest: Parsed manifest dict
            - mode: 'on_the_fly' or 'pre_computed'
            - train_path: Path to training data (FASTA or shard directory)
            - val_path: Path to validation data (FASTA or shard directory)
            For pre-computed mode only:
            - train_shards: List of validated training shard paths
            - val_shards: List of validated validation shard paths

    Raises:
        FileNotFoundError: If manifest or data files missing
        ValidationError: If validation fails

    Example:
        >>> result = validate_distillation_dataset("output/data/kd_dataset")
        >>> print(f"Mode: {result['mode']}")
        >>> if result['mode'] == 'pre_computed':
        ...     print(f"Train shards: {len(result['train_shards'])}")
    """
    dataset_dir = Path(dataset_dir)

    logger.info(f"Validating distillation dataset: {dataset_dir}")

    # Validate manifest
    manifest_path = dataset_dir / ".data_manifest"
    manifest = validate_dataset_manifest(manifest_path, expected_mode="distillation")

    on_the_fly = manifest.get('on_the_fly', False)

    if on_the_fly:
        # On-the-fly mode: validate FASTA files exist
        logger.info("On-the-fly mode: validating FASTA files...")

        train_fasta = dataset_dir / manifest['train_fasta']
        val_fasta = dataset_dir / manifest['val_fasta']

        if not train_fasta.exists():
            raise FileNotFoundError(
                f"Training FASTA file not found: {train_fasta}\n\n"
                f"The manifest references this file but it doesn't exist.\n"
                f"Check that data preparation completed successfully."
            )

        if not val_fasta.exists():
            raise FileNotFoundError(
                f"Validation FASTA file not found: {val_fasta}\n\n"
                f"The manifest references this file but it doesn't exist.\n"
                f"Check that data preparation completed successfully."
            )

        logger.info(f"✓ Distillation dataset validation complete (on-the-fly mode)")
        logger.info(f"  Train FASTA: {train_fasta.name}")
        logger.info(f"  Val FASTA: {val_fasta.name}")

        return {
            'manifest': manifest,
            'mode': 'on_the_fly',
            'train_path': train_fasta,
            'val_path': val_fasta,
        }

    else:
        # Pre-computed mode: validate HDF5 shards
        logger.info("Pre-computed mode: validating HDF5 shards...")

        train_dir = dataset_dir / manifest['train_dir']
        val_dir = dataset_dir / manifest['val_dir']

        # Validate train shards
        logger.info("Validating training shards...")
        train_shards = validate_shard_files(train_dir, check_contents=True)

        # Validate validation shards
        logger.info("Validating validation shards...")
        val_shards = validate_shard_files(val_dir, check_contents=True)

        logger.info(f"✓ Distillation dataset validation complete (pre-computed mode)")
        logger.info(f"  Train shards: {len(train_shards)}")
        logger.info(f"  Val shards: {len(val_shards)}")
        logger.info(f"  Train sequences: {manifest.get('train_sequences', 'unknown')}")
        logger.info(f"  Val sequences: {manifest.get('val_sequences', 'unknown')}")

        return {
            'manifest': manifest,
            'mode': 'pre_computed',
            'train_path': train_dir,
            'val_path': val_dir,
            'train_shards': train_shards,
            'val_shards': val_shards,
        }
