"""
Pre-flight validation for dataset manifests and shard files.

Provides comprehensive validation checks before training to catch data issues
early with clear, actionable error messages.

Manifest parsing and field validation is delegated to ``manifest.py``
(via ``read_manifest``).  This module adds **shard-level** validation
and provides convenience wrappers for each pipeline mode.

Two shard formats are supported:
- **Binary shards** (.bin + .idx.npy): Used by pretraining (flat memmap files)
- **HDF5 shards** (.h5): Used by distillation pre-computed mode
"""

from pathlib import Path
from typing import Union, List, Optional
import numpy as np
import h5py

from nanoplm.data.manifest import (
    read_manifest,
    validate_manifest_for_pipeline,
)
from nanoplm.utils import logger


class ValidationError(Exception):
    """
    Custom exception for dataset validation failures.

    Provides clear, actionable error messages that guide users to fix
    data preparation issues before training.
    """
    pass


def validate_hdf5_shards(
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
        >>> shard_paths = validate_hdf5_shards(
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
                    keys = list(f.keys())

                    # Determine structure: group-based (distillation) or flat (pretraining)
                    # Group-based: top-level keys are numeric sample indices ('0', '1', ...)
                    # Flat: top-level keys are dataset names ('input_ids', 'attention_mask', ...)
                    is_grouped = len(keys) > 0 and keys[0].isdigit()

                    if is_grouped:
                        # Validate first group has required datasets
                        first_group = f[keys[0]]
                        if 'input_ids' not in first_group:
                            raise ValidationError(
                                f"Shard group missing 'input_ids' dataset: {shard_path}\n\n"
                                f"This shard file is incomplete or corrupted.\n"
                                f"Delete the shard and regenerate:\n"
                                f"  rm {shard_path}\n"
                                f"  nanoplm data from-yaml"
                            )
                        num_samples = len(keys)
                    else:
                        # Flat structure: check for top-level 'input_ids'
                        if 'input_ids' not in f:
                            raise ValidationError(
                                f"Shard missing 'input_ids' dataset: {shard_path}\n\n"
                                f"This shard file is incomplete or corrupted.\n"
                                f"Delete the shard and regenerate:\n"
                                f"  rm {shard_path}\n"
                                f"  nanoplm data from-yaml"
                            )
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


def validate_pretrain_shards(
    shard_dir: Union[str, Path],
    expected_count: Optional[int] = None,
    check_contents: bool = True,
) -> List[Path]:
    """
    Validate binary pretrain shard files (.bin + .idx.npy) in a directory.

    Each shard consists of a pair:
    - ``shard_NNNN.bin``: concatenated uint8 tokens
    - ``shard_NNNN.idx.npy``: int32 array of per-sequence lengths

    Args:
        shard_dir: Directory containing binary shard files
        expected_count: Expected number of shard pairs (optional)
        check_contents: Whether to validate file contents (index readability,
            non-empty, lengths consistent with .bin size)

    Returns:
        List[Path]: Sorted list of validated ``.bin`` shard paths

    Raises:
        ValidationError: If shards are missing, unpaired, corrupt, or empty
    """
    shard_dir = Path(shard_dir)

    if not shard_dir.exists():
        raise ValidationError(
            f"Shard directory does not exist: {shard_dir}\n\n"
            f"The data preparation may have failed or the path is incorrect.\n"
            f"Run: nanoplm data from-yaml"
        )

    if not shard_dir.is_dir():
        raise ValidationError(
            f"Path is not a directory: {shard_dir}\n\n"
            f"Expected a directory containing binary shard files (.bin + .idx.npy)."
        )

    bin_files = sorted(shard_dir.glob("*.bin"))

    if not bin_files:
        raise ValidationError(
            f"No binary shard files (*.bin) found in: {shard_dir}\n\n"
            f"The directory exists but contains no shard files.\n"
            f"Data preparation may have failed. Check logs and re-run:\n"
            f"  nanoplm data from-yaml"
        )

    if expected_count is not None and len(bin_files) != expected_count:
        raise ValidationError(
            f"Expected {expected_count} shard files but found {len(bin_files)} in: {shard_dir}\n\n"
            f"Found files:\n" + "\n".join(f"  - {f.name}" for f in bin_files[:5]) +
            (f"\n  ... and {len(bin_files) - 5} more" if len(bin_files) > 5 else "") + "\n\n"
            f"Some shard files may be missing. Re-run data preparation:\n"
            f"  nanoplm data from-yaml"
        )

    # Check every .bin has a matching .idx.npy
    for bin_path in bin_files:
        idx_path = bin_path.with_name(bin_path.stem + ".idx.npy")
        if not idx_path.exists():
            raise ValidationError(
                f"Index file missing for shard: {bin_path.name}\n\n"
                f"Expected: {idx_path.name}\n"
                f"The shard pair is incomplete. Re-run data preparation:\n"
                f"  nanoplm data from-yaml"
            )

    if check_contents:
        logger.info(f"Validating {len(bin_files)} binary shards in {shard_dir.name}...")

        for bin_path in bin_files:
            idx_path = bin_path.with_name(bin_path.stem + ".idx.npy")

            try:
                sizes = np.load(str(idx_path))
            except Exception as e:
                raise ValidationError(
                    f"Cannot read index file {idx_path.name}: {e}\n\n"
                    f"The file may be corrupted. Delete and regenerate:\n"
                    f"  rm {bin_path} {idx_path}\n"
                    f"  nanoplm data from-yaml"
                )

            if sizes.ndim != 1:
                raise ValidationError(
                    f"Index file has unexpected shape {sizes.shape}: {idx_path.name}\n\n"
                    f"Expected a 1-D array of sequence lengths."
                )

            num_sequences = len(sizes)
            if num_sequences == 0:
                raise ValidationError(
                    f"Shard is empty (0 sequences): {bin_path.name}\n\n"
                    f"Delete and regenerate:\n"
                    f"  rm {bin_path} {idx_path}\n"
                    f"  nanoplm data from-yaml"
                )

            expected_bytes = int(sizes.sum())
            actual_bytes = bin_path.stat().st_size
            if actual_bytes != expected_bytes:
                raise ValidationError(
                    f"Size mismatch in shard {bin_path.name}: "
                    f"index says {expected_bytes} bytes but .bin is {actual_bytes} bytes.\n\n"
                    f"The shard may be corrupted or partially written. Delete and regenerate:\n"
                    f"  rm {bin_path} {idx_path}\n"
                    f"  nanoplm data from-yaml"
                )

        logger.info(f"✓ All {len(bin_files)} binary shards validated successfully")

    return bin_files


def validate_pretrain_dataset(dataset_dir: Union[str, Path]) -> dict:
    """
    Validate complete pretraining dataset (manifest + binary shards).

    Reads and validates the manifest via ``read_manifest`` (typed dataclass),
    checks pipeline mode, then validates binary shard files (.bin + .idx.npy).

    Args:
        dataset_dir: Root directory containing .data_manifest and train/val subdirs

    Returns:
        dict: Validation results including:
            - manifest: PretrainManifest dataclass
            - train_shards: List of validated training shard paths (.bin)
            - val_shards: List of validated validation shard paths (.bin)

    Raises:
        FileNotFoundError: If manifest or directories missing
        ValidationError: If shard validation fails
        ValueError: If manifest is malformed or wrong pipeline mode

    Example:
        >>> result = validate_pretrain_dataset("output/data/pretrain_shards")
        >>> print(f"Train shards: {len(result['train_shards'])}")
        >>> print(f"Val shards: {len(result['val_shards'])}")
    """
    dataset_dir = Path(dataset_dir)

    logger.info(f"Validating pretraining dataset: {dataset_dir}")

    # Read and validate manifest (typed dataclass with __post_init__ validation)
    manifest = read_manifest(dataset_dir)
    validate_manifest_for_pipeline(manifest, expected_mode="pretrain")

    # Get shard directories from typed manifest
    train_dir = dataset_dir / manifest.train_dir
    val_dir = dataset_dir / manifest.val_dir

    # Validate train shards (binary format: .bin + .idx.npy)
    logger.info("Validating training shards...")
    train_shards = validate_pretrain_shards(train_dir, check_contents=True)

    # Validate validation shards
    logger.info("Validating validation shards...")
    val_shards = validate_pretrain_shards(val_dir, check_contents=True)

    logger.info(f"✓ Pretraining dataset validation complete")
    logger.info(f"  Train shards: {len(train_shards)}")
    logger.info(f"  Val shards: {len(val_shards)}")
    logger.info(f"  Train sequences: {manifest.train_sequences}")
    logger.info(f"  Val sequences: {manifest.val_sequences}")

    return {
        'manifest': manifest,
        'train_shards': train_shards,
        'val_shards': val_shards,
    }


def validate_distillation_dataset(dataset_dir: Union[str, Path]) -> dict:
    """
    Validate complete distillation dataset (manifest + data files).

    Reads and validates the manifest via ``read_manifest`` (typed dataclass),
    checks pipeline mode, then validates data files (FASTA or HDF5 shards).

    Args:
        dataset_dir: Root directory containing .data_manifest and data files

    Returns:
        dict: Validation results including:
            - manifest: DistillationManifest dataclass
            - mode: 'on_the_fly' or 'pre_computed'
            - train_path: Path to training data (FASTA or shard directory)
            - val_path: Path to validation data (FASTA or shard directory)
            For pre-computed mode only:
            - train_shards: List of validated training shard paths
            - val_shards: List of validated validation shard paths

    Raises:
        FileNotFoundError: If manifest or data files missing
        ValidationError: If shard validation fails
        ValueError: If manifest is malformed or wrong pipeline mode

    Example:
        >>> result = validate_distillation_dataset("output/data/kd_dataset")
        >>> print(f"Mode: {result['mode']}")
        >>> if result['mode'] == 'pre_computed':
        ...     print(f"Train shards: {len(result['train_shards'])}")
    """
    dataset_dir = Path(dataset_dir)

    logger.info(f"Validating distillation dataset: {dataset_dir}")

    # Read and validate manifest (typed dataclass with __post_init__ validation)
    manifest = read_manifest(dataset_dir)
    validate_manifest_for_pipeline(manifest, expected_mode="distillation")

    on_the_fly = manifest.on_the_fly

    if on_the_fly:
        # On-the-fly mode: validate FASTA files exist
        logger.info("On-the-fly mode: validating FASTA files...")

        train_fasta = dataset_dir / manifest.train_fasta
        val_fasta = dataset_dir / manifest.val_fasta

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

        train_dir = dataset_dir / manifest.train_dir
        val_dir = dataset_dir / manifest.val_dir

        # Validate train shards
        logger.info("Validating training shards...")
        train_shards = validate_hdf5_shards(train_dir, check_contents=True)

        # Validate validation shards
        logger.info("Validating validation shards...")
        val_shards = validate_hdf5_shards(val_dir, check_contents=True)

        logger.info(f"✓ Distillation dataset validation complete (pre-computed mode)")
        logger.info(f"  Train shards: {len(train_shards)}")
        logger.info(f"  Val shards: {len(val_shards)}")
        logger.info(f"  Train sequences: {manifest.train_sequences}")
        logger.info(f"  Val sequences: {manifest.val_sequences}")

        return {
            'manifest': manifest,
            'mode': 'pre_computed',
            'train_path': train_dir,
            'val_path': val_dir,
            'train_shards': train_shards,
            'val_shards': val_shards,
        }
