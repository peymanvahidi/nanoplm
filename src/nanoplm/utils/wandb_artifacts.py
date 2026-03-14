from pathlib import Path

import wandb

from nanoplm.utils.logger import logger


_UPLOAD_MARKER_KEY = "_nanoplm_source_snapshot_uploaded"


def _is_repo_root(path: Path) -> bool:
    return (
        (path / "src").is_dir()
        and (path / "pretrain.yaml").is_file()
        and (path / "params.yaml").is_file()
    )


def _resolve_repo_root() -> Path:
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if _is_repo_root(parent):
            return parent

    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        if _is_repo_root(parent):
            return parent

    return cwd


def upload_run_source_snapshot() -> None:
    """Upload run inputs and source tree to the active W&B run."""
    if wandb.run is None:
        return

    try:
        if bool(wandb.run.summary.get(_UPLOAD_MARKER_KEY, False)):
            return
    except Exception:
        pass

    try:
        repo_root = _resolve_repo_root()
        artifact = wandb.Artifact(
            name=f"nanoplm-run-source-{wandb.run.id}",
            type="code",
            description="Full pretrain.yaml, params.yaml, and src/ tree for this run.",
        )

        files_added = 0
        for rel_path in ("pretrain.yaml", "params.yaml"):
            file_path = repo_root / rel_path
            if file_path.is_file():
                artifact.add_file(str(file_path), name=rel_path)
                files_added += 1
            else:
                logger.warning(f"W&B source snapshot skipped missing file: {file_path}")

        src_dir = repo_root / "src"
        if src_dir.is_dir():
            artifact.add_dir(str(src_dir), name="src")
        else:
            logger.warning(f"W&B source snapshot skipped missing directory: {src_dir}")

        if files_added == 0 and not src_dir.is_dir():
            return

        wandb.run.log_artifact(artifact)
        try:
            wandb.run.summary[_UPLOAD_MARKER_KEY] = True
        except Exception:
            pass
        logger.info("Uploaded W&B source snapshot artifact (pretrain.yaml, params.yaml, src/).")
    except Exception as exc:
        logger.warning(f"W&B source snapshot upload failed, continuing without artifact. Error: {exc}")
