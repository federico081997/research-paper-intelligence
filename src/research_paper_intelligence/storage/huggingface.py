"""Download project datasets from Hugging Face Hub."""

import logging
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


def download_file(
    repository_id: str,
    remote_filename: str,
    destination: Path,
    force: bool = False,
) -> Path:
    """Download a dataset file to its configured local location.

    Args:
        repository_id: Hugging Face dataset repository identifier.
        remote_filename: File path inside the remote repository.
        destination: Local path where the file should be stored.
        force: Replace the local file when it already exists.

    Returns:
        Path to the downloaded local file.
    """
    if destination.exists() and not force:
        logger.debug("Dataset already exists at %s", destination)
        return destination

    logger.info(
        "Downloading %s from %s",
        remote_filename,
        repository_id,
    )

    cached_path = Path(
        hf_hub_download(
            repo_id=repository_id,
            filename=remote_filename,
            repo_type="dataset",
            force_download=force,
        )
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_path, destination)

    logger.info("Dataset saved to %s", destination)

    return destination
