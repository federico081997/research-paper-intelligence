"""Download project datasets from Hugging Face Hub."""

import logging
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError

logger = logging.getLogger(__name__)


def download_file(
    repository_id: str,
    remote_filename: str,
    destination: Path,
    force: bool = False,
    missing_ok: bool = False,
) -> Path | None:
    """Download a dataset file to its configured local location.

    Args:
        repository_id: Hugging Face dataset repository identifier.
        remote_filename: File path inside the remote repository.
        destination: Local path where the file should be stored.
        force: Replace the local file when it already exists.
        missing_ok: Return None when the remote file does not exist.

    Returns:
        Path to the local file, or None when the remote file is missing
        and missing_ok is True.

    Raises:
        RemoteEntryNotFoundError: If the remote file does not exist and
            missing_ok is False.
    """
    if destination.exists() and not force:
        logger.debug(
            "Using existing local file %s.",
            remote_filename,
            destination,
        )
        return destination

    logger.info(
        "Attempting to download %s from repository %s.",
        remote_filename,
        repository_id,
    )

    try:
        cached_path = Path(
            hf_hub_download(
                repo_id=repository_id,
                filename=remote_filename,
                repo_type="dataset",
                force_download=force,
            )
        )
    except RemoteEntryNotFoundError:
        if not missing_ok:
            raise

        logger.info(
            "Remote file %s was not found in repository %s.",
            remote_filename,
            repository_id,
        )
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_path, destination)

    logger.info(
        "%s has been successfully saved to %s.",
        remote_filename,
        destination,
    )

    return destination
