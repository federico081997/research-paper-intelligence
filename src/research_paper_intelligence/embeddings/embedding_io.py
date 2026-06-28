"""Load and save embedding matrices."""

from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray


def save_embeddings(
    embeddings: np.ndarray,
    path: Path,
) -> None:
    """Save an embedding matrix to disk.

    Args:
        embeddings: Embedding matrix to save.
        path: Destination path for the embedding matrix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)


def load_embeddings(path: Path) -> NDArray[np.float32]:
    """Load an embedding matrix from a file.

    Args:
        path: Path containing the saved embedding matrix.

    Returns:
        Loaded embedding matrix as a float32 NumPy array.

    Raises:
        FileNotFoundError: If the embedding file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Embeddings were not found at: {path}")

    embeddings = np.load(
        path,
        allow_pickle=False,
    ).astype(np.float32, copy=False)

    return cast(NDArray[np.float32], embeddings)
