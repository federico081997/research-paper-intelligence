"""FAISS index IO utilities."""

from pathlib import Path

import faiss


def load_faiss_index(path: Path) -> faiss.Index:
    """Load a FAISS index from a file.

    Args:
        path: Path to the FAISS index file.

    Returns:
        The loaded FAISS index.
    """
    if not path.exists():
        raise FileNotFoundError(f"FAISS index file not found at {path}")

    return faiss.read_index(str(path))


def save_faiss_index(index: faiss.Index, output_path: Path) -> None:
    """Save a FAISS index to the disk.

    Args:
        index: The FAISS index to save.
        output_path: The path to save the FAISS index to.
    """
    faiss.write_index(index, str(output_path))
