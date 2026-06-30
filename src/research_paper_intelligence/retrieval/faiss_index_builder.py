"""Utility functions for indexing embeddings with FAISS."""

import faiss
import numpy as np


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS index from the given embeddings.

    Args:
        embeddings: The embeddings to index.

    Returns:
        The populated FAISS index.
    """
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index
