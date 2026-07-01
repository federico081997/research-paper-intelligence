"""Utility function for semantic search using FAISS index."""

import faiss
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


def semantic_search(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    candidate_top_k: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Retrieve relevant papers using semantic search.

    Args:
        query: The search query.
        model: The SentenceTransformer model for generating embeddings.
        index: The FAISS index for semantic search.
        candidate_top_k: The number of candidate papers to retrieve.

    Returns:
        A tuple containing the indices of the retrieved papers and their
            similarity scores.

    Raises:
        ValueError: If the query is empty or candidate_top_k is not a positive
            integer.
        RuntimeError: If the FAISS index is empty.
    """
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Query cannot be empty.")

    if candidate_top_k <= 0:
        raise ValueError("candidate_top_k must be a positive integer.")

    if index.ntotal == 0:
        raise RuntimeError("The FAISS index is empty.")

    # Number of candidates to retrieve
    search_k = min(candidate_top_k, index.ntotal)

    # Encode the query
    encoded_query = model.encode(
        [cleaned_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Convert the query embedding to a contiguous array
    query_embedding = np.ascontiguousarray(
        encoded_query,
        dtype=np.float32,
    )

    # Perform the semantic search
    distances, indices = index.search(query_embedding, search_k)

    positions: NDArray[np.int64] = np.array(indices[0], dtype=np.int64)
    scores: NDArray[np.float64] = np.array(distances[0], dtype=np.float64)

    return positions, scores
