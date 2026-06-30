"""Utility functions for calculating TF-IDF scores."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_tfidf_scores(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix: csr_matrix,
    candidate_positions: Sequence[int],
) -> NDArray[np.float64]:
    """Calculate TF-IDF scores for a query and a list of candidate positions.

    Args:
        query: The query string.
        vectorizer: The TF-IDF vectorizer.
        matrix: The TF-IDF matrix.
        candidate_positions: The list of candidate positions.

    Returns:
        The TF-IDF scores between the query and the candidate papers.
    """
    cleaned_query = query.lower().strip()
    candidate_indices = np.asarray(candidate_positions)

    if not cleaned_query:
        raise ValueError("Query cannot be empty.")

    query_vector = vectorizer.transform([cleaned_query])
    candidate_matrix = matrix[candidate_indices, :]

    scores = np.asarray(
        cosine_similarity(
            query_vector,
            candidate_matrix,
        ),
        dtype=np.float64,
    ).ravel()

    return scores
