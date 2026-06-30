"""Utilities for reading and writing TF-IDF indices."""

from pathlib import Path
from typing import cast

import joblib
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


def save_tfidf_index(
    vectorizer: TfidfVectorizer,
    matrix: csr_matrix,
    vectorizer_path: Path,
    matrix_path: Path,
) -> None:
    """Save the fitted TF-IDF vectorizer and sparse paper matrix to disk.

    Args:
        vectorizer: TF-IDF Vectorizer to save.
        matrix: TF-IDF matrix to save.
        vectorizer_path: Path to the TF-IDF vectorizer file.
        matrix_path: Path to the TF-IDF matrix file.
    """
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, vectorizer_path)
    save_npz(matrix_path, matrix)


def load_tfidf_index(
    vectorizer_path: Path,
    matrix_path: Path,
) -> tuple[TfidfVectorizer, csr_matrix]:
    """Load the fitted TF-IDF vectorizer and sparse matrix from the disk.

    Args:
        vectorizer_path: Path to the TF-IDF vectorizer file.
        matrix_path: Path to the TF-IDF matrix file.
    """
    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer file not found at {vectorizer_path}"
        )

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"TF-IDF matrix file not found at {matrix_path}"
        )

    vectorizer = cast(TfidfVectorizer, joblib.load(vectorizer_path))
    matrix = csr_matrix(load_npz(matrix_path))

    return vectorizer, matrix
