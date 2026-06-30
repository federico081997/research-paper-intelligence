"""Tests for loading and saving TF-IDF indices."""

from pathlib import Path

import joblib
import numpy as np
import pytest
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from research_paper_intelligence.storage.tfidf_index_io import (
    load_tfidf_index,
    save_tfidf_index,
)


@pytest.fixture
def tfidf_artifacts() -> tuple[TfidfVectorizer, csr_matrix]:
    """Return a fitted vectorizer and TF-IDF matrix."""
    texts = [
        "finite volume numerical methods",
        "machine learning semantic search",
    ]

    vectorizer = TfidfVectorizer()
    matrix = csr_matrix(vectorizer.fit_transform(texts))

    return vectorizer, matrix


class TestSaveTfidfIndex:
    """Tests for save_tfidf_index."""

    def test_saves_vectorizer_and_matrix(
        self,
        tmp_path: Path,
        tfidf_artifacts: tuple[TfidfVectorizer, csr_matrix],
    ) -> None:
        """Tests that it saves the vectorizer and matrix."""
        vectorizer, matrix = tfidf_artifacts
        vectorizer_path = tmp_path / "tfidf" / "vectorizer.joblib"
        matrix_path = tmp_path / "tfidf" / "matrix.npz"

        save_tfidf_index(
            vectorizer=vectorizer,
            matrix=matrix,
            vectorizer_path=vectorizer_path,
            matrix_path=matrix_path,
        )

        assert vectorizer_path.exists()
        assert matrix_path.exists()


class TestLoadTfidfIndex:
    """Tests for load_tfidf_index."""

    def test_loads_vectorizer_and_matrix(
        self,
        tmp_path: Path,
        tfidf_artifacts: tuple[TfidfVectorizer, csr_matrix],
    ) -> None:
        """Tests that it loads the vectorizer and matrix."""
        vectorizer, matrix = tfidf_artifacts
        vectorizer_path = tmp_path / "vectorizer.joblib"
        matrix_path = tmp_path / "matrix.npz"

        joblib.dump(vectorizer, vectorizer_path)
        save_npz(matrix_path, matrix)

        loaded_vectorizer, loaded_matrix = load_tfidf_index(
            vectorizer_path=vectorizer_path,
            matrix_path=matrix_path,
        )

        assert isinstance(loaded_vectorizer, TfidfVectorizer)
        assert isinstance(loaded_matrix, csr_matrix)
        assert loaded_vectorizer.vocabulary_ == vectorizer.vocabulary_
        assert loaded_matrix.shape == matrix.shape
        np.testing.assert_array_equal(
            loaded_matrix.toarray(), matrix.toarray()
        )

    def test_raises_error_when_vectorizer_is_missing(
        self,
        tmp_path: Path,
    ) -> None:
        """Tests that it raises an error when the vectorizer is missing."""
        vectorizer_path = tmp_path / "missing_vectorizer.joblib"
        matrix_path = tmp_path / "matrix.npz"

        with pytest.raises(
            FileNotFoundError,
            match=f"TF-IDF vectorizer file not found at {vectorizer_path}",
        ):
            load_tfidf_index(vectorizer_path, matrix_path)

    def test_raises_error_when_matrix_is_missing(
        self,
        tmp_path: Path,
        tfidf_artifacts: tuple[TfidfVectorizer, csr_matrix],
    ) -> None:
        """Tests that it raises an error when the matrix is missing."""
        vectorizer, _ = tfidf_artifacts
        vectorizer_path = tmp_path / "vectorizer.joblib"
        matrix_path = tmp_path / "missing_matrix.npz"

        joblib.dump(vectorizer, vectorizer_path)

        with pytest.raises(
            FileNotFoundError,
            match=f"TF-IDF matrix file not found at {matrix_path}",
        ):
            load_tfidf_index(vectorizer_path, matrix_path)
