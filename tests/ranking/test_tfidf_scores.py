"""Tests for the tfidf_scores module."""

from unittest.mock import Mock

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from research_paper_intelligence.ranking import tfidf_scores


class TestCalculateTfidfScores:
    """Tests the calculate_tfidf_scores function."""

    def test_returns_scores_for_candidate_positions(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tests that it returns scores for candidate positions."""
        vectorizer = Mock()
        query_vector = csr_matrix([[1.0, 0.0]])
        vectorizer.transform.return_value = query_vector

        matrix = csr_matrix(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ]
        )

        cosine_similarity_mock = Mock(return_value=np.array([[0.9, 0.4]]))
        monkeypatch.setattr(
            tfidf_scores,
            "cosine_similarity",
            cosine_similarity_mock,
        )

        result = tfidf_scores.calculate_tfidf_scores(
            query="  MACHINE LEARNING  ",
            vectorizer=vectorizer,
            matrix=matrix,
            candidate_positions=np.array([0, 2], dtype=np.int64),
        )

        np.testing.assert_allclose(result, [0.9, 0.4])
        vectorizer.transform.assert_called_once_with(["machine learning"])

        expected_candidate_matrix = matrix[[0, 2], :]
        actual_candidate_matrix = cosine_similarity_mock.call_args.args[1]

        np.testing.assert_array_equal(
            actual_candidate_matrix.toarray(),
            expected_candidate_matrix.toarray(),
        )

    def test_rejects_empty_query(self) -> None:
        """Tests that it rejects empty queries."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            tfidf_scores.calculate_tfidf_scores(
                query="   ",
                vectorizer=Mock(),
                matrix=csr_matrix((1, 1)),
                candidate_positions=[0],
            )
