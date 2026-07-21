"""Tests for the hybrid ranking module."""

from unittest.mock import Mock

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from research_paper_intelligence.ranking import hybrid_ranking
from research_paper_intelligence.ranking.hybrid_ranking import (
    build_explanation,
)


class TestBuildExplanation:
    """Tests for the build_explanation function."""

    @pytest.mark.parametrize(
        ("semantic_score", "expected_reason"),
        [
            (0.75, "very strong semantic similarity"),
            (0.65, "strong topical similarity"),
            (0.50, "moderate semantic similarity"),
        ],
    )
    def test_builds_semantic_explanations_at_thresholds(
        self,
        semantic_score: float,
        expected_reason: str,
    ) -> None:
        """Tests semantic explanations at the semantic score threshold."""
        result = build_explanation(
            semantic_score=semantic_score,
            keyword_overlap_score=0.0,
            tfidf_score=0.0,
            recency_score=0.0,
        )

        assert result == f"Recommended because of {expected_reason}."

    @pytest.mark.parametrize(
        ("tfidf_score", "expected_reason"),
        [
            (0.75, "strong coverage of important query terms"),
            (0.50, "moderate coverage of important query terms"),
        ],
    )
    def test_builds_tfidf_explanations_at_thresholds(
        self,
        tfidf_score: float,
        expected_reason: str,
    ) -> None:
        """Tests semantic explanations at the TF-IDF score threshold."""
        result = build_explanation(
            semantic_score=0.0,
            keyword_overlap_score=0.0,
            tfidf_score=tfidf_score,
            recency_score=0.0,
        )

        assert result == f"Recommended because of {expected_reason}."

    def test_combines_multiple_reasons(self) -> None:
        """Tests that multiple reasons are combined in the explanation."""
        result = build_explanation(
            semantic_score=0.80,
            keyword_overlap_score=0.60,
            tfidf_score=0.55,
            recency_score=0.80,
        )

        assert result == (
            "Recommended because of very strong semantic similarity, "
            "clear overlap in technical keywords, moderate coverage of "
            "important query terms, and relatively recent publication."
        )

    def test_returns_fallback_when_no_threshold_is_reached(self) -> None:
        """Tests that a fallback explanation is returned."""
        result = build_explanation(
            semantic_score=0.49,
            keyword_overlap_score=0.0,
            tfidf_score=0.49,
            recency_score=0.74,
        )

        assert result == (
            "Recommended based on the overall embedding similarity pattern."
        )

    def test_reports_partial_keyword_overlap(self) -> None:
        """Tests that partial keyword overlap is reported."""
        result = build_explanation(
            semantic_score=0.0,
            keyword_overlap_score=0.20,
            tfidf_score=0.0,
            recency_score=0.0,
        )

        assert result == (
            "Recommended because of some overlap in technical terminology."
        )


class TestHybridSearch:
    """Tests the hybrid search functionality."""

    def test_returns_highest_ranked_results(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tests that it returns the highest-ranked results."""
        papers = {
            10: Mock(name="paper_10"),
            20: Mock(name="paper_20"),
            30: Mock(name="paper_30"),
        }

        repository = Mock(spec=hybrid_ranking.PaperRepository)
        repository.get_by_position.side_effect = lambda position: papers[
            int(position)
        ]

        semantic_search_mock = Mock(
            return_value=(
                np.array([10, 20, 30], dtype=np.int64),
                np.array([0.8, 0.4, 0.6], dtype=np.float64),
            )
        )
        keyword_score_mock = Mock(return_value=np.array([0.1, 0.9, 0.2]))
        tfidf_score_mock = Mock(return_value=np.array([0.2, 0.8, 0.3]))
        recency_score_mock = Mock(return_value=np.array([0.4, 0.3, 0.9]))
        explanation_mock = Mock(return_value="Test explanation.")

        monkeypatch.setattr(
            hybrid_ranking,
            "semantic_search",
            semantic_search_mock,
        )
        monkeypatch.setattr(
            hybrid_ranking,
            "calculate_keyword_overlap_scores",
            keyword_score_mock,
        )
        monkeypatch.setattr(
            hybrid_ranking,
            "calculate_tfidf_scores",
            tfidf_score_mock,
        )
        monkeypatch.setattr(
            hybrid_ranking,
            "calculate_recency_scores",
            recency_score_mock,
        )
        monkeypatch.setattr(
            hybrid_ranking,
            "build_explanation",
            explanation_mock,
        )

        results = hybrid_ranking.hybrid_search(
            query="machine learning",
            paper_repository=repository,
            model=Mock(),
            index=Mock(),
            vectorizer=Mock(),
            tfidf_matrix=csr_matrix((3, 3)),
            candidate_k=3,
            result_k=2,
            semantic_weight=0.5,
            tfidf_weight=0.2,
            keyword_weight=0.2,
            recency_weight=0.1,
            half_life_years=5.0,
        )

        assert len(results) == 2
        assert results[0].paper is papers[20]
        assert results[1].paper is papers[30]
        assert [result.rank for result in results] == [1, 2]

        np.testing.assert_allclose(
            [result.hybrid_score for result in results],
            [0.72, 0.59],
        )

        assert all(
            result.explanation == "Test explanation." for result in results
        )

    def test_rejects_non_positive_result_k(self) -> None:
        """Tests that a non-positive result_k raises a ValueError."""
        with pytest.raises(
            ValueError,
            match="result_k must be a positive integer",
        ):
            hybrid_ranking.hybrid_search(
                query="machine learning",
                paper_repository=Mock(),
                model=Mock(),
                index=Mock(),
                vectorizer=Mock(),
                tfidf_matrix=csr_matrix((0, 0)),
                candidate_k=3,
                result_k=0,
                semantic_weight=0.5,
                tfidf_weight=0.2,
                keyword_weight=0.2,
                recency_weight=0.1,
                half_life_years=5.0,
            )
