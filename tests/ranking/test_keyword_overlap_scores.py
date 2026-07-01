"""Tests keyword overlap scores function."""

from unittest.mock import Mock

import numpy as np

from research_paper_intelligence.domain.paper import Paper
from research_paper_intelligence.ranking.keyword_overlap_scores import (
    calculate_keyword_overlap_scores,
)


class TestCalculateKeywordOverlapScores:
    """Tests keyword overlap scores function."""

    def test_calculates_overlap_for_candidate_papers(self) -> None:
        """Tests a successful calculation of the keyword overlap scores."""
        vectorizer = Mock()
        vectorizer.build_analyzer.return_value = lambda text: (
            text.lower().split()
        )

        paper_one = Mock(spec=Paper)
        paper_one.title = "Machine learning"
        paper_one.abstract = "prediction methods"

        paper_two = Mock(spec=Paper)
        paper_two.title = "Fluid dynamics"
        paper_two.abstract = "numerical simulation"

        result = calculate_keyword_overlap_scores(
            query="machine learning fluid",
            candidate_papers=[paper_one, paper_two],
            vectorizer=vectorizer,
        )

        np.testing.assert_allclose(
            result,
            np.array([2 / 3, 1 / 3], dtype=np.float64),
        )

    def test_returns_zeros_when_query_has_no_valid_terms(self) -> None:
        """Tests that zeros are returned when the query has no valid terms."""
        vectorizer = Mock()
        vectorizer.build_analyzer.return_value = lambda text: []

        papers = [Mock(spec=Paper), Mock(spec=Paper)]

        result = calculate_keyword_overlap_scores(
            query="the and of",
            candidate_papers=papers,
            vectorizer=vectorizer,
        )

        np.testing.assert_array_equal(
            result,
            np.zeros(2, dtype=np.float64),
        )
