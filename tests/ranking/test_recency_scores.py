"""Tests for recency-score calculation."""

from datetime import date, timedelta
from unittest.mock import Mock

import numpy as np
import pytest

from research_paper_intelligence.domain.paper import Paper
from research_paper_intelligence.ranking import recency_scores


class TestCalculateRecencyScore:
    """Tests for calculate_recency_score function."""

    def test_returns_one_for_paper_published_today(self) -> None:
        """Tests that it returns one for the paper published today."""
        score = recency_scores.calculate_recency_score(
            published_date=date.today(),
            half_life_years=2.0,
        )

        assert score == pytest.approx(1.0)

    def test_returns_half_after_one_half_life(self) -> None:
        """Tests that it returns half after one half-life."""
        published_date = date.today() - timedelta(days=365)

        score = recency_scores.calculate_recency_score(
            published_date=published_date,
            half_life_years=1.0,
        )

        assert score == pytest.approx(0.5, abs=0.001)

    def test_returns_one_for_future_date(self) -> None:
        """Tests that it returns one for a future date."""
        future_date = date.today() + timedelta(days=30)

        score = recency_scores.calculate_recency_score(
            published_date=future_date,
            half_life_years=2.0,
        )

        assert score == pytest.approx(1.0)

    @pytest.mark.parametrize("half_life_years", [0.0, -1.0])
    def test_raises_error_for_non_positive_half_life(
        self,
        half_life_years: float,
    ) -> None:
        """Tests that it raises an error when half-life is non-positive."""
        with pytest.raises(
            ValueError,
            match="Half-life years must be positive.",
        ):
            recency_scores.calculate_recency_score(
                published_date=date.today(),
                half_life_years=half_life_years,
            )


class TestCalculateRecencyScores:
    """Tests the calculation of recency scores."""

    def test_calculates_score_for_each_candidate(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tests that it calculates scores for each candidate."""
        papers = [Mock(spec=Paper), Mock(spec=Paper)]
        papers[0].published_date = date(2025, 1, 1)
        papers[1].published_date = date(2020, 1, 1)

        score_mock = Mock(side_effect=[0.9, 0.4])
        monkeypatch.setattr(
            recency_scores,
            "calculate_recency_score",
            score_mock,
        )

        result = recency_scores.calculate_recency_scores(
            candidate_papers=papers,
            half_life_years=5.0,
        )

        np.testing.assert_allclose(
            result,
            np.array([0.9, 0.4], dtype=np.float64),
        )

        assert score_mock.call_count == 2
        score_mock.assert_any_call(date(2025, 1, 1), 5.0)
        score_mock.assert_any_call(date(2020, 1, 1), 5.0)

    def test_returns_empty_array_for_no_candidates(self) -> None:
        """Tests that it returns an empty array for no candidates."""
        result = recency_scores.calculate_recency_scores(
            candidate_papers=[],
            half_life_years=5.0,
        )

        assert result.size == 0
