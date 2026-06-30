"""Tests for recency-score calculation."""

from datetime import date, timedelta

import pytest

from research_paper_intelligence.ranking.recency_score import (
    calculate_recency_score,
)


class TestCalculateRecencyScore:
    """Tests for calculate_recency_score function."""

    def test_returns_one_for_paper_published_today(self) -> None:
        """Tests that it returns one for the paper published today."""
        score = calculate_recency_score(
            published_date=date.today(),
            half_life_years=2.0,
        )

        assert score == pytest.approx(1.0)

    def test_returns_half_after_one_half_life(self) -> None:
        """Tests that it returns half after one half-life."""
        published_date = date.today() - timedelta(days=365)

        score = calculate_recency_score(
            published_date=published_date,
            half_life_years=1.0,
        )

        assert score == pytest.approx(0.5, abs=0.001)

    def test_returns_one_for_future_date(self) -> None:
        """Tests that it returns one for future date."""
        future_date = date.today() + timedelta(days=30)

        score = calculate_recency_score(
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
            calculate_recency_score(
                published_date=date.today(),
                half_life_years=half_life_years,
            )
