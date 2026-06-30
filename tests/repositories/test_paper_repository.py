"""Tests for the PaperRepository class."""

from datetime import date
from unittest.mock import call, patch

import pandas as pd
import pytest

from research_paper_intelligence.domain.paper import Paper
from research_paper_intelligence.repositories.paper_repository import (
    PaperRepository,
)

# -----------------------------------------------------------------------------
#   TestPaperRepository
# -----------------------------------------------------------------------------


class TestPaperRepository:
    """Tests for the PaperRepository class."""

    @pytest.fixture
    def paper_dataframe(self) -> pd.DataFrame:
        """Create a sample processed-paper DataFrame."""
        return pd.DataFrame(
            {
                "id": ["paper-001", "paper-002"],
                "title": [
                    "Finite Volume Methods",
                    "Machine Learning for Engineering",
                ],
                "summary": [
                    "A paper about finite volume methods.",
                    "A paper about machine learning.",
                ],
                "authors": [
                    "Alice Smith, Bob Jones",
                    "Carol Brown",
                ],
                "category": [
                    "Computational Engineering",
                    "Machine Learning",
                ],
                "published_date": [
                    date(2024, 1, 15),
                    date(2025, 6, 20),
                ],
            },
            index=[10, 20],
        )

    def test_get_by_position_returns_expected_paper(
        self,
        paper_dataframe: pd.DataFrame,
    ) -> None:
        """get_by_position should convert a DataFrame row into a Paper."""
        repository = PaperRepository(paper_dataframe)

        result = repository.get_by_position(0)

        assert isinstance(result, Paper)
        assert result.paper_id == "paper-001"
        assert result.title == "Finite Volume Methods"
        assert result.abstract == "A paper about finite volume methods."
        assert result.authors == "Alice Smith, Bob Jones"
        assert result.category == "Computational Engineering"
        assert result.published_date == date(2024, 1, 15)

    def test_get_by_position_converts_values_to_strings(self) -> None:
        """Repository should convert paper fields to strings."""
        dataframe = pd.DataFrame(
            {
                "id": [123],
                "title": [456],
                "summary": [789],
                "authors": [101],
                "category": [202],
                "published_date": ["2024-01-15"],
            }
        )
        repository = PaperRepository(dataframe)

        result = repository.get_by_position(0)

        assert result.paper_id == "123"
        assert result.title == "456"
        assert result.abstract == "789"
        assert result.authors == "101"
        assert result.category == "202"

    def test_get_by_id_returns_expected_paper(
        self,
        paper_dataframe: pd.DataFrame,
    ) -> None:
        """get_by_id should return the paper with the matching ID."""
        repository = PaperRepository(paper_dataframe)

        result = repository.get_by_id("paper-002")

        assert result is not None
        assert result.paper_id == "paper-002"
        assert result.title == "Machine Learning for Engineering"
        assert result.published_date == date(2025, 6, 20)

    def test_get_by_id_returns_none_when_paper_does_not_exist(
        self,
        paper_dataframe: pd.DataFrame,
    ) -> None:
        """get_by_id should return None when no paper matches the ID."""
        repository = PaperRepository(paper_dataframe)

        result = repository.get_by_id("unknown-paper")

        assert result is None

    def test_constructor_resets_dataframe_index(
        self,
        paper_dataframe: pd.DataFrame,
    ) -> None:
        """Repository should reset the DataFrame index."""
        repository = PaperRepository(paper_dataframe)

        result = repository.get_by_id("paper-001")

        assert result is not None
        assert result.paper_id == "paper-001"

    def test_get_by_position_raises_index_error_for_invalid_position(
        self,
        paper_dataframe: pd.DataFrame,
    ) -> None:
        """get_by_position should raise IndexError for an invalid position."""
        repository = PaperRepository(paper_dataframe)

        with pytest.raises(IndexError):
            repository.get_by_position(10)

    def test_get_by_position_raises_error_for_invalid_date(self) -> None:
        """get_by_position should reject an invalid published date."""
        dataframe = pd.DataFrame(
            {
                "id": ["paper-001"],
                "title": ["Example paper"],
                "summary": ["Example abstract"],
                "authors": ["Example Author"],
                "category": ["Example Category"],
                "published_date": ["not-a-valid-date"],
            }
        )
        repository = PaperRepository(dataframe)

        with pytest.raises(ValueError):
            repository.get_by_position(0)

    def test_get_all_returns_papers_in_dataset_order(
        self,
        paper_dataframe: pd.DataFrame,
    ) -> None:
        """get_all should return papers in dataset order."""
        repository = PaperRepository(paper_dataframe)

        papers = [
            Paper(
                paper_id="paper-001",
                title="Finite Volume Methods",
                abstract="A paper about finite volume methods",
                authors="Alice Smith, Bob Jones",
                category="Computational Engineering",
                published_date=date(2024, 1, 15),
            ),
            Paper(
                paper_id="paper-002",
                title="Machine Learning for Engineering",
                abstract="A paper about machine learning",
                authors="Carol Brown",
                category="Machine Learning",
                published_date=date(2025, 6, 20),
            ),
        ]

        with patch.object(
            repository,
            "get_by_position",
            side_effect=papers,
        ) as mock_get_by_position:
            result = repository.get_all()

        assert result == papers
        assert mock_get_by_position.call_args_list == [
            call(0),
            call(1),
        ]

    def test_get_all_returns_empty_list_when_repository_is_empty(self) -> None:
        """get_all should return an empty list when the repository is empty."""
        repository = PaperRepository(pd.DataFrame())

        result = repository.get_all()

        assert result == []
