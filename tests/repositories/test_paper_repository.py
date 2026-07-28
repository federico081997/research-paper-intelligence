"""Tests for the processed-paper repository."""

from datetime import date

import pandas as pd
import pytest

from research_paper_intelligence.domain.paper import Paper
from research_paper_intelligence.repositories.paper_repository import (
    PaperRepository,
)


@pytest.fixture
def repository_dataframe() -> pd.DataFrame:
    """Create representative processed paper data."""
    return pd.DataFrame(
        {
            "id": [
                "2401.12345",
                "cs/9308101v1",
                "2502.54321",
            ],
            "title": [
                "Finite volume methods",
                "Machine learning methods",
                "Semantic paper retrieval",
            ],
            "summary": [
                "An abstract about finite volume methods.",
                "An abstract about machine learning.",
                "An abstract about semantic retrieval.",
            ],
            "authors": [
                "Author One, Author Two",
                "Author Three",
                "Author Four, Author Five",
            ],
            "category": [
                "Computational Engineering",
                "Machine Learning",
                "Information Retrieval",
            ],
            "published_date": [
                "2025-01-15",
                pd.Timestamp("2024-06-10"),
                "2023-11-05T14:30:00",
            ],
        },
        index=[10, 20, 30],
    )


@pytest.fixture
def paper_repository(
    repository_dataframe: pd.DataFrame,
) -> PaperRepository:
    """Create a repository containing representative papers."""
    return PaperRepository(repository_dataframe)


class TestPaperRepositoryInitialization:
    """Tests for PaperRepository initialisation."""

    def test_resets_dataframe_index(
        self,
        repository_dataframe: pd.DataFrame,
    ) -> None:
        """Reset the supplied DataFrame index to sequential positions."""
        repository = PaperRepository(repository_dataframe)

        assert repository.dataframe.index.tolist() == [0, 1, 2]

    def test_preserves_dataframe_rows(
        self,
        repository_dataframe: pd.DataFrame,
    ) -> None:
        """Preserve all rows when resetting the DataFrame index."""
        repository = PaperRepository(repository_dataframe)

        assert repository.dataframe["id"].tolist() == [
            "2401.12345",
            "cs/9308101v1",
            "2502.54321",
        ]

    def test_does_not_modify_input_dataframe_index(
        self,
        repository_dataframe: pd.DataFrame,
    ) -> None:
        """Leave the index of the supplied DataFrame unchanged."""
        PaperRepository(repository_dataframe)

        assert repository_dataframe.index.tolist() == [10, 20, 30]


class TestPaperRepositoryLength:
    """Tests for the repository length implementation."""

    def test_returns_number_of_papers(
        self,
        paper_repository: PaperRepository,
    ) -> None:
        """Return the number of rows stored in the repository."""
        assert len(paper_repository) == 3

    def test_returns_zero_for_empty_repository(self) -> None:
        """Return zero when the repository contains no papers."""
        dataframe = pd.DataFrame(
            columns=[
                "id",
                "title",
                "summary",
                "authors",
                "category",
                "published_date",
            ]
        )
        repository = PaperRepository(dataframe)

        assert len(repository) == 0


class TestGetByPosition:
    """Tests for retrieving papers by DataFrame position."""

    def test_returns_paper_at_requested_position(
        self,
        paper_repository: PaperRepository,
    ) -> None:
        """Return a domain paper constructed from the requested row."""
        result = paper_repository.get_by_position(0)

        assert isinstance(result, Paper)
        assert result.paper_id == "2401.12345"
        assert result.title == "Finite volume methods"
        assert result.abstract == ("An abstract about finite volume methods.")
        assert result.authors == "Author One, Author Two"
        assert result.category == "Computational Engineering"
        assert result.published_date == date(2025, 1, 15)

    def test_uses_positional_indexing_after_index_reset(
        self,
        paper_repository: PaperRepository,
    ) -> None:
        """Retrieve papers according to dataset position."""
        result = paper_repository.get_by_position(1)

        assert result.paper_id == "cs/9308101v1"
        assert result.title == "Machine learning methods"

    @pytest.mark.parametrize(
        ("position", "expected_paper_id"),
        [
            (0, "2401.12345"),
            (1, "cs/9308101v1"),
            (2, "2502.54321"),
        ],
    )
    def test_returns_expected_paper_for_each_position(
        self,
        paper_repository: PaperRepository,
        position: int,
        expected_paper_id: str,
    ) -> None:
        """Return the paper corresponding to each valid position."""
        result = paper_repository.get_by_position(position)

        assert result.paper_id == expected_paper_id

    def test_converts_publication_timestamp_to_date(
        self,
        paper_repository: PaperRepository,
    ) -> None:
        """Convert pandas timestamps to standard date values."""
        result = paper_repository.get_by_position(1)

        assert result.published_date == date(2024, 6, 10)
        assert type(result.published_date) is date

    def test_converts_datetime_string_to_date(
        self,
        paper_repository: PaperRepository,
    ) -> None:
        """Discard the time component of publication datetime strings."""
        result = paper_repository.get_by_position(2)

        assert result.published_date == date(2023, 11, 5)

    def test_converts_paper_fields_to_strings(self) -> None:
        """Convert scalar paper metadata into string values."""
        dataframe = pd.DataFrame(
            {
                "id": [12345],
                "title": [678],
                "summary": [3.14],
                "authors": [42],
                "category": [7],
                "published_date": ["2025-01-15"],
            }
        )
        repository = PaperRepository(dataframe)

        result = repository.get_by_position(0)

        assert result.paper_id == "12345"
        assert result.title == "678"
        assert result.abstract == "3.14"
        assert result.authors == "42"
        assert result.category == "7"

    def test_supports_negative_positions(
        self,
        paper_repository: PaperRepository,
    ) -> None:
        """Use pandas positional semantics for negative positions."""
        result = paper_repository.get_by_position(-1)

        assert result.paper_id == "2502.54321"

    @pytest.mark.parametrize("position", [3, 10, -4])
    def test_rejects_out_of_range_position(
        self,
        paper_repository: PaperRepository,
        position: int,
    ) -> None:
        """Raise an IndexError for positions outside the repository."""
        with pytest.raises(IndexError):
            paper_repository.get_by_position(position)

    def test_rejects_invalid_publication_date(self) -> None:
        """Raise an error when a publication date cannot be parsed."""
        dataframe = pd.DataFrame(
            {
                "id": ["2401.12345"],
                "title": ["Paper title"],
                "summary": ["Paper abstract"],
                "authors": ["Author One"],
                "category": ["Machine Learning"],
                "published_date": ["not-a-date"],
            }
        )
        repository = PaperRepository(dataframe)

        with pytest.raises(ValueError):
            repository.get_by_position(0)


class TestGetById:
    """Tests for retrieving papers by identifier."""

    def test_returns_matching_paper(
        self,
        paper_repository: PaperRepository,
    ) -> None:
        """Return the paper with the supplied identifier."""
        result = paper_repository.get_by_id("cs/9308101v1")

        assert result is not None
        assert result.paper_id == "cs/9308101v1"
        assert result.title == "Machine learning methods"
        assert result.published_date == date(2024, 6, 10)

    def test_returns_none_when_identifier_is_not_found(
        self,
        paper_repository: PaperRepository,
    ) -> None:
        """Return None when no paper has the supplied identifier."""
        result = paper_repository.get_by_id("missing-paper")

        assert result is None

    def test_matches_identifiers_exactly(
        self,
        paper_repository: PaperRepository,
    ) -> None:
        """Use exact and case-sensitive identifier matching."""
        result = paper_repository.get_by_id("CS/9308101V1")

        assert result is None

    def test_returns_first_match_for_duplicate_identifiers(self) -> None:
        """Return the first paper when duplicate identifiers are present."""
        dataframe = pd.DataFrame(
            {
                "id": [
                    "2401.12345",
                    "2401.12345",
                ],
                "title": [
                    "First paper",
                    "Second paper",
                ],
                "summary": [
                    "First abstract",
                    "Second abstract",
                ],
                "authors": [
                    "Author One",
                    "Author Two",
                ],
                "category": [
                    "cs.LG",
                    "cs.AI",
                ],
                "published_date": [
                    "2025-01-10",
                    "2025-01-11",
                ],
            }
        )
        repository = PaperRepository(dataframe)

        result = repository.get_by_id("2401.12345")

        assert result is not None
        assert result.title == "First paper"


class TestGetAll:
    """Tests for retrieving all repository papers."""

    def test_returns_every_paper(
        self,
        paper_repository: PaperRepository,
    ) -> None:
        """Return one domain paper for every repository row."""
        result = paper_repository.get_all()

        assert len(result) == 3
        assert all(isinstance(paper, Paper) for paper in result)

    def test_preserves_dataset_order(
        self,
        paper_repository: PaperRepository,
    ) -> None:
        """Return papers in their original dataset order."""
        result = paper_repository.get_all()

        assert [paper.paper_id for paper in result] == [
            "2401.12345",
            "cs/9308101v1",
            "2502.54321",
        ]

    def test_returns_empty_list_for_empty_repository(self) -> None:
        """Return an empty list when no papers are stored."""
        dataframe = pd.DataFrame(
            columns=[
                "id",
                "title",
                "summary",
                "authors",
                "category",
                "published_date",
            ]
        )
        repository = PaperRepository(dataframe)

        result = repository.get_all()

        assert result == []
