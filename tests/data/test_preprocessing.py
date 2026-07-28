"""Tests for the dataset preprocessing utilities."""

from datetime import datetime

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from research_paper_intelligence.data.preprocessing import (
    REQUIRED_COLUMNS,
    clean_text,
    normalize_arxiv_id,
    parse_authors,
    preprocess_dataset,
)


@pytest.fixture
def raw_dataframe() -> pd.DataFrame:
    """Create representative raw research-paper data."""
    return pd.DataFrame(
        {
            "id": [
                "abs-2401.12345",
                "cs-9308101v1",
            ],
            "title": [
                "  Finite   volume\nmethods  ",
                "Machine learning",
            ],
            "summary": [
                "  A finite-volume\tstudy.  ",
                "A machine-learning study.",
            ],
            "category": [
                "  Computational   Engineering ",
                " cs.LG ",
            ],
            "authors": [
                "['Author One', 'Author Two']",
                ("Author Three", "Author Four"),
            ],
            "published_date": [
                "2025-01-10",
                "2024-02-15T12:30:00",
            ],
            "unused_column": [
                "unused value 1",
                "unused value 2",
            ],
        }
    )


class TestCleanText:
    """Tests for the clean_text function."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("Paper title", "Paper title"),
            ("  Paper title  ", "Paper title"),
            ("Paper    title", "Paper title"),
            ("Paper\n\ttitle", "Paper title"),
            ("  Paper\n title\twith   spaces  ", "Paper title with spaces"),
            ("", ""),
            ("   ", ""),
            (None, ""),
            (pd.NA, ""),
            (float("nan"), ""),
            (123, "123"),
            (12.5, "12.5"),
            (True, "True"),
        ],
    )
    def test_cleans_text_values(
        self,
        value: object,
        expected: str,
    ) -> None:
        """Normalize whitespace and convert values to strings."""
        result = clean_text(value)

        assert result == expected

    def test_preserves_internal_punctuation(self) -> None:
        """Preserve punctuation while normalizing whitespace."""
        result = clean_text(
            "  Finite-volume methods: a review.  "
        )

        assert result == "Finite-volume methods: a review."


class TestParseAuthors:
    """Tests for the parse_authors function."""

    @pytest.mark.parametrize(
        ("authors", "expected"),
        [
            (None, ""),
            (pd.NA, ""),
            (float("nan"), ""),
            ("", ""),
            ("   ", ""),
            (
                ["Author One", "Author Two"],
                "Author One, Author Two",
            ),
            (
                ("Author One", "Author Two"),
                "Author One, Author Two",
            ),
            (
                [" Author One ", "", "  ", "Author Two"],
                "Author One, Author Two",
            ),
            (
                (" Author One ", "Author Two "),
                "Author One, Author Two",
            ),
            (
                "['Author One', 'Author Two']",
                "Author One, Author Two",
            ),
            (
                "('Author One', 'Author Two')",
                "Author One, Author Two",
            ),
            (
                "[' Author One ', ' Author Two ']",
                "Author One, Author Two",
            ),
            (
                "'Author One'",
                "Author One",
            ),
            (
                "Author One, Author Two",
                "Author One, Author Two",
            ),
            (
                "  Author One, Author Two  ",
                "Author One, Author Two",
            ),
            (
                "123",
                "123",
            ),
            (
                123,
                "123",
            ),
        ],
    )
    def test_converts_authors_to_readable_string(
        self,
        authors: object,
        expected: str,
    ) -> None:
        """Convert supported author representations to readable text."""
        result = parse_authors(authors)

        assert result == expected

    def test_preserves_malformed_serialised_list(self) -> None:
        """Preserve a string that cannot be parsed as a Python literal."""
        authors = "['Author One', 'Author Two'"

        result = parse_authors(authors)

        assert result == authors

    def test_omits_empty_authors_from_sequence(self) -> None:
        """Exclude empty values from list and tuple inputs."""
        result = parse_authors(
            ["Author One", "", "   ", "Author Two"]
        )

        assert result == "Author One, Author Two"


class TestNormalizeArxivId:
    """Tests for the normalize_arxiv_id function."""

    @pytest.mark.parametrize(
        ("paper_id", "expected"),
        [
            (
                "abs-0901.4761",
                "0901.4761",
            ),
            (
                "abs-0901.4761v1",
                "0901.4761v1",
            ),
            (
                "0901.4761",
                "0901.4761",
            ),
            (
                "2401.12345",
                "2401.12345",
            ),
            (
                "2401.12345v3",
                "2401.12345v3",
            ),
            (
                "  abs-2401.12345v2  ",
                "2401.12345v2",
            ),
        ],
    )
    def test_normalises_modern_arxiv_ids(
        self,
        paper_id: str,
        expected: str,
    ) -> None:
        """Normalize modern arXiv identifiers."""
        result = normalize_arxiv_id(paper_id)

        assert result == expected

    @pytest.mark.parametrize(
        ("paper_id", "expected"),
        [
            (
                "cs-9308101",
                "cs/9308101",
            ),
            (
                "cs-9308101v1",
                "cs/9308101v1",
            ),
            (
                "hep-th-9901001",
                "hep-th/9901001",
            ),
            (
                "hep-th-9901001v2",
                "hep-th/9901001v2",
            ),
            (
                "  math-0301001v3  ",
                "math/0301001v3",
            ),
        ],
    )
    def test_normalises_legacy_arxiv_ids(
        self,
        paper_id: str,
        expected: str,
    ) -> None:
        """Convert legacy dataset identifiers to arXiv path format."""
        result = normalize_arxiv_id(paper_id)

        assert result == expected

    def test_rejects_empty_identifier(self) -> None:
        """Reject an empty paper identifier."""
        with pytest.raises(
            ValueError,
            match="paper_id must not be empty",
        ):
            normalize_arxiv_id("")

    def test_rejects_whitespace_only_identifier(self) -> None:
        """Reject a paper identifier containing only whitespace."""
        with pytest.raises(
            ValueError,
            match="paper_id must not be empty",
        ):
            normalize_arxiv_id("   ")

    @pytest.mark.parametrize(
        "paper_id",
        [
            "invalid-id",
            "abs-invalid",
            "2401.123",
            "2401.123456",
            "2401.12345v",
            "cs-123456",
            "cs-12345678",
        ],
    )
    def test_rejects_unsupported_identifier_formats(
        self,
        paper_id: str,
    ) -> None:
        """Reject identifiers that do not match supported arXiv formats."""
        with pytest.raises(
            ValueError,
            match="Unsupported arXiv paper ID format",
        ):
            normalize_arxiv_id(paper_id)

    def test_rejects_modern_id_without_dot_separator(self) -> None:
        """Require a literal dot in modern arXiv identifiers."""
        with pytest.raises(
            ValueError,
            match="Unsupported arXiv paper ID format",
        ):
            normalize_arxiv_id("2401x12345")


class TestPreprocessDataset:
    """Tests for the preprocess_dataset function."""

    def test_keeps_only_required_columns(
        self,
        raw_dataframe: pd.DataFrame,
    ) -> None:
        """Remove columns that are not required by the application."""
        result = preprocess_dataset(raw_dataframe)

        assert set(result.columns) == REQUIRED_COLUMNS
        assert "unused_column" not in result.columns

    @pytest.mark.parametrize(
        "missing_column",
        sorted(REQUIRED_COLUMNS),
    )
    def test_rejects_missing_required_column(
        self,
        raw_dataframe: pd.DataFrame,
        missing_column: str,
    ) -> None:
        """Raise a KeyError when a required column is missing."""
        dataframe = raw_dataframe.drop(columns=missing_column)

        with pytest.raises(
            KeyError,
            match=missing_column,
        ):
            preprocess_dataset(dataframe)

    def test_lists_all_missing_columns_in_error(self) -> None:
        """Report every required column absent from the DataFrame."""
        dataframe = pd.DataFrame(
            {
                "id": ["abs-2401.12345"],
                "title": ["Paper title"],
            }
        )

        with pytest.raises(KeyError) as exc_info:
            preprocess_dataset(dataframe)

        error_message = str(exc_info.value)

        for column in REQUIRED_COLUMNS.difference(dataframe.columns):
            assert column in error_message

    def test_does_not_modify_input_dataframe(
        self,
        raw_dataframe: pd.DataFrame,
    ) -> None:
        """Leave the original DataFrame unchanged."""
        original = raw_dataframe.copy(deep=True)

        preprocess_dataset(raw_dataframe)

        assert_frame_equal(raw_dataframe, original)

    def test_removes_rows_with_missing_title(self) -> None:
        """Remove papers that have no title."""
        dataframe = pd.DataFrame(
            {
                "id": [
                    "abs-2401.12345",
                    "abs-2401.12346",
                ],
                "title": [
                    "Valid paper",
                    None,
                ],
                "summary": [
                    "Valid summary",
                    "Summary without a title",
                ],
                "category": [
                    "cs.LG",
                    "cs.LG",
                ],
                "authors": [
                    "['Author One']",
                    "['Author Two']",
                ],
                "published_date": [
                    "2025-01-10",
                    "2025-01-11",
                ],
            }
        )

        result = preprocess_dataset(dataframe)

        assert len(result) == 1
        assert result.iloc[0]["id"] == "2401.12345"

    def test_removes_rows_with_missing_summary(self) -> None:
        """Remove papers that have no abstract."""
        dataframe = pd.DataFrame(
            {
                "id": [
                    "abs-2401.12345",
                    "abs-2401.12346",
                ],
                "title": [
                    "Valid paper",
                    "Paper without a summary",
                ],
                "summary": [
                    "Valid summary",
                    None,
                ],
                "category": [
                    "cs.LG",
                    "cs.LG",
                ],
                "authors": [
                    "['Author One']",
                    "['Author Two']",
                ],
                "published_date": [
                    "2025-01-10",
                    "2025-01-11",
                ],
            }
        )

        result = preprocess_dataset(dataframe)

        assert len(result) == 1
        assert result.iloc[0]["id"] == "2401.12345"

    def test_removes_duplicate_title_and_summary_pairs(self) -> None:
        """Keep only the first paper with an identical title and summary."""
        dataframe = pd.DataFrame(
            {
                "id": [
                    "abs-2401.12345",
                    "abs-2401.12346",
                    "abs-2401.12347",
                ],
                "title": [
                    "Duplicate paper",
                    "Duplicate paper",
                    "Different paper",
                ],
                "summary": [
                    "The same summary.",
                    "The same summary.",
                    "A different summary.",
                ],
                "category": [
                    "cs.LG",
                    "cs.LG",
                    "cs.AI",
                ],
                "authors": [
                    "['Author One']",
                    "['Author Two']",
                    "['Author Three']",
                ],
                "published_date": [
                    "2025-01-10",
                    "2025-01-11",
                    "2025-01-12",
                ],
            }
        )

        result = preprocess_dataset(dataframe)

        assert len(result) == 2
        assert result["id"].tolist() == [
            "2401.12345",
            "2401.12347",
        ]

    def test_resets_index_after_removing_rows(self) -> None:
        """Reset the DataFrame index after filtering duplicates."""
        dataframe = pd.DataFrame(
            {
                "id": [
                    "abs-2401.12345",
                    "abs-2401.12346",
                    "abs-2401.12347",
                ],
                "title": [
                    "Duplicate paper",
                    "Duplicate paper",
                    "Unique paper",
                ],
                "summary": [
                    "Duplicate summary",
                    "Duplicate summary",
                    "Unique summary",
                ],
                "category": [
                    "cs.LG",
                    "cs.LG",
                    "cs.AI",
                ],
                "authors": [
                    "['Author One']",
                    "['Author Two']",
                    "['Author Three']",
                ],
                "published_date": [
                    "2025-01-10",
                    "2025-01-11",
                    "2025-01-12",
                ],
            },
            index=[5, 8, 12],
        )

        result = preprocess_dataset(dataframe)

        assert result.index.tolist() == [0, 1]

    def test_converts_published_dates_to_datetime(
        self,
        raw_dataframe: pd.DataFrame,
    ) -> None:
        """Convert publication dates into pandas datetime values."""
        result = preprocess_dataset(raw_dataframe)

        assert pd.api.types.is_datetime64_any_dtype(
            result["published_date"]
        )
        assert result.loc[0, "published_date"] == pd.Timestamp(
            "2025-01-10"
        )
        assert result.loc[1, "published_date"] == pd.Timestamp(
            "2024-02-15 12:30:00"
        )

    def test_accepts_existing_datetime_values(self) -> None:
        """Preserve valid datetime values during preprocessing."""
        publication_date = datetime(2025, 1, 10, 12, 30)

        dataframe = pd.DataFrame(
            {
                "id": ["abs-2401.12345"],
                "title": ["Paper title"],
                "summary": ["Paper summary"],
                "category": ["cs.LG"],
                "authors": ["['Author One']"],
                "published_date": [publication_date],
            }
        )

        result = preprocess_dataset(dataframe)

        assert result.loc[0, "published_date"] == pd.Timestamp(
            publication_date
        )

    def test_rejects_invalid_published_date(self) -> None:
        """Raise a ValueError when a publication date cannot be parsed."""
        dataframe = pd.DataFrame(
            {
                "id": ["abs-2401.12345"],
                "title": ["Paper title"],
                "summary": ["Paper summary"],
                "category": ["cs.LG"],
                "authors": ["['Author One']"],
                "published_date": ["not-a-date"],
            }
        )

        with pytest.raises(ValueError):
            preprocess_dataset(dataframe)

    def test_cleans_text_columns(
        self,
        raw_dataframe: pd.DataFrame,
    ) -> None:
        """Normalize whitespace in title, category, and summary fields."""
        result = preprocess_dataset(raw_dataframe)

        assert result.loc[0, "title"] == "Finite volume methods"
        assert result.loc[0, "summary"] == "A finite-volume study."
        assert result.loc[0, "category"] == (
            "Computational Engineering"
        )
        assert result.loc[1, "category"] == "cs.LG"

    def test_parses_author_values(
        self,
        raw_dataframe: pd.DataFrame,
    ) -> None:
        """Convert author lists and tuples into readable strings."""
        result = preprocess_dataset(raw_dataframe)

        assert result["authors"].tolist() == [
            "Author One, Author Two",
            "Author Three, Author Four",
        ]

    def test_normalises_paper_identifiers(
        self,
        raw_dataframe: pd.DataFrame,
    ) -> None:
        """Convert modern and legacy identifiers to valid arXiv IDs."""
        result = preprocess_dataset(raw_dataframe)

        assert result["id"].tolist() == [
            "2401.12345",
            "cs/9308101v1",
        ]

    def test_rejects_invalid_arxiv_identifier(self) -> None:
        """Raise a ValueError when a paper ID has an unsupported format."""
        dataframe = pd.DataFrame(
            {
                "id": ["invalid-paper-id"],
                "title": ["Paper title"],
                "summary": ["Paper summary"],
                "category": ["cs.LG"],
                "authors": ["['Author One']"],
                "published_date": ["2025-01-10"],
            }
        )

        with pytest.raises(
            ValueError,
            match="Unsupported arXiv paper ID format",
        ):
            preprocess_dataset(dataframe)

    def test_handles_missing_category_and_authors(self) -> None:
        """Convert missing optional text metadata into empty strings."""
        dataframe = pd.DataFrame(
            {
                "id": ["abs-2401.12345"],
                "title": ["Paper title"],
                "summary": ["Paper summary"],
                "category": [None],
                "authors": [None],
                "published_date": ["2025-01-10"],
            }
        )

        result = preprocess_dataset(dataframe)

        assert result.loc[0, "category"] == ""
        assert result.loc[0, "authors"] == ""

    def test_returns_cleaned_dataframe(self) -> None:
        """Apply all preprocessing operations to the dataset."""
        dataframe = pd.DataFrame(
            {
                "id": ["  abs-2401.12345v2  "],
                "title": ["  Finite   volume methods  "],
                "summary": ["  Paper\nabstract  "],
                "category": ["  Computational   Mechanics "],
                "authors": ["['Author One', 'Author Two']"],
                "published_date": ["2025-01-10"],
                "unused": ["remove me"],
            }
        )

        result = preprocess_dataset(dataframe)

        assert len(result) == 1
        assert set(result.columns) == REQUIRED_COLUMNS
        assert result.iloc[0].to_dict() == {
            "id": "2401.12345v2",
            "title": "Finite volume methods",
            "summary": "Paper abstract",
            "category": "Computational Mechanics",
            "authors": "Author One, Author Two",
            "published_date": pd.Timestamp("2025-01-10"),
        }
