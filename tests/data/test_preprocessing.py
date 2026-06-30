"""Tests performed on the "preprocessing" module."""

from pathlib import Path

import pandas as pd
import pytest

from research_paper_intelligence.data.preprocessing import (
    REQUIRED_COLUMNS,
    clean_text,
    parse_authors,
    preprocess_dataset,
)

# -----------------------------------------------------------------------------
#   TestCleanText
# -----------------------------------------------------------------------------


class TestCleanText:
    """Tests performed on the "clean_text" function."""

    @pytest.mark.parametrize(
        ("input_text", "expected"),
        [
            pytest.param("Clean text", "Clean text"),
            pytest.param("  leading and trailing  ", "leading and trailing"),
            pytest.param(
                "multiple    internal    spaces", "multiple internal spaces"
            ),
            pytest.param("text\twith\ttabs", "text with tabs"),
            pytest.param("text\nwith\nnewlines", "text with newlines"),
            pytest.param(" \t\n ", ""),
            pytest.param("", ""),
        ],
        ids=[
            "already-clean",
            "leading-and-trailing-spaces",
            "multiple-internal-spaces",
            "tabs",
            "newlines",
            "whitespace-only",
            "empty-string",
        ],
    )
    def test_normalizes_whitespace(
        self,
        input_text: str,
        expected: str,
    ) -> None:
        """Normalize leading, trailing, and repeated whitespace."""
        result = clean_text(input_text)

        assert result == expected

    @pytest.mark.parametrize(
        ("input_value", "expected"),
        [
            pytest.param(None, ""),
            pytest.param(float("nan"), ""),
            pytest.param(pd.NA, ""),
            pytest.param(pd.NaT, ""),
        ],
        ids=[
            "none",
            "float-nan",
            "pandas-na",
            "pandas-nat",
        ],
    )
    def test_null_values(
        self,
        input_value: int,
        expected: str,
    ) -> None:
        """Convert null values into empty strings."""
        result = clean_text(input_value)

        assert result == expected

    @pytest.mark.parametrize(
        ("input_value", "expected"),
        [
            pytest.param(123, "123"),
            pytest.param(3.2, "3.2"),
            pytest.param(True, "True"),
        ],
        ids=[
            "integer",
            "float",
            "boolean",
        ],
    )
    def test_converts_non_string_values_to_strings(
        self,
        input_value: int,
        expected: str,
    ) -> None:
        """Convert non-string values to strings."""
        result = clean_text(input_value)

        assert result == expected


# -----------------------------------------------------------------------------
#   TestParseAuthors
# -----------------------------------------------------------------------------


class TestParseAuthors:
    """Tests for the parse_authors preprocessing function."""

    @pytest.mark.parametrize(
        "authors",
        [
            pytest.param(None),
            pytest.param(float("nan")),
            pytest.param(pd.NA),
            pytest.param(pd.NaT),
        ],
        ids=[
            "none",
            "float-nan",
            "pandas-na",
            "pandas-nat",
        ],
    )
    def test_missing_values_return_empty_string(
        self,
        authors: object,
    ) -> None:
        """Return an empty string if no authors were given."""
        assert parse_authors(authors) == ""

    @pytest.mark.parametrize(
        ("authors", "expected"),
        [
            pytest.param(["Alice", "Bob"], "Alice, Bob"),
            pytest.param(("Alice", "Bob"), "Alice, Bob"),
            pytest.param([" Alice ", " Bob "], "Alice, Bob"),
            pytest.param((" Alice ", " Bob "), "Alice, Bob"),
            pytest.param(["Alice", "", "   ", "Bob"], "Alice, Bob"),
            pytest.param([], ""),
            pytest.param((), ""),
            pytest.param([1, 2, 3], "1, 2, 3"),
        ],
        ids=[
            "list",
            "tuple",
            "list-with-whitespace",
            "tuple-with-whitespace",
            "list-with-empty-authors",
            "empty-list",
            "empty-tuple",
            "list-of-integers",
        ],
    )
    def test_actual_lists_and_tuples_are_joined(
        self,
        authors: object,
        expected: str,
    ) -> None:
        """Join items of lists or tuples using a comma."""
        assert parse_authors(authors) == expected

    @pytest.mark.parametrize(
        "authors",
        [
            pytest.param(""),
            pytest.param(" "),
            pytest.param("   "),
            pytest.param("\t"),
            pytest.param("\n"),
        ],
        ids=[
            "empty-string",
            "single-space",
            "multiple-spaces",
            "tab",
            "newline",
        ],
    )
    def test_empty_or_whitespace_strings_return_empty_string(
        self,
        authors: str,
    ) -> None:
        """Return an empty string for blank or whitespace-only values."""
        assert parse_authors(authors) == ""

    @pytest.mark.parametrize(
        ("authors", "expected"),
        [
            pytest.param("['Alice', 'Bob']", "Alice, Bob"),
            pytest.param('["Alice", "Bob"]', "Alice, Bob"),
            pytest.param("('Alice', 'Bob')", "Alice, Bob"),
            pytest.param("[' Alice ', ' Bob ']", "Alice, Bob"),
            pytest.param("[1, 2, 3]", "1, 2, 3"),
            pytest.param("[]", ""),
            pytest.param("()", ""),
        ],
        ids=[
            "string-list-single-quotes",
            "string-list-double-quotes",
            "string-tuple",
            "string-list-with-whitespace",
            "string-list-of-integers",
            "string-empty-list",
            "string-empty-tuple",
        ],
    )
    def test_string_representations_of_collections_are_parsed(
        self,
        authors: str,
        expected: str,
    ) -> None:
        """Successfully parse string representations of collections."""
        assert parse_authors(authors) == expected

    @pytest.mark.parametrize(
        ("authors", "expected"),
        [
            pytest.param("'Alice Smith'", "Alice Smith"),
            pytest.param('"Alice Smith"', "Alice Smith"),
            pytest.param("' Alice Smith '", "Alice Smith"),
            pytest.param('" Alice Smith "', "Alice Smith"),
        ],
        ids=[
            "single-quoted-string",
            "double-quoted-string",
            "single-quoted-string-with-whitespace",
            "double-quoted-string-with-whitespace",
        ],
    )
    def test_quoted_python_strings_are_unwrapped(
        self,
        authors: str,
        expected: str,
    ) -> None:
        """Unwrap quoted python strings."""
        assert parse_authors(authors) == expected

    @pytest.mark.parametrize(
        ("authors", "expected"),
        [
            pytest.param("Alice Smith", "Alice Smith"),
            pytest.param(" Alice Smith ", "Alice Smith"),
            pytest.param("Alice, Bob", "Alice, Bob"),
            pytest.param("nan", "nan"),
        ],
        ids=[
            "plain-name",
            "plain-name-with-whitespace",
            "comma-separated-names",
            "string-nan",
        ],
    )
    def test_non_literal_strings_are_preserved(
        self,
        authors: str,
        expected: str,
    ) -> None:
        """Preserve non-literal strings."""
        assert parse_authors(authors) == expected

    def test_invalid_python_syntax_is_preserved(self) -> None:
        """Preserve invalid python syntax."""
        authors = "['Alice', 'Bob'"

        assert parse_authors(authors) == authors

    @pytest.mark.parametrize(
        ("authors", "expected"),
        [
            pytest.param("123", "123"),
            pytest.param("True", "True"),
            pytest.param("None", "None"),
            pytest.param("{'name': 'Alice'}", "{'name': 'Alice'}"),
        ],
        ids=[
            "integer-literal",
            "boolean-literal",
            "none-literal",
            "dictionary-literal",
        ],
    )
    def test_other_valid_literals_preserve_original_string(
        self,
        authors: str,
        expected: str,
    ) -> None:
        """Preserve original string literals."""
        assert parse_authors(authors) == expected

    @pytest.mark.parametrize(
        ("authors", "expected"),
        [
            pytest.param(123, "123"),
            pytest.param(3.14, "3.14"),
            pytest.param(True, "True"),
            pytest.param(False, "False"),
        ],
        ids=[
            "integer",
            "float",
            "true",
            "false",
        ],
    )
    def test_other_values_are_converted_to_strings(
        self,
        authors: object,
        expected: str,
    ) -> None:
        """Convert other values to strings."""
        assert parse_authors(authors) == expected


# -----------------------------------------------------------------------------
#   TestPreprocessDataset
# -----------------------------------------------------------------------------


class TestPreprocessDataset:
    """Tests for the preprocess_dataset function."""

    def test_preprocesses_valid_dataset(
        self,
        valid_csv: Path,
    ) -> None:
        """Preprocess a valid research-paper dataset."""
        dataframe = pd.read_csv(valid_csv)

        result = preprocess_dataset(dataframe)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.loc[0, "title"] == "Paper title"
        assert result.loc[0, "summary"] == "Paper abstract"
        assert result.loc[0, "category"] == "Category"
        assert result.loc[0, "authors"] == "Author"

    def test_does_not_modify_original_dataframe(
        self,
        valid_csv: Path,
    ) -> None:
        """Leave the input DataFrame unchanged."""
        dataframe = pd.read_csv(valid_csv)
        original = dataframe.copy(deep=True)

        preprocess_dataset(dataframe)

        pd.testing.assert_frame_equal(dataframe, original)

    @pytest.mark.parametrize(
        "missing_column",
        sorted(REQUIRED_COLUMNS),
    )
    def test_raises_key_error_for_missing_required_column(
        self,
        valid_csv: Path,
        missing_column: str,
    ) -> None:
        """Reject a dataset missing any required column."""
        dataframe = pd.read_csv(valid_csv)
        dataframe = dataframe.drop(columns=[missing_column])

        missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
        with pytest.raises(
            KeyError,
            match="The following columns are not present in the processed "
            "data: " + ", ".join(missing_columns),
        ):
            preprocess_dataset(dataframe)

    @pytest.mark.parametrize(
        "missing_column",
        ["title", "summary"],
    )
    def test_removes_rows_missing_search_text(
        self,
        valid_csv: Path,
        missing_column: str,
    ) -> None:
        """Remove rows missing a title or summary."""
        dataframe = pd.read_csv(valid_csv)

        missing_row = dataframe.copy()
        missing_row.loc[0, missing_column] = None

        dataframe = pd.concat(
            [dataframe, missing_row],
            ignore_index=True,
        )

        result = preprocess_dataset(dataframe)

        assert len(result) == 1
        assert result.loc[0, "title"] == "Paper title"

    def test_removes_duplicate_title_and_summary_pairs(
        self,
        valid_csv: Path,
    ) -> None:
        """Remove duplicate papers based on title and summary."""
        dataframe = pd.read_csv(valid_csv)

        dataframe = pd.concat(
            [dataframe, dataframe],
            ignore_index=True,
        )

        result = preprocess_dataset(dataframe)

        assert len(result) == 1
        assert list(result.index) == [0]

    def test_keeps_papers_with_same_title_but_different_summary(
        self,
        valid_csv: Path,
    ) -> None:
        """Keep rows when only the title is duplicated."""
        dataframe = pd.read_csv(valid_csv)

        second_paper = dataframe.copy()
        second_paper.loc[0, "summary"] = "A different abstract"

        dataframe = pd.concat(
            [dataframe, second_paper],
            ignore_index=True,
        )

        result = preprocess_dataset(dataframe)

        assert len(result) == 2

    def test_converts_published_date_to_datetime(
        self,
        valid_csv: Path,
    ) -> None:
        """Convert publication dates to pandas datetime values."""
        dataframe = pd.read_csv(valid_csv)

        result = preprocess_dataset(dataframe)

        assert pd.api.types.is_datetime64_any_dtype(result["published_date"])
        assert result.loc[0, "published_date"] == pd.Timestamp("2025-01-10")

    def test_raises_value_error_for_invalid_published_date(
        self,
        valid_csv: Path,
    ) -> None:
        """Reject invalid publication-date values."""
        dataframe = pd.read_csv(valid_csv)
        dataframe.loc[0, "published_date"] = "not-a-date"

        with pytest.raises(ValueError):
            preprocess_dataset(dataframe)

    def test_cleans_text_columns(
        self,
        valid_csv: Path,
    ) -> None:
        """Normalize whitespace in the configured text columns."""
        dataframe = pd.read_csv(valid_csv)

        dataframe.loc[0, "title"] = "  Paper    title  "
        dataframe.loc[0, "summary"] = " Paper\n abstract "
        dataframe.loc[0, "category"] = "  Machine   Learning "

        result = preprocess_dataset(dataframe)

        assert result.loc[0, "title"] == "Paper title"
        assert result.loc[0, "summary"] == "Paper abstract"
        assert result.loc[0, "category"] == "Machine Learning"

    def test_parses_authors(
        self,
        valid_csv: Path,
    ) -> None:
        """Convert a serialized author list into readable text."""
        dataframe = pd.read_csv(valid_csv)
        dataframe.loc[0, "authors"] = "['Alice Smith', 'Bob Jones']"

        result = preprocess_dataset(dataframe)

        assert result.loc[0, "authors"] == "Alice Smith, Bob Jones"

    def test_removes_irrelevant_columns(
        self,
        valid_csv: Path,
    ) -> None:
        """Exclude columns that are not part of preprocessing."""
        dataframe = pd.read_csv(valid_csv)
        dataframe["irrelevant_column"] = ["unused"]

        result = preprocess_dataset(dataframe)

        assert "irrelevant_column" not in result.columns
        assert set(result.columns) == set(REQUIRED_COLUMNS)
