"""
Tests performed on the ``preprocessing`` module.
"""

import pandas as pd
import pytest

from research_paper_intelligence.preprocessing import (
    clean_text,
    parse_authors
)

# -----------------------------------------------------------------------------
#   TestCleanText
# -----------------------------------------------------------------------------

class TestCleanText:
    """
    Tests performed on the ``clean_text`` function.
    """

    @pytest.mark.parametrize(
        ("input_text", "expected"),
        [
            pytest.param("Clean text", "Clean text"),
            pytest.param("  leading and trailing  ", "leading and trailing"),
            pytest.param("multiple    internal    spaces", "multiple internal spaces"),
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

        ]
    )
    def test_normalizes_whitespace(
        self,
        input_text: str,
        expected: str,
    ) -> None:
        """
        Normalize leading, trailing and repeated whitespace.
        """
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
        ]
    )
    def test_null_values(
        self,
        input_value: int,
        expected: str,
    ) -> None:
        """
        Convert null values into empty strings.
        """
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
        ]
    )
    def test_converts_non_string_values_to_strings(
        self,
        input_value: int,
        expected: str,
    ) -> None:
        """
        Convert non-string values to strings.
        """
        result = clean_text(input_value)

        assert result == expected

# -----------------------------------------------------------------------------
#   TestParseAuthors
# -----------------------------------------------------------------------------

class TestParseAuthors:
    """
    Tests for the parse_authors preprocessing function.
    """

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
        ]
    )
    def test_missing_values_return_empty_string(
        self,
        authors: object,
    ) -> None:
        """
        Return an empty string if no authors were given.
        """
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
        """
        Join items of lists or tuples using a comma.
        """
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
        """
        Return an empty string when passed in an empty string or escape
        characters.
        """
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
        """
        Parse string representations of collections and successfully return
        a formatted string.
        """
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
        """
        Unwrap quoted python strings.
        """
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
        """
        Preserve non-literal strings.
        """
        assert parse_authors(authors) == expected

    def test_invalid_python_syntax_is_preserved(self) -> None:
        """
        Preserve invalid python syntax.
        """
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
        """
        Preserve original string literals.
        """
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
        """
        Convert other values to strings.
        """
        assert parse_authors(authors) == expected
