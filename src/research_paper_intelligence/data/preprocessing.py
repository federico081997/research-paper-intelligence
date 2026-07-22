"""Preprocessing utilities for the dataset."""

from ast import literal_eval
from typing import Any, cast
import re

import pandas as pd
from pandas.api.types import is_scalar

from research_paper_intelligence.config import get_settings

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "id",
        "title",
        "summary",
        "category",
        "authors",
        "published_date",
    }
)

MODERN_ARXIV_ID_PATTERN = re.compile(
    r"^\d{4}.\d{4,5}(?:v\d+)?$"
)

LEGACY_ARXIV_ID_PATTERN = re.compile(
    r"^(?P<archive>.+)-(?P<number>\d{7}(?:v\d+)?)$"
)

def clean_text(text: object) -> str:
    """Cleans the text, which removes extra and trailing whitespaces.

    Args:
        text: Input text value.

    Returns:
        A cleaned string. If the input is missing, it returns an empty string.
    """
    # Check if the input text is null or scalar.
    if text is None or (is_scalar(text) and bool(pd.isna(cast(Any, text)))):
        return ""

    return " ".join(str(text).split())


def parse_authors(authors: object) -> str:
    """Converts the authors field into a human-readable string.

    Args:
        authors: Input author field

    Returns:
        A readable string representation of the Authors field.
    """
    # Handle missing DataFrame None values.
    if authors is None:
        return ""

    # Handle lists and tuples before calling isna().
    if isinstance(authors, (list, tuple)):
        return ", ".join(
            str(author).strip() for author in authors if str(author).strip()
        )

    # Handle Nans in the Dataframe.
    if bool(pd.isna(cast(Any, authors))):
        return ""

    if isinstance(authors, str):
        stripped_authors = authors.strip()

        if not stripped_authors:
            return ""

        try:
            parsed_authors: object = literal_eval(stripped_authors)
        except (ValueError, SyntaxError):
            return stripped_authors

        # Check if the string-to-tuple conversion has been successful.
        if isinstance(parsed_authors, (list, tuple)):
            return ", ".join(str(author).strip() for author in parsed_authors)

        # Handle quoted Python strings.
        if isinstance(parsed_authors, str):
            return parsed_authors.strip()

        # Preserve the original input.
        return stripped_authors

    # Return the original input converted into a string if any of the above
    # checks failed.
    return str(authors).strip()

def normalize_arxiv_id(paper_id: str) -> str:
    """Convert a dataset paper ID into a valid arXiv identifier.

    The dataset contains two formats:

    - Modern IDs: ``abs-0901.4761v1``
    - Legacy IDs: ``cs-9308101v1``

    These are normalized to:

    - ``0901.4761v1``
    - ``cs/9308101v1``

    Args:
        paper_id: The paper identifier stored in the dataset.

    Returns:
        A valid arXiv identifier.

    Raises:
        ValueError: If the identifier is empty or has an unsupported format.
    """
    normalized_id = paper_id.strip()

    if not normalized_id:
        raise ValueError("paper_id must not be empty")

    # Modern dataset format:
    # abs-0901.4761v1 -> 0901.4761v1
    normalized_id = normalized_id.removeprefix("abs-")

    if MODERN_ARXIV_ID_PATTERN.fullmatch(normalized_id):
        return normalized_id

    # Legacy dataset format:
    # cs-9308101v1 -> cs/9308101v1
    match = LEGACY_ARXIV_ID_PATTERN.fullmatch(normalized_id)

    if match is not None:
        archive = match.group("archive")
        number = match.group("number")
        return f"{archive}/{number}"

    raise ValueError(f"Unsupported arXiv paper ID format: {paper_id!r}")


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the dataset with relevant columns.

    Args:
        df: A pandas DataFrame containing the dataset.

    Returns:
        A cleaned pandas DataFrame.

    Raises:
        KeyError: If the required columns are not present in the dataset.
        ValueError: If a published date cannot be parsed of if the arXiv
            format is invalid.
    """
    df = df.copy()

    # Check if there are any missing columns in the dataset.
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise KeyError(
            "The following columns are not present in the processed data: "
            + ", ".join(missing_columns)
        )

    # Keep only relevant columns.
    df = df[list(REQUIRED_COLUMNS)]

    # Remove rows missing title or abstract, which are not useful for search.
    df = df.dropna(subset=["title", "summary"])

    # Remove duplicate records based on 'title' and 'abstract'.
    df = df.drop_duplicates(subset=["title", "summary"]).reset_index(drop=True)

    # Ensure the published date column is in datetime format.
    df["published_date"] = pd.to_datetime(
        df["published_date"], format="mixed", errors="raise"
    )

    # Clean text fields.
    df[["title", "category", "summary"]] = df[
        ["title", "category", "summary"]
    ].map(clean_text)

    # Convert authors into a readable string format.
    df["authors"] = df["authors"].apply(parse_authors)

    # Normalize arXiv identifiers
    df["id"] = df["id"].map(normalize_arxiv_id)

    return df
