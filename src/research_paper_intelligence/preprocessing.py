"""
Preprocessing utilities for the dataset
"""

from ast import literal_eval
from typing import Any, cast

import pandas as pd
from pandas.api.types import is_scalar

from research_paper_intelligence.config import get_settings


settings = get_settings()

def clean_text(text: object) -> str:
    """
    Function to clean the text, which removes extra and trailing whitespaces.
    Args:
        text: Input text value.

    Returns:
        A cleaned string. If the input is missing, it returns an empty string.
    """

    # Check if the input text is null or scalar
    if text is None or (
            is_scalar(text) and bool(pd.isna(cast(Any, text)))
    ):
        return ""

    return " ".join(str(text).split())


def parse_authors(authors: object) -> str:
    """
    Converts the authors field into a human-readable string.

    Args:
        authors: Input author field

    Returns:
        A readable string representation of the Authors field.
    """

    # Handle missing DataFrame None values.
    if authors is None:
        return ""

    # Handle lists and tuples before calling isna()
    if isinstance(authors, (list, tuple)):
        return ", ".join(
            str(author).strip()
            for author in authors
            if str(author).strip()
        )

    # Handle NaNs in the Dataframe
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

        # Check if the string to tuple conversion has been successful
        if isinstance(parsed_authors, (list, tuple)):
            return ", ".join(
                str(author).strip()
                for author in parsed_authors
            )

        # Handle quoted Python strings
        if isinstance(parsed_authors, str):
            return parsed_authors.strip()

        # Preserve the original input
        return stripped_authors

    # Return the original input converted into a string if any of the above
    # checks failed
    return str(authors).strip()

