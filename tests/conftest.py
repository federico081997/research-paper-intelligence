"""Shared Pytest fixtures for the application."""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def valid_csv(tmp_path: Path) -> Path:
    """Create a valid csv file path."""
    csv_path = tmp_path / "dataset.csv"

    pd.DataFrame(
        {
            "id": ["c123"],
            "title": ["Paper title"],
            "summary": ["Paper abstract"],
            "category": ["Category"],
            "authors": ["Author"],
            "published_date": ["2025-01-10"],
        }
    ).to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for CSV-saving tests."""
    return pd.DataFrame(
        {
            "id": ["c123", "b987"],
            "title": ["Paper A", "Paper B"],
            "summary": ["Summary A", "Summary B"],
            "category": ["Category A", "Category B"],
            "authors": ["Author A", "Author B"],
            "published_date": ["2025-01-10", "2025-02-15"],
        }
    )


@pytest.fixture
def sample_processed_dataframe() -> pd.DataFrame:
    """Create a processed sample dataframe for embedding pipeline tests."""
    return pd.DataFrame(
        {
            "id": ["c123", "b987"],
            "title": ["Paper A", "Paper B"],
            "summary": ["Summary A", "Summary B"],
            "category": ["Category A", "Category B"],
            "authors": ["Author A", "Author B"],
            "published_date": ["2025-01-10", "2025-02-15"],
            "combined_text": ["Paper A Summary A", "Paper B Summary B"],
        }
    )
