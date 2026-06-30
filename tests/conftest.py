"""Shared Pytest fixtures for the application."""

from pathlib import Path
from types import SimpleNamespace

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
def simple_settings() -> SimpleNamespace:
    """Create a simple settings configuration."""
    settings = SimpleNamespace(
        hf_repository="user/research-papers",
        hf_tfidf_vectorizer_file="tfidf/vectorizer.joblib",
        hf_tfidf_matrix_file="tfidf/matrix.npz",
        hf_processed_papers_file="data/processed_papers.csv",
        tfidf_vectorizer_path=Path("artifacts/vectorizer.joblib"),
        tfidf_matrix_path=Path("artifacts/matrix.npz"),
        processed_papers_path=Path("data/processed_papers.csv"),
    )
    return settings
