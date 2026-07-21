"""Shared Pytest fixtures for the application."""

from datetime import date
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pandas as pd
import pytest

from research_paper_intelligence.config import Settings
from research_paper_intelligence.domain.search_result import (
    SearchResult,
)
from research_paper_intelligence.services.search_service import (
    SearchService,
)


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
def simple_settings() -> Settings:
    """Create a simple settings configuration."""
    settings = Settings(
        hf_repository="user/research-papers",
        hf_tfidf_vectorizer_file="tfidf/vectorizer.joblib",
        hf_tfidf_matrix_file="tfidf/matrix.npz",
        hf_processed_papers_file="data/processed_papers.csv",
        tfidf_vectorizer_path=Path("artifacts/vectorizer.joblib"),
        tfidf_matrix_path=Path("artifacts/matrix.npz"),
        processed_papers_path=Path("data/processed_papers.csv"),
        semantic_weight=0.65,
        tfidf_weight=0.20,
        keyword_weight=0.10,
        recency_weight=0.05,
        half_life_years=5.0,
        candidate_top_k=100,
    )
    return settings


@pytest.fixture
def search_result() -> SearchResult:
    """Create a representative domain search result."""
    paper = Mock()
    paper.paper_id = "paper-001"
    paper.title = "Finite volume methods"
    paper.abstract = "An abstract about finite volume methods."
    paper.authors = "Author One, Author Two"
    paper.category = "Computational Engineering"
    paper.published_date = date(2025, 1, 15)

    result = Mock()
    result.paper = paper
    result.rank = 1
    result.semantic_score = 0.91
    result.tfidf_score = 0.72
    result.keyword_overlap_score = 0.65
    result.recency_score = 0.80
    result.hybrid_score = 0.84
    result.explanation = "Strong semantic similarity."

    return cast(SearchResult, result)


@pytest.fixture
def search_service() -> Mock:
    """Create a mocked search service."""
    return Mock(spec=SearchService)
