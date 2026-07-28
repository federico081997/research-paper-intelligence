"""Shared Pytest fixtures for the application."""

from datetime import date
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from research_paper_intelligence.api.schemas.search import SearchResultItem
from research_paper_intelligence.assistant.models import RetrievedPaper
from research_paper_intelligence.config import Settings
from research_paper_intelligence.domain.search_result import (
    SearchResult,
)
from research_paper_intelligence.services.assistant_service import (
    ResearchAssistant,
)
from research_paper_intelligence.services.search_service import (
    SearchService,
)


@pytest.fixture
def valid_csv(tmp_path: Path) -> Path:
    """Create a valid research-paper CSV file."""
    dataframe = pd.DataFrame(
        {
            "id": ["abs-2401.12345"],
            "title": ["Paper title"],
            "summary": ["Paper abstract"],
            "category": ["Category"],
            "authors": ["['Author']"],
            "published_date": ["2024-01-15"],
        }
    )

    csv_path = tmp_path / "valid.csv"
    dataframe.to_csv(csv_path, index=False)

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


@pytest.fixture
def assistant_service() -> Mock:
    """Create a mocked research assistant service."""
    return Mock(spec=ResearchAssistant)


@pytest.fixture
def retrieved_papers() -> list[RetrievedPaper]:
    """Create representative papers for retrieval-grading tests."""
    return [
        RetrievedPaper(
            paper_id="2401.12345",
            title="Finite volume methods for solid mechanics",
            abstract=(
                "A block-coupled finite volume method is developed "
                "for computational solid mechanics."
            ),
            authors="Author One, Author Two",
            category="Computational Engineering",
            published_date=date(2025, 1, 15),
            rank=1,
            hybrid_score=0.91,
        ),
        RetrievedPaper(
            paper_id="2402.67890",
            title="Elastoplasticity using finite volume discretisation",
            abstract=(
                "This paper studies elastoplastic constitutive models "
                "within a finite volume framework."
            ),
            authors="Author Three",
            category="Computational Mechanics",
            published_date=date(2024, 6, 10),
            rank=2,
            hybrid_score=0.84,
        ),
    ]


@pytest.fixture
def http_response(
    request: pytest.FixtureRequest,
) -> Mock:
    """Create a mocked HTTP response using the module-specific payload."""
    response_payload = cast(
        dict[str, object],
        request.getfixturevalue("response_payload"),
    )

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = response_payload

    return response


@pytest.fixture
def http_client(http_response: Mock) -> Mock:
    """Create a mocked HTTP client for GET and POST requests."""
    client = Mock()
    client.get.return_value = http_response
    client.post.return_value = http_response

    return client


@pytest.fixture
def http_client_context(
    http_client: Mock,
) -> MagicMock:
    """Create a mocked HTTP client context manager."""
    context_manager = MagicMock()
    context_manager.__enter__.return_value = http_client
    context_manager.__exit__.return_value = False

    return context_manager


@pytest.fixture
def http_client_constructor(
    monkeypatch: pytest.MonkeyPatch,
    http_client_context: MagicMock,
    request: pytest.FixtureRequest,
) -> Mock:
    """Patch the HTTP client constructor in the module under test."""
    client_module = cast(
        ModuleType,
        request.getfixturevalue("client_module"),
    )
    httpx_module = client_module.httpx2

    constructor = Mock(return_value=http_client_context)

    monkeypatch.setattr(
        httpx_module,
        "Client",
        constructor,
    )

    return constructor


@pytest.fixture
def search_results() -> list[SearchResultItem]:
    """Create representative paper search results."""
    return [
        SearchResultItem(
            paper_id="2401.12345",
            title="Finite volume methods",
            abstract="An abstract about finite volume methods.",
            authors="Author One, Author Two",
            category="Computational Engineering",
            published_date=date(2025, 1, 15),
            rank=1,
            semantic_score=0.91,
            tfidf_score=0.72,
            keyword_overlap_score=0.65,
            recency_score=0.80,
            hybrid_score=0.84,
            explanation="Strong semantic similarity.",
        ),
        SearchResultItem(
            paper_id="2402.67890",
            title="Semantic scientific search",
            abstract="An abstract about semantic paper retrieval.",
            authors="Author Three",
            category="Information Retrieval",
            published_date=date(2024, 6, 10),
            rank=2,
            semantic_score=0.87,
            tfidf_score=0.68,
            keyword_overlap_score=0.55,
            recency_score=0.70,
            hybrid_score=0.79,
            explanation="Strong topical similarity.",
        ),
    ]
