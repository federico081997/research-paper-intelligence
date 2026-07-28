"""Tests for the paper search application service."""

from typing import cast
from unittest.mock import MagicMock, Mock

import faiss
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from research_paper_intelligence.config import Settings
from research_paper_intelligence.domain.search_result import SearchResult
from research_paper_intelligence.repositories.paper_repository import (
    PaperRepository,
)
from research_paper_intelligence.services import search_service
from research_paper_intelligence.services.search_service import SearchService


@pytest.fixture
def paper_repository() -> MagicMock:
    """Create a mocked paper repository."""
    repository = MagicMock(spec=PaperRepository)
    repository.__len__.return_value = 130_000

    return repository


@pytest.fixture
def embedding_model() -> Mock:
    """Create a mocked sentence-transformer model."""
    return Mock(spec=SentenceTransformer)


@pytest.fixture
def faiss_index() -> faiss.IndexFlatIP:
    """Create a small FAISS inner-product index."""
    index = faiss.IndexFlatIP(3)
    index.add(
        np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
            dtype="float32",
        )
    )

    return index


@pytest.fixture
def tfidf_vectorizer() -> Mock:
    """Create a mocked fitted TF-IDF vectorizer."""
    vectorizer = Mock(spec=TfidfVectorizer)
    vectorizer.vocabulary_ = {
        "finite": 0,
        "volume": 1,
        "method": 2,
        "mechanics": 3,
    }

    return vectorizer


@pytest.fixture
def tfidf_matrix() -> csr_matrix:
    """Create a representative sparse TF-IDF document matrix."""
    return csr_matrix((130_000, 4))


@pytest.fixture
def search_settings() -> Mock:
    """Create settings required by the search service."""
    settings = Mock(spec=Settings)
    settings.candidate_top_k = 100
    settings.semantic_weight = 0.65
    settings.tfidf_weight = 0.20
    settings.keyword_weight = 0.10
    settings.recency_weight = 0.05
    settings.half_life_years = 5.0
    settings.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    return settings


@pytest.fixture
def service(
    paper_repository: MagicMock,
    embedding_model: Mock,
    faiss_index: faiss.IndexFlatIP,
    tfidf_vectorizer: Mock,
    tfidf_matrix: csr_matrix,
    search_settings: Mock,
) -> SearchService:
    """Create a search service containing mocked dependencies."""
    return SearchService(
        paper_repository=cast(PaperRepository, paper_repository),
        model=cast(SentenceTransformer, embedding_model),
        index=faiss_index,
        vectorizer=cast(TfidfVectorizer, tfidf_vectorizer),
        tfidf_matrix=tfidf_matrix,
        settings=cast(Settings, search_settings),
    )


class TestSearchServiceInitialization:
    """Tests for SearchService initialisation."""

    def test_stores_search_dependencies(
        self,
        service: SearchService,
        paper_repository: MagicMock,
        embedding_model: Mock,
        faiss_index: faiss.IndexFlatIP,
        tfidf_vectorizer: Mock,
        tfidf_matrix: csr_matrix,
        search_settings: Mock,
    ) -> None:
        """Store all dependencies supplied to the service."""
        assert service.paper_repository is paper_repository
        assert service.model is embedding_model
        assert service.index is faiss_index
        assert service.vectorizer is tfidf_vectorizer
        assert service.tfidf_matrix is tfidf_matrix
        assert service.settings is search_settings


class TestSearch:
    """Tests for the SearchService.search method."""

    def test_calls_hybrid_search_with_service_dependencies(
        self,
        monkeypatch: pytest.MonkeyPatch,
        service: SearchService,
        paper_repository: MagicMock,
        embedding_model: Mock,
        faiss_index: faiss.IndexFlatIP,
        tfidf_vectorizer: Mock,
        tfidf_matrix: csr_matrix,
    ) -> None:
        """Pass dependencies and ranking settings to hybrid search."""
        expected_results = [
            cast(SearchResult, Mock(name="result_1")),
            cast(SearchResult, Mock(name="result_2")),
        ]
        hybrid_search_mock = Mock(return_value=expected_results)

        monkeypatch.setattr(
            search_service,
            "hybrid_search",
            hybrid_search_mock,
        )

        result = service.search(
            query="finite volume solid mechanics",
            result_k=7,
        )

        assert result is expected_results
        hybrid_search_mock.assert_called_once_with(
            query="finite volume solid mechanics",
            paper_repository=paper_repository,
            model=embedding_model,
            index=faiss_index,
            vectorizer=tfidf_vectorizer,
            tfidf_matrix=tfidf_matrix,
            candidate_k=100,
            result_k=7,
            semantic_weight=0.65,
            tfidf_weight=0.20,
            keyword_weight=0.10,
            recency_weight=0.05,
            half_life_years=5.0,
        )

    @pytest.mark.parametrize(
        ("query", "result_k"),
        [
            ("finite volume methods", 1),
            ("semantic scientific search", 5),
            ("neural operators", 10),
        ],
    )
    def test_forwards_query_and_result_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
        service: SearchService,
        query: str,
        result_k: int,
    ) -> None:
        """Forward different queries and result counts unchanged."""
        hybrid_search_mock = Mock(return_value=[])

        monkeypatch.setattr(
            search_service,
            "hybrid_search",
            hybrid_search_mock,
        )

        service.search(query=query, result_k=result_k)

        assert hybrid_search_mock.call_args.kwargs["query"] == query
        assert hybrid_search_mock.call_args.kwargs["result_k"] == result_k

    def test_returns_empty_list_when_hybrid_search_finds_no_results(
        self,
        monkeypatch: pytest.MonkeyPatch,
        service: SearchService,
    ) -> None:
        """Return an empty list when hybrid search finds no papers."""
        monkeypatch.setattr(
            search_service,
            "hybrid_search",
            Mock(return_value=[]),
        )

        result = service.search(
            query="unavailable research topic",
            result_k=5,
        )

        assert result == []

    def test_propagates_hybrid_search_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        service: SearchService,
    ) -> None:
        """Propagate errors raised by the hybrid-ranking function."""
        monkeypatch.setattr(
            search_service,
            "hybrid_search",
            Mock(side_effect=RuntimeError("Hybrid search failed.")),
        )

        with pytest.raises(
            RuntimeError,
            match="Hybrid search failed",
        ):
            service.search(
                query="finite volume methods",
                result_k=5,
            )


class TestSearchServiceMetadata:
    """Tests for search-service metadata properties."""

    def test_returns_repository_paper_count(
        self,
        service: SearchService,
        paper_repository: MagicMock,
    ) -> None:
        """Return the number of papers stored in the repository."""
        result = service.paper_count

        assert result == 130_000
        paper_repository.__len__.assert_called_once_with()

    def test_returns_configured_embedding_model(
        self,
        service: SearchService,
    ) -> None:
        """Return the configured sentence-transformer model name."""
        assert service.embedding_model == (
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def test_returns_hybrid_retrieval_strategy(
        self,
        service: SearchService,
    ) -> None:
        """Return the service retrieval-strategy description."""
        assert service.retrieval_strategy == "Hybrid"

    def test_returns_ranking_components(
        self,
        service: SearchService,
    ) -> None:
        """Return all components used by hybrid ranking."""
        assert service.ranking_components == [
            "Semantic similarity",
            "TF-IDF similarity",
            "Keyword overlap",
            "Publication recency",
        ]

    def test_returns_faiss_index_type(
        self,
        service: SearchService,
    ) -> None:
        """Return the concrete FAISS index class name."""
        assert service.faiss_index_type == "IndexFlatIP"

    def test_returns_faiss_index_size(
        self,
        service: SearchService,
    ) -> None:
        """Return the number of vectors stored in the FAISS index."""
        assert service.faiss_index_size == 3
        assert isinstance(service.faiss_index_size, int)

    def test_returns_tfidf_document_count(
        self,
        service: SearchService,
    ) -> None:
        """Return the number of documents represented by the TF-IDF matrix."""
        assert service.tfidf_document_count == 130_000
        assert isinstance(service.tfidf_document_count, int)

    def test_returns_tfidf_vocabulary_size(
        self,
        service: SearchService,
    ) -> None:
        """Return the fitted TF-IDF vocabulary size."""
        assert service.tfidf_vocabulary_size == 4
