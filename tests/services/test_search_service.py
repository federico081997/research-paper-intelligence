"""Tests the search service module."""

from unittest.mock import Mock

import pytest
from scipy.sparse import csr_matrix

from research_paper_intelligence.config import Settings
from research_paper_intelligence.services import search_service


class TestSearchService:
    """Tests the search service class."""

    def test_search_calls_hybrid_search(
        self,
        monkeypatch: pytest.MonkeyPatch,
        simple_settings: Settings,
    ) -> None:
        """Tests that the search method calls the hybrid_search function."""
        repository = Mock()
        model = Mock()
        index = Mock()
        vectorizer = Mock()
        matrix = csr_matrix((2, 2))

        expected_results = [Mock(), Mock()]
        hybrid_search_mock = Mock(return_value=expected_results)

        monkeypatch.setattr(
            search_service,
            "hybrid_search",
            hybrid_search_mock,
        )

        service = search_service.SearchService(
            paper_repository=repository,
            model=model,
            index=index,
            vectorizer=vectorizer,
            tfidf_matrix=matrix,
            settings=simple_settings,
        )

        result = service.search(
            query="finite volume methods",
            result_k=10,
        )

        assert result == expected_results

        hybrid_search_mock.assert_called_once_with(
            query="finite volume methods",
            paper_repository=repository,
            model=model,
            index=index,
            vectorizer=vectorizer,
            tfidf_matrix=matrix,
            candidate_k=100,
            result_k=10,
            semantic_weight=0.65,
            tfidf_weight=0.20,
            keyword_weight=0.10,
            recency_weight=0.05,
            half_life_years=5.0,
        )
