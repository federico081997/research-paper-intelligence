"""Tests for search request and response schemas."""

from datetime import date

import pytest
from pydantic import ValidationError

from research_paper_intelligence.api.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from research_paper_intelligence.domain.search_result import SearchResult


class TestSearchRequest:
    """Test the search request schema."""

    def test_accepts_valid_request(self) -> None:
        """Test that the schema accepts a valid request."""
        request = SearchRequest(
            query="finite volume elastoplasticity",
            result_k=20,
        )

        assert request.query == "finite volume elastoplasticity"
        assert request.result_k == 20

    def test_uses_default_result_k(self) -> None:
        """Test that the schema uses the default result_k value."""
        request = SearchRequest(query="machine learning")

        assert request.result_k == 10

    def test_rejects_empty_query(self) -> None:
        """Test that the schema rejects an empty query."""
        with pytest.raises(ValidationError):
            SearchRequest(query="")

    def test_rejects_query_longer_than_500_characters(self) -> None:
        """Test that the schema rejects a query longer than 500 characters."""
        with pytest.raises(ValidationError):
            SearchRequest(query="a" * 501)

    @pytest.mark.parametrize("result_k", [-1, 0, 101])
    def test_rejects_invalid_result_k(self, result_k: int) -> None:
        """Test that the schema rejects an invalid result_k value."""
        with pytest.raises(ValidationError):
            SearchRequest(
                query="machine learning",
                result_k=result_k,
            )

    @pytest.mark.parametrize("result_k", [1, 100])
    def test_accepts_result_k_boundary_values(
        self,
        result_k: int,
    ) -> None:
        """Test that the schema accepts result_k boundary values."""
        request = SearchRequest(
            query="machine learning",
            result_k=result_k,
        )

        assert request.result_k == result_k


class TestSearchResultItem:
    """Test conversion of a domain result into an API result item."""

    def test_constructs_item_from_search_result(
        self,
        search_result: SearchResult,
    ) -> None:
        """Test that it constructs a SearchResultItem from a SearchResult."""
        item = SearchResultItem.from_search_result(search_result)

        assert item.paper_id == "paper-001"
        assert item.title == "Finite volume methods"
        assert item.abstract == "An abstract about finite volume methods."
        assert item.authors == "Author One, Author Two"
        assert item.category == "Computational Engineering"
        assert item.published_date == date(2025, 1, 15)
        assert item.rank == 1
        assert item.semantic_score == pytest.approx(0.91)
        assert item.tfidf_score == pytest.approx(0.72)
        assert item.keyword_overlap_score == pytest.approx(0.65)
        assert item.recency_score == pytest.approx(0.80)
        assert item.hybrid_score == pytest.approx(0.84)
        assert item.explanation == "Strong semantic similarity."


class TestSearchResponse:
    """Test construction of the complete search response."""

    def test_constructs_response_from_search_results(
        self,
        search_result: SearchResult,
    ) -> None:
        """Test that it constructs a SearchResponse from search results."""
        response = SearchResponse.from_search_results(
            results=[search_result],
            time_elapsed=42.75,
        )

        assert response.total == 1
        assert response.time_elapsed == pytest.approx(42.75)
        assert len(response.results) == 1
        assert response.results[0].paper_id == "paper-001"
        assert response.results[0].rank == 1

    def test_constructs_empty_response(self) -> None:
        """Test that it constructs an empty SearchResponse."""
        response = SearchResponse.from_search_results(
            results=[],
            time_elapsed=1.25,
        )

        assert response.results == []
        assert response.total == 0
        assert response.time_elapsed == pytest.approx(1.25)
