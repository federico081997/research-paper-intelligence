"""Tests for the research-paper search router."""

from collections.abc import Iterator
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from research_paper_intelligence.api.dependencies import (
    get_search_service,
)
from research_paper_intelligence.api.routers.search_router import search_router
from research_paper_intelligence.domain.search_result import SearchResult


@pytest.fixture
def client(search_service: Mock) -> Iterator[TestClient]:
    """Create a test client with the search dependency overridden."""
    app = FastAPI()
    app.include_router(search_router)

    def override_get_search_service() -> Mock:
        """Override the get_search_service dependency."""
        return search_service

    app.dependency_overrides[get_search_service] = override_get_search_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


class TestSearchPapers:
    """Test the search-papers endpoint."""

    def test_returns_structured_search_response(
        self,
        client: TestClient,
        search_service: Mock,
        search_result: SearchResult,
    ) -> None:
        """Test that it returns a structured SearchResponse."""
        search_service.search.return_value = [search_result]

        with patch(
            "research_paper_intelligence.api.routers.search_router"
            ".perf_counter",
            side_effect=[10.0, 10.25],
        ):
            response = client.get(
                "/search/",
                params={
                    "query": "finite volume elastoplasticity",
                    "result_k": 5,
                },
            )

        assert response.status_code == 200

        response_data = response.json()

        assert response_data["total"] == 1
        assert response_data["time_elapsed"] == pytest.approx(0.25)

        assert len(response_data["results"]) == 1
        assert response_data["results"][0]["paper_id"] == (
            search_result.paper.paper_id
        )
        assert response_data["results"][0]["title"] == (
            search_result.paper.title
        )
        assert response_data["results"][0]["rank"] == search_result.rank
        assert response_data["results"][0]["hybrid_score"] == pytest.approx(
            search_result.hybrid_score
        )

        search_service.search.assert_called_once_with(
            "finite volume elastoplasticity",
            5,
        )

    def test_uses_default_result_count(
        self,
        client: TestClient,
        search_service: Mock,
    ) -> None:
        """Test the default result count."""
        search_service.search.return_value = []

        with patch(
            "research_paper_intelligence.api.routers.search_router."
            "perf_counter",
            side_effect=[20.0, 20.1],
        ):
            response = client.get(
                "/search/",
                params={"query": "machine learning"},
            )

        assert response.status_code == 200
        assert response.json()["results"] == []
        assert response.json()["total"] == 0

        search_service.search.assert_called_once_with(
            "machine learning",
            10,
        )

    @pytest.mark.parametrize(
        "params",
        [
            {},
            {"query": ""},
            {"query": "machine learning", "result_k": 0},
            {"query": "machine learning", "result_k": 101},
            {"query": "a" * 501},
        ],
    )
    def test_rejects_invalid_query_parameters(
        self,
        client: TestClient,
        search_service: Mock,
        params: dict[str, str | int],
    ) -> None:
        """Test that it rejects invalid query parameters."""
        response = client.get(
            "/search/",
            params=params,
        )

        assert response.status_code == 422
        search_service.search.assert_not_called()
