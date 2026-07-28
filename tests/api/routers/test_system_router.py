"""Tests for the system-information API router."""

from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from research_paper_intelligence.api.dependencies import get_search_service
from research_paper_intelligence.api.routers.system_router import (
    get_system_info,
    system_router,
)
from research_paper_intelligence.api.schemas.system import SystemInfoResponse


@pytest.fixture
def system_search_service() -> Mock:
    """Create a mocked search service containing system metadata."""
    service = Mock()
    service.paper_count = 130_000
    service.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    service.retrieval_strategy = "hybrid"
    service.ranking_components = [
        "semantic",
        "tfidf",
        "keyword",
        "recency",
    ]
    service.faiss_index_type = "IndexFlatIP"
    service.faiss_index_size = 130_000
    service.tfidf_document_count = 130_000
    service.tfidf_vocabulary_size = 50_000

    return service


@pytest.fixture
def system_app(system_search_service: Mock) -> FastAPI:
    """Create a FastAPI application containing the system router."""
    app = FastAPI(version="1.0.0")
    app.include_router(system_router)

    app.dependency_overrides[get_search_service] = lambda: (
        system_search_service
    )

    return app


@pytest.fixture
def system_client(system_app: FastAPI) -> TestClient:
    """Create a test client for the system router."""
    return TestClient(system_app)


class TestGetSystemInfo:
    """Tests for the system-information endpoint."""

    def test_returns_system_information_from_search_service(
        self,
        system_search_service: Mock,
    ) -> None:
        """Build the response from the search-service metadata."""
        request = Mock(spec=Request)
        request.app.version = "1.0.0"

        result = get_system_info(
            request=request,
            search_request=system_search_service,
        )

        assert result == SystemInfoResponse(
            status="ready",
            paper_count=130_000,
            embedding_model=("sentence-transformers/all-MiniLM-L6-v2"),
            retrieval_strategy="hybrid",
            ranking_components=[
                "semantic",
                "tfidf",
                "keyword",
                "recency",
            ],
            faiss_index_type="IndexFlatIP",
            faiss_index_size=130_000,
            tfidf_document_count=130_000,
            tfidf_vocabulary_size=50_000,
            api_version="1.0.0",
        )

    def test_system_endpoint_returns_successful_response(
        self,
        system_client: TestClient,
    ) -> None:
        """Return HTTP 200 from the system-information endpoint."""
        response = system_client.get("/system/")

        assert response.status_code == 200

    def test_system_endpoint_returns_expected_payload(
        self,
        system_client: TestClient,
    ) -> None:
        """Return the expected system-information payload."""
        response = system_client.get("/system/")

        assert response.json() == {
            "status": "ready",
            "paper_count": 130_000,
            "embedding_model": ("sentence-transformers/all-MiniLM-L6-v2"),
            "retrieval_strategy": "hybrid",
            "ranking_components": [
                "semantic",
                "tfidf",
                "keyword",
                "recency",
            ],
            "faiss_index_type": "IndexFlatIP",
            "faiss_index_size": 130_000,
            "tfidf_document_count": 130_000,
            "tfidf_vocabulary_size": 50_000,
            "api_version": "1.0.0",
        }

    @pytest.mark.parametrize(
        ("response_field", "service_attribute"),
        [
            ("paper_count", "paper_count"),
            ("embedding_model", "embedding_model"),
            ("retrieval_strategy", "retrieval_strategy"),
            ("ranking_components", "ranking_components"),
            ("faiss_index_type", "faiss_index_type"),
            ("faiss_index_size", "faiss_index_size"),
            ("tfidf_document_count", "tfidf_document_count"),
            ("tfidf_vocabulary_size", "tfidf_vocabulary_size"),
        ],
    )
    def test_maps_search_service_metadata_to_response(
        self,
        system_client: TestClient,
        system_search_service: Mock,
        response_field: str,
        service_attribute: str,
    ) -> None:
        """Map each search-service property to its response field."""
        response = system_client.get("/system/")

        assert response.json()[response_field] == getattr(
            system_search_service,
            service_attribute,
        )

    def test_uses_application_version(
        self,
        system_client: TestClient,
    ) -> None:
        """Read the API version from the FastAPI application."""
        response = system_client.get("/system/")

        assert response.json()["api_version"] == "1.0.0"
