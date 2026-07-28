"""Tests for the health API endpoint."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from research_paper_intelligence.api.routers.health_router import (
    health_check,
    health_router,
)


@pytest.fixture
def health_app() -> FastAPI:
    """Create a FastAPI application containing the health router."""
    app = FastAPI()
    app.include_router(health_router)

    return app


@pytest.fixture
def health_client(health_app: FastAPI) -> TestClient:
    """Create a test client for the health router."""
    return TestClient(health_app)


class TestHealthCheck:
    """Tests for the health check endpoint."""

    def test_returns_healthy_status(self) -> None:
        """Return a dictionary indicating that the API is healthy."""
        result = health_check()

        assert result == {"status": "healthy"}

    def test_health_endpoint_returns_successful_response(
        self,
        health_client: TestClient,
    ) -> None:
        """Return HTTP 200 from the health endpoint."""
        response = health_client.get("/health")

        assert response.status_code == 200

    def test_health_endpoint_returns_expected_payload(
        self,
        health_client: TestClient,
    ) -> None:
        """Return the expected JSON health payload."""
        response = health_client.get("/health")

        assert response.json() == {"status": "healthy"}

    def test_health_endpoint_returns_json_content_type(
        self,
        health_client: TestClient,
    ) -> None:
        """Return the health response as JSON."""
        response = health_client.get("/health")

        assert response.headers["content-type"] == "application/json"
