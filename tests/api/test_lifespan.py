"""Tests for the FastAPI application lifespan."""

from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from research_paper_intelligence.api.lifespan import lifespan


class TestLifespan:
    """Test the FastAPI application lifespan."""

    def test_initialises_and_releases_search_service(
        self,
        search_service: Mock,
    ) -> None:
        """Initialize the service at startup and release it at shutdown."""
        app = FastAPI(lifespan=lifespan)

        with patch(
            "research_paper_intelligence.api.lifespan.create_search_service",
            return_value=search_service,
        ) as create_search_service_mock:
            assert getattr(app.state, "search_service", None) is None

            with TestClient(app):
                assert app.state.search_service is search_service
                create_search_service_mock.assert_called_once_with()

            assert app.state.search_service is None
