"""Tests for the FastAPI application entry point."""

from fastapi import FastAPI

from research_paper_intelligence.api.app import (
    create_app,
    fastapi_app,
)


class TestCreateApp:
    """Test creation and configuration of the FastAPI application."""

    def test_creates_fastapi_application(self) -> None:
        """Create and return a FastAPI application."""
        app = create_app()

        assert isinstance(app, FastAPI)

    def test_configures_application_metadata(self) -> None:
        """Configure the expected API metadata."""
        app = create_app()

        assert app.title == "Research Paper Intelligence API"
        assert app.version == "1.0.0"
        assert app.description == (
            "An API for discovering, ranking and analyzing research "
            "papers using hybrid retrieval and agentic AI workflows."
        )

    def test_registers_search_router(self) -> None:
        """Register the search endpoint under the API version prefix."""
        app = create_app()
        openapi_schema = app.openapi()

        assert "/api/v1/search/" in openapi_schema["paths"]
        assert "get" in openapi_schema["paths"]["/api/v1/search/"]

    def test_search_endpoint_has_expected_tag(self) -> None:
        """Expose the search endpoint under the Search documentation tag."""
        app = create_app()
        search_operation = app.openapi()["paths"]["/api/v1/search/"]["get"]

        assert search_operation["tags"] == ["Search"]


class TestFastAPIApp:
    """Test the module-level FastAPI application."""

    def test_initialises_application_instance(self) -> None:
        """Expose an initialised FastAPI application instance."""
        assert isinstance(fastapi_app, FastAPI)

    def test_initialised_application_contains_search_route(self) -> None:
        """Include the search endpoint in the module-level application."""
        paths = fastapi_app.openapi()["paths"]

        assert "/api/v1/search/" in paths
