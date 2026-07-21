"""Tests for the FastAPI dependencies."""

from unittest.mock import Mock

from fastapi import FastAPI, Request

from research_paper_intelligence.api.dependencies import (
    get_search_service,
)
from research_paper_intelligence.services.search_service import SearchService


class TestGetSearchService:
    """Test retrieval of the search service dependency."""

    def test_returns_search_service_from_application_state(
        self,
        search_service: SearchService,
    ) -> None:
        """Return the service stored in the FastAPI application state."""
        app = FastAPI()
        app.state.search_service = search_service

        request = Mock(spec=Request)
        request.app = app

        result = get_search_service(request)

        assert result is search_service
