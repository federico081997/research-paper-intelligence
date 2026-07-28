"""Tests for the FastAPI application dependencies."""

from typing import get_args
from unittest.mock import Mock

import pytest
from fastapi import Request
from fastapi.params import Depends

from research_paper_intelligence.api.dependencies import (
    AssistantServiceDependency,
    SearchServiceDependency,
    get_assistant_service,
    get_search_service,
)
from research_paper_intelligence.services.assistant_service import (
    ResearchAssistant,
)
from research_paper_intelligence.services.search_service import SearchService


class TestGetSearchService:
    """Tests for the get_search_service dependency."""

    def test_returns_search_service_from_application_state(
        self,
        search_service: Mock,
    ) -> None:
        """Return the search service stored in the application state."""
        request = Mock(spec=Request)
        request.app.state.search_service = search_service

        result = get_search_service(request)

        assert result is search_service


class TestGetAssistantService:
    """Tests for the get_assistant_service dependency."""

    def test_returns_assistant_service_from_application_state(
        self,
    ) -> None:
        """Return the assistant stored in the application state."""
        assistant_service = Mock(spec=ResearchAssistant)
        request = Mock(spec=Request)
        request.app.state.assistant = assistant_service

        result = get_assistant_service(request)

        assert result is assistant_service


class TestDependencyAliases:
    """Tests for the FastAPI annotated dependency aliases."""

    @pytest.mark.parametrize(
        (
            "dependency_alias",
            "expected_service_type",
            "expected_provider",
        ),
        [
            (
                SearchServiceDependency,
                SearchService,
                get_search_service,
            ),
            (
                AssistantServiceDependency,
                ResearchAssistant,
                get_assistant_service,
            ),
        ],
    )
    def test_configures_expected_dependency_provider(
        self,
        dependency_alias: object,
        expected_service_type: type[object],
        expected_provider: object,
    ) -> None:
        """Associate each service type with its dependency provider."""
        service_type, dependency = get_args(dependency_alias)

        assert service_type is expected_service_type
        assert isinstance(dependency, Depends)
        assert dependency.dependency is expected_provider
