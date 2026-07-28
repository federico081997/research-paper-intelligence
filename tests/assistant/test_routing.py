"""Tests for the research-assistant graph routing functions."""

from typing import cast
from unittest.mock import Mock

import pytest

from research_paper_intelligence.assistant import routing
from research_paper_intelligence.assistant.routing import (
    route_after_planning,
    route_after_retrieval,
)
from research_paper_intelligence.assistant.state import AssistantState


class TestRouteAfterPlanning:
    """Tests for routing after request planning."""

    def test_routes_direct_request_to_direct_answer(self) -> None:
        """Route a direct request to the direct-answer node."""
        state = cast(
            AssistantState,
            {
                "request_type": "direct",
            },
        )

        result = route_after_planning(state)

        assert result == "generate_direct_answer"

    def test_routes_retrieval_request_to_paper_retrieval(self) -> None:
        """Route a retrieval request to the paper-retrieval node."""
        state = cast(
            AssistantState,
            {
                "request_type": "retrieval",
            },
        )

        result = route_after_planning(state)

        assert result == "retrieve_papers"


class TestRouteAfterRetrieval:
    """Tests for routing after retrieval grading."""

    def test_routes_sufficient_retrieval_to_grounded_answer(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Route sufficient evidence to grounded-answer generation."""
        settings = Mock()
        settings.max_query_rewrites = 2
        get_settings_mock = Mock(return_value=settings)

        monkeypatch.setattr(
            routing,
            "get_settings",
            get_settings_mock,
        )

        state = cast(
            AssistantState,
            {
                "retrieval_sufficient": True,
                "rewrite_count": 0,
            },
        )

        result = route_after_retrieval(state)

        assert result == "generate_grounded_answer"
        get_settings_mock.assert_called_once_with()

    @pytest.mark.parametrize(
        "rewrite_count",
        [
            0,
            1,
        ],
    )
    def test_routes_insufficient_retrieval_to_query_rewrite(
        self,
        monkeypatch: pytest.MonkeyPatch,
        rewrite_count: int,
    ) -> None:
        """Rewrite the query while rewrite attempts remain."""
        settings = Mock()
        settings.max_query_rewrites = 2

        monkeypatch.setattr(
            routing,
            "get_settings",
            Mock(return_value=settings),
        )

        state = cast(
            AssistantState,
            {
                "retrieval_sufficient": False,
                "rewrite_count": rewrite_count,
            },
        )

        result = route_after_retrieval(state)

        assert result == "rewrite_query"

    def test_treats_missing_rewrite_count_as_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Allow a rewrite when no previous rewrite count exists."""
        settings = Mock()
        settings.max_query_rewrites = 2

        monkeypatch.setattr(
            routing,
            "get_settings",
            Mock(return_value=settings),
        )

        state = cast(
            AssistantState,
            {
                "retrieval_sufficient": False,
            },
        )

        result = route_after_retrieval(state)

        assert result == "rewrite_query"

    @pytest.mark.parametrize(
        "rewrite_count",
        [
            2,
            3,
            10,
        ],
    )
    def test_routes_to_limited_answer_when_rewrite_limit_is_reached(
        self,
        monkeypatch: pytest.MonkeyPatch,
        rewrite_count: int,
    ) -> None:
        """Generate a limited answer when no rewrites remain."""
        settings = Mock()
        settings.max_query_rewrites = 2

        monkeypatch.setattr(
            routing,
            "get_settings",
            Mock(return_value=settings),
        )

        state = cast(
            AssistantState,
            {
                "retrieval_sufficient": False,
                "rewrite_count": rewrite_count,
            },
        )

        result = route_after_retrieval(state)

        assert result == "generate_limited_answer"

    def test_routes_to_limited_answer_when_rewrites_are_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Skip rewriting when the configured rewrite limit is zero."""
        settings = Mock()
        settings.max_query_rewrites = 0

        monkeypatch.setattr(
            routing,
            "get_settings",
            Mock(return_value=settings),
        )

        state = cast(
            AssistantState,
            {
                "retrieval_sufficient": False,
            },
        )

        result = route_after_retrieval(state)

        assert result == "generate_limited_answer"

    def test_sufficient_retrieval_takes_priority_over_rewrite_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Use a grounded answer whenever the evidence is sufficient."""
        settings = Mock()
        settings.max_query_rewrites = 2

        monkeypatch.setattr(
            routing,
            "get_settings",
            Mock(return_value=settings),
        )

        state = cast(
            AssistantState,
            {
                "retrieval_sufficient": True,
                "rewrite_count": 2,
            },
        )

        result = route_after_retrieval(state)

        assert result == "generate_grounded_answer"

    def test_uses_configured_maximum_rewrite_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Use the rewrite limit returned by the application settings."""
        settings = Mock()
        settings.max_query_rewrites = 4
        get_settings_mock = Mock(return_value=settings)

        monkeypatch.setattr(
            routing,
            "get_settings",
            get_settings_mock,
        )

        state = cast(
            AssistantState,
            {
                "retrieval_sufficient": False,
                "rewrite_count": 3,
            },
        )

        result = route_after_retrieval(state)

        assert result == "rewrite_query"
        get_settings_mock.assert_called_once_with()
