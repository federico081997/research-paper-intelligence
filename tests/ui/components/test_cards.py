"""Tests for the Streamlit card-rendering utilities."""

from typing import cast
from unittest.mock import MagicMock, Mock, call

import pytest
from streamlit.navigation.page import StreamlitPage

from research_paper_intelligence.api.schemas.search import (
    SearchResponse,
    SearchResultItem,
)
from research_paper_intelligence.ui.components import cards
from research_paper_intelligence.ui.components.cards import (
    render_feature_card,
    render_result_card,
    render_result_list_card,
)


@pytest.fixture
def container_context() -> MagicMock:
    """Create a mocked Streamlit container context manager."""
    context = MagicMock()
    context.__enter__.return_value = context
    context.__exit__.return_value = False

    return context


@pytest.fixture
def search_response(
    search_results: list[SearchResultItem],
) -> SearchResponse:
    """Create a representative search response."""
    return SearchResponse(
        total=2,
        time_elapsed=0.1254,
        results=search_results,
    )


class TestRenderFeatureCard:
    """Tests for the render_feature_card function."""

    def test_renders_feature_content(
        self,
        monkeypatch: pytest.MonkeyPatch,
        container_context: MagicMock,
    ) -> None:
        """Render the feature title, description, and status."""
        container_mock = Mock(return_value=container_context)
        markdown_mock = Mock()
        write_mock = Mock()
        caption_mock = Mock()
        button_mock = Mock()

        monkeypatch.setattr(cards.st, "container", container_mock)
        monkeypatch.setattr(cards.st, "markdown", markdown_mock)
        monkeypatch.setattr(cards.st, "write", write_mock)
        monkeypatch.setattr(cards.st, "caption", caption_mock)
        monkeypatch.setattr(cards.st, "button", button_mock)

        render_feature_card(
            key="search",
            title="Paper Search",
            description="Search the research-paper collection.",
            icon="🔎",
            status="Available",
        )

        container_mock.assert_called_once_with(key="feature-card-search")
        markdown_mock.assert_called_once_with("### 🔎 Paper Search")
        write_mock.assert_called_once_with(
            "Search the research-paper collection."
        )
        caption_mock.assert_called_once_with("Available")

    def test_renders_page_link_when_page_is_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
        container_context: MagicMock,
    ) -> None:
        """Render a page link for an available feature."""
        page = cast(StreamlitPage, Mock())
        page_link_mock = Mock()
        button_mock = Mock()

        monkeypatch.setattr(
            cards.st,
            "container",
            Mock(return_value=container_context),
        )
        monkeypatch.setattr(cards.st, "markdown", Mock())
        monkeypatch.setattr(cards.st, "write", Mock())
        monkeypatch.setattr(cards.st, "caption", Mock())
        monkeypatch.setattr(cards.st, "page_link", page_link_mock)
        monkeypatch.setattr(cards.st, "button", button_mock)

        render_feature_card(
            key="search",
            title="Paper Search",
            description="Search for papers.",
            icon="🔎",
            status="Available",
            page=page,
            button_label="Open search",
        )

        page_link_mock.assert_called_once_with(
            page,
            label="Open search",
            icon=":material/arrow_forward:",
            width="stretch",
        )
        button_mock.assert_not_called()

    def test_renders_disabled_button_when_page_is_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        container_context: MagicMock,
    ) -> None:
        """Render a disabled placeholder button for a future feature."""
        page_link_mock = Mock()
        button_mock = Mock()

        monkeypatch.setattr(
            cards.st,
            "container",
            Mock(return_value=container_context),
        )
        monkeypatch.setattr(cards.st, "markdown", Mock())
        monkeypatch.setattr(cards.st, "write", Mock())
        monkeypatch.setattr(cards.st, "caption", Mock())
        monkeypatch.setattr(cards.st, "page_link", page_link_mock)
        monkeypatch.setattr(cards.st, "button", button_mock)

        render_feature_card(
            key="assistant",
            title="Assistant",
            description="Ask questions about research papers.",
            icon="🤖",
            status="In development",
        )

        button_mock.assert_called_once_with(
            "Coming soon",
            disabled=True,
            width="stretch",
        )
        page_link_mock.assert_not_called()

    def test_uses_default_page_link_label(
        self,
        monkeypatch: pytest.MonkeyPatch,
        container_context: MagicMock,
    ) -> None:
        """Use the default feature-link label when none is supplied."""
        page = cast(StreamlitPage, Mock())
        page_link_mock = Mock()

        monkeypatch.setattr(
            cards.st,
            "container",
            Mock(return_value=container_context),
        )
        monkeypatch.setattr(cards.st, "markdown", Mock())
        monkeypatch.setattr(cards.st, "write", Mock())
        monkeypatch.setattr(cards.st, "caption", Mock())
        monkeypatch.setattr(cards.st, "page_link", page_link_mock)

        render_feature_card(
            key="search",
            title="Paper Search",
            description="Search for papers.",
            icon="🔎",
            status="Available",
            page=page,
        )

        assert page_link_mock.call_args.kwargs["label"] == "Open feature"

    def test_uses_container_as_context_manager(
        self,
        monkeypatch: pytest.MonkeyPatch,
        container_context: MagicMock,
    ) -> None:
        """Enter and exit the feature-card container."""
        monkeypatch.setattr(
            cards.st,
            "container",
            Mock(return_value=container_context),
        )
        monkeypatch.setattr(cards.st, "markdown", Mock())
        monkeypatch.setattr(cards.st, "write", Mock())
        monkeypatch.setattr(cards.st, "caption", Mock())
        monkeypatch.setattr(cards.st, "button", Mock())

        render_feature_card(
            key="assistant",
            title="Assistant",
            description="Ask research questions.",
            icon="🤖",
            status="Coming soon",
        )

        container_context.__enter__.assert_called_once_with()
        container_context.__exit__.assert_called_once()


class TestRenderResultCard:
    """Tests for the render_result_card function."""

    def test_renders_result_card_content(
        self,
        monkeypatch: pytest.MonkeyPatch,
        container_context: MagicMock,
        search_results: list[SearchResultItem],
    ) -> None:
        """Render the paper title, metadata, and hybrid score."""
        result = search_results[0]
        button_mock = Mock()
        caption_mock = Mock()
        on_click = Mock()

        monkeypatch.setattr(
            cards.st,
            "session_state",
            {"selected_paper_id": None},
        )
        monkeypatch.setattr(
            cards.st,
            "container",
            Mock(return_value=container_context),
        )
        monkeypatch.setattr(cards.st, "button", button_mock)
        monkeypatch.setattr(cards.st, "caption", caption_mock)

        render_result_card(
            result=result,
            selected_paper_id_key="selected_paper_id",
            on_click=on_click,
        )

        button_mock.assert_called_once_with(
            label="**#1 - Finite volume methods**",
            key="result-title-2401.12345",
            type="tertiary",
            width="stretch",
            on_click=on_click,
            args=("2401.12345",),
        )
        assert caption_mock.call_args_list == [
            call(
                "Author One, Author Two  ·  "
                "Computational Engineering  ·  "
                "2025-01-15"
            ),
            call("Hybrid match score: 0.840"),
        ]

    def test_uses_selected_button_key_for_selected_paper(
        self,
        monkeypatch: pytest.MonkeyPatch,
        container_context: MagicMock,
        search_results: list[SearchResultItem],
    ) -> None:
        """Use the selected title key when the paper is selected."""
        result = search_results[0]
        button_mock = Mock()

        monkeypatch.setattr(
            cards.st,
            "session_state",
            {"selected_paper_id": result.paper_id},
        )
        monkeypatch.setattr(
            cards.st,
            "container",
            Mock(return_value=container_context),
        )
        monkeypatch.setattr(cards.st, "button", button_mock)
        monkeypatch.setattr(cards.st, "caption", Mock())

        render_result_card(
            result=result,
            selected_paper_id_key="selected_paper_id",
            on_click=Mock(),
        )

        assert button_mock.call_args.kwargs["key"] == (
            "selected-result-title-2401.12345"
        )

    def test_uses_unselected_button_key_for_other_paper(
        self,
        monkeypatch: pytest.MonkeyPatch,
        container_context: MagicMock,
        search_results: list[SearchResultItem],
    ) -> None:
        """Use the normal title key when another paper is selected."""
        result = search_results[0]
        button_mock = Mock()

        monkeypatch.setattr(
            cards.st,
            "session_state",
            {"selected_paper_id": "different-paper"},
        )
        monkeypatch.setattr(
            cards.st,
            "container",
            Mock(return_value=container_context),
        )
        monkeypatch.setattr(cards.st, "button", button_mock)
        monkeypatch.setattr(cards.st, "caption", Mock())

        render_result_card(
            result=result,
            selected_paper_id_key="selected_paper_id",
            on_click=Mock(),
        )

        assert button_mock.call_args.kwargs["key"] == (
            "result-title-2401.12345"
        )

    def test_creates_bordered_container_with_paper_id_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        container_context: MagicMock,
        search_results: list[SearchResultItem],
    ) -> None:
        """Create a bordered result container with a stable key."""
        result = search_results[0]
        container_mock = Mock(return_value=container_context)

        monkeypatch.setattr(
            cards.st,
            "session_state",
            {"selected_paper_id": None},
        )
        monkeypatch.setattr(cards.st, "container", container_mock)
        monkeypatch.setattr(cards.st, "button", Mock())
        monkeypatch.setattr(cards.st, "caption", Mock())

        render_result_card(
            result=result,
            selected_paper_id_key="selected_paper_id",
            on_click=Mock(),
        )

        container_mock.assert_called_once_with(
            key="result-card-2401.12345",
            border=True,
        )

    def test_passes_paper_id_to_click_callback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        container_context: MagicMock,
        search_results: list[SearchResultItem],
    ) -> None:
        """Configure the button to pass the paper ID to its callback."""
        result = search_results[0]
        on_click = Mock()
        button_mock = Mock()

        monkeypatch.setattr(
            cards.st,
            "session_state",
            {"selected_paper_id": None},
        )
        monkeypatch.setattr(
            cards.st,
            "container",
            Mock(return_value=container_context),
        )
        monkeypatch.setattr(cards.st, "button", button_mock)
        monkeypatch.setattr(cards.st, "caption", Mock())

        render_result_card(
            result=result,
            selected_paper_id_key="selected_paper_id",
            on_click=on_click,
        )

        assert button_mock.call_args.kwargs["on_click"] is on_click
        assert button_mock.call_args.kwargs["args"] == (result.paper_id,)

    def test_formats_hybrid_score_to_three_decimal_places(
        self,
        monkeypatch: pytest.MonkeyPatch,
        container_context: MagicMock,
        search_results: list[SearchResultItem],
    ) -> None:
        """Display the hybrid score with three decimal places."""
        result = search_results[0].model_copy(update={"hybrid_score": 0.8})
        caption_mock = Mock()

        monkeypatch.setattr(
            cards.st,
            "session_state",
            {"selected_paper_id": None},
        )
        monkeypatch.setattr(
            cards.st,
            "container",
            Mock(return_value=container_context),
        )
        monkeypatch.setattr(cards.st, "button", Mock())
        monkeypatch.setattr(cards.st, "caption", caption_mock)

        render_result_card(
            result=result,
            selected_paper_id_key="selected_paper_id",
            on_click=Mock(),
        )

        caption_mock.assert_any_call("Hybrid match score: 0.800")

    def test_raises_key_error_when_selection_state_is_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        search_results: list[SearchResultItem],
    ) -> None:
        """Raise a KeyError when the selected-paper key is uninitialised."""
        monkeypatch.setattr(cards.st, "session_state", {})

        with pytest.raises(
            KeyError,
            match="selected_paper_id",
        ):
            render_result_card(
                result=search_results[0],
                selected_paper_id_key="selected_paper_id",
                on_click=Mock(),
            )


class TestRenderResultListCard:
    """Tests for the render_result_list_card function."""

    def test_renders_result_list_header(
        self,
        monkeypatch: pytest.MonkeyPatch,
        container_context: MagicMock,
        search_response: SearchResponse,
    ) -> None:
        """Render the result-list heading and summary caption."""
        subheader_mock = Mock()
        caption_mock = Mock()

        monkeypatch.setattr(cards.st, "subheader", subheader_mock)
        monkeypatch.setattr(cards.st, "caption", caption_mock)
        monkeypatch.setattr(
            cards.st,
            "container",
            Mock(return_value=container_context),
        )
        monkeypatch.setattr(cards, "render_result_card", Mock())

        render_result_list_card(
            response=search_response,
            selected_paper_id_key="selected_paper_id",
            on_click=Mock(),
        )

        subheader_mock.assert_called_once_with("Search results")
        caption_mock.assert_called_once_with(
            "2 papers returned in 0.125 seconds"
        )

    def test_creates_scrollable_result_container(
        self,
        monkeypatch: pytest.MonkeyPatch,
        container_context: MagicMock,
        search_response: SearchResponse,
    ) -> None:
        """Create the result list using the configured container options."""
        container_mock = Mock(return_value=container_context)

        monkeypatch.setattr(cards.st, "subheader", Mock())
        monkeypatch.setattr(cards.st, "caption", Mock())
        monkeypatch.setattr(cards.st, "container", container_mock)
        monkeypatch.setattr(cards, "render_result_card", Mock())

        render_result_list_card(
            response=search_response,
            selected_paper_id_key="selected_paper_id",
            on_click=Mock(),
        )

        container_mock.assert_called_once_with(
            height=760,
            key="result-list-body",
            border=False,
        )

    def test_renders_each_result_in_response_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
        container_context: MagicMock,
        search_response: SearchResponse,
        search_results: list[SearchResultItem],
    ) -> None:
        """Render every result in the order returned by the API."""
        on_click = Mock()
        render_result_card_mock = Mock()

        monkeypatch.setattr(cards.st, "subheader", Mock())
        monkeypatch.setattr(cards.st, "caption", Mock())
        monkeypatch.setattr(
            cards.st,
            "container",
            Mock(return_value=container_context),
        )
        monkeypatch.setattr(
            cards,
            "render_result_card",
            render_result_card_mock,
        )

        render_result_list_card(
            response=search_response,
            selected_paper_id_key="selected_paper_id",
            on_click=on_click,
        )

        assert render_result_card_mock.call_args_list == [
            call(
                search_results[0],
                "selected_paper_id",
                on_click,
            ),
            call(
                search_results[1],
                "selected_paper_id",
                on_click,
            ),
        ]

    def test_does_not_render_result_cards_for_empty_response(
        self,
        monkeypatch: pytest.MonkeyPatch,
        container_context: MagicMock,
    ) -> None:
        """Render no paper cards when the response contains no results."""
        response = SearchResponse(
            total=0,
            time_elapsed=0.025,
            results=[],
        )
        render_result_card_mock = Mock()

        monkeypatch.setattr(cards.st, "subheader", Mock())
        monkeypatch.setattr(cards.st, "caption", Mock())
        monkeypatch.setattr(
            cards.st,
            "container",
            Mock(return_value=container_context),
        )
        monkeypatch.setattr(
            cards,
            "render_result_card",
            render_result_card_mock,
        )

        render_result_list_card(
            response=response,
            selected_paper_id_key="selected_paper_id",
            on_click=Mock(),
        )

        render_result_card_mock.assert_not_called()
