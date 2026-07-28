"""Tests for the Research Paper Intelligence search page."""

import importlib
import sys
from collections.abc import Generator
from types import ModuleType
from unittest.mock import MagicMock, Mock, call

import pytest
import streamlit as st

from research_paper_intelligence.api.schemas.search import (
    SearchResponse,
    SearchResultItem,
)

SEARCH_PAGE_MODULE = "research_paper_intelligence.ui.pages.search"
SEARCH_CLIENT_MODULE = (
    "research_paper_intelligence.ui.api_clients.search_client"
)
CARDS_MODULE = "research_paper_intelligence.ui.components.cards"


@pytest.fixture
def search_response(
    search_results: list[SearchResultItem],
) -> SearchResponse:
    """Create a representative search response."""
    return SearchResponse(
        total=2,
        time_elapsed=0.125,
        results=search_results,
    )


@pytest.fixture
def empty_search_response() -> SearchResponse:
    """Create a search response containing no papers."""
    return SearchResponse(
        total=0,
        time_elapsed=0.025,
        results=[],
    )


@pytest.fixture
def context_manager() -> MagicMock:
    """Create a mocked context manager."""
    context = MagicMock()
    context.__enter__.return_value = context
    context.__exit__.return_value = False

    return context


@pytest.fixture
def search_page(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[ModuleType, None, None]:
    """Import the search page with import-time dependencies mocked."""
    session_state: dict[str, object] = {}

    search_papers_mock = Mock()
    render_result_list_card_mock = Mock()

    search_client_module = ModuleType(SEARCH_CLIENT_MODULE)
    search_client_module.search_papers = search_papers_mock

    cards_module = ModuleType(CARDS_MODULE)
    cards_module.render_result_list_card = render_result_list_card_mock

    monkeypatch.setitem(
        sys.modules,
        SEARCH_CLIENT_MODULE,
        search_client_module,
    )
    monkeypatch.setitem(
        sys.modules,
        CARDS_MODULE,
        cards_module,
    )

    container_context = MagicMock()
    form_context = MagicMock()
    spinner_context = MagicMock()
    form_columns = [
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]

    monkeypatch.setattr(
        st,
        "session_state",
        session_state,
        raising=False,
    )
    monkeypatch.setattr(st, "title", Mock())
    monkeypatch.setattr(st, "markdown", Mock())
    monkeypatch.setattr(st, "caption", Mock())
    monkeypatch.setattr(st, "subheader", Mock())
    monkeypatch.setattr(st, "info", Mock())
    monkeypatch.setattr(st, "error", Mock())
    monkeypatch.setattr(st, "write", Mock())
    monkeypatch.setattr(st, "metric", Mock())
    monkeypatch.setattr(st, "link_button", Mock())
    monkeypatch.setattr(st, "divider", Mock())
    monkeypatch.setattr(
        st,
        "container",
        Mock(return_value=container_context),
    )
    monkeypatch.setattr(
        st,
        "form",
        Mock(return_value=form_context),
    )
    monkeypatch.setattr(
        st,
        "spinner",
        Mock(return_value=spinner_context),
    )
    monkeypatch.setattr(
        st,
        "columns",
        Mock(return_value=form_columns),
    )
    monkeypatch.setattr(
        st,
        "text_input",
        Mock(return_value=""),
    )
    monkeypatch.setattr(
        st,
        "number_input",
        Mock(return_value=10),
    )
    monkeypatch.setattr(
        st,
        "form_submit_button",
        Mock(return_value=False),
    )

    sys.modules.pop(SEARCH_PAGE_MODULE, None)

    module = importlib.import_module(SEARCH_PAGE_MODULE)

    session_state.clear()

    for function_name in (
        "title",
        "markdown",
        "caption",
        "subheader",
        "info",
        "error",
        "write",
        "metric",
        "link_button",
        "divider",
        "container",
        "form",
        "spinner",
        "columns",
        "text_input",
        "number_input",
        "form_submit_button",
    ):
        getattr(st, function_name).reset_mock()

    search_papers_mock.reset_mock()
    render_result_list_card_mock.reset_mock()

    yield module

    sys.modules.pop(SEARCH_PAGE_MODULE, None)


class TestInitializeSearchState:
    """Tests for search-page state initialization."""

    def test_initializes_missing_state_values(
        self,
        search_page: ModuleType,
    ) -> None:
        """Create all required search state values."""
        search_page.initialize_search_state()

        state = search_page.st.session_state

        assert state == {
            search_page.SEARCH_RESPONSE_KEY: None,
            search_page.SELECTED_PAPER_ID_KEY: None,
            search_page.LAST_SEARCH_QUERY_KEY: "",
        }

    def test_preserves_existing_state_values(
        self,
        search_page: ModuleType,
        search_response: SearchResponse,
    ) -> None:
        """Leave existing search state values unchanged."""
        search_page.st.session_state.update(
            {
                search_page.SEARCH_RESPONSE_KEY: search_response,
                search_page.SELECTED_PAPER_ID_KEY: "2401.12345",
                search_page.LAST_SEARCH_QUERY_KEY: "existing query",
            }
        )

        search_page.initialize_search_state()

        assert (
            search_page.st.session_state[search_page.SEARCH_RESPONSE_KEY]
            is search_response
        )
        assert (
            search_page.st.session_state[search_page.SELECTED_PAPER_ID_KEY]
            == "2401.12345"
        )
        assert (
            search_page.st.session_state[search_page.LAST_SEARCH_QUERY_KEY]
            == "existing query"
        )


class TestRenderHero:
    """Tests for rendering the search-page hero."""

    def test_renders_title_and_description(
        self,
        search_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render the search-page title and introductory text."""
        title_mock = Mock()
        markdown_mock = Mock()

        monkeypatch.setattr(
            search_page.st,
            "title",
            title_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "markdown",
            markdown_mock,
        )

        search_page.render_hero()

        title_mock.assert_called_once_with("Research Paper Search")

        description = markdown_mock.call_args.args[0]

        assert "Retrieve and rank papers" in description
        assert "semantic similarity" in description
        assert "TF-IDF" in description
        assert "publication recency" in description


class TestRenderSearchForm:
    """Tests for rendering the paper-search form."""

    def test_returns_submitted_query_and_result_count(
        self,
        search_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Return values entered through the search form."""
        monkeypatch.setattr(
            search_page.st,
            "container",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            search_page.st,
            "form",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            search_page.st,
            "columns",
            Mock(
                return_value=[
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                ]
            ),
        )
        monkeypatch.setattr(
            search_page.st,
            "markdown",
            Mock(),
        )
        monkeypatch.setattr(
            search_page.st,
            "caption",
            Mock(),
        )
        monkeypatch.setattr(
            search_page.st,
            "text_input",
            Mock(return_value="finite volume methods"),
        )
        monkeypatch.setattr(
            search_page.st,
            "number_input",
            Mock(return_value=20),
        )
        monkeypatch.setattr(
            search_page.st,
            "form_submit_button",
            Mock(return_value=True),
        )

        result = search_page.render_search_form()

        assert result == (
            True,
            "finite volume methods",
            20,
        )

    def test_creates_search_form_container(
        self,
        search_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Create the outer card and inner Streamlit form."""
        container_context = MagicMock()
        form_context = MagicMock()
        container_mock = Mock(return_value=container_context)
        form_mock = Mock(return_value=form_context)

        monkeypatch.setattr(
            search_page.st,
            "container",
            container_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "form",
            form_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "columns",
            Mock(
                return_value=[
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                ]
            ),
        )
        monkeypatch.setattr(
            search_page.st,
            "markdown",
            Mock(),
        )
        monkeypatch.setattr(
            search_page.st,
            "caption",
            Mock(),
        )
        monkeypatch.setattr(
            search_page.st,
            "text_input",
            Mock(return_value="query"),
        )
        monkeypatch.setattr(
            search_page.st,
            "number_input",
            Mock(return_value=10),
        )
        monkeypatch.setattr(
            search_page.st,
            "form_submit_button",
            Mock(return_value=False),
        )

        search_page.render_search_form()

        container_mock.assert_called_once_with(
            key="search-form-card",
            border=True,
        )
        form_mock.assert_called_once_with(
            key="search-form",
            clear_on_submit=False,
            border=False,
        )

        container_context.__enter__.assert_called_once_with()
        container_context.__exit__.assert_called_once()
        form_context.__enter__.assert_called_once_with()
        form_context.__exit__.assert_called_once()

    def test_creates_search_form_columns(
        self,
        search_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Create columns for the query, result count, and button."""
        columns = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]
        columns_mock = Mock(return_value=columns)

        monkeypatch.setattr(
            search_page.st,
            "container",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            search_page.st,
            "form",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            search_page.st,
            "columns",
            columns_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "markdown",
            Mock(),
        )
        monkeypatch.setattr(
            search_page.st,
            "caption",
            Mock(),
        )
        monkeypatch.setattr(
            search_page.st,
            "text_input",
            Mock(return_value="query"),
        )
        monkeypatch.setattr(
            search_page.st,
            "number_input",
            Mock(return_value=10),
        )
        monkeypatch.setattr(
            search_page.st,
            "form_submit_button",
            Mock(return_value=False),
        )

        search_page.render_search_form()

        columns_mock.assert_called_once_with(
            [3, 1, 1],
            vertical_alignment="bottom",
        )

        for column in columns:
            column.__enter__.assert_called_once_with()
            column.__exit__.assert_called_once()

    def test_configures_search_query_input(
        self,
        search_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Configure the search-query text field."""
        text_input_mock = Mock(return_value="query")

        monkeypatch.setattr(
            search_page.st,
            "container",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            search_page.st,
            "form",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            search_page.st,
            "columns",
            Mock(
                return_value=[
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                ]
            ),
        )
        monkeypatch.setattr(
            search_page.st,
            "markdown",
            Mock(),
        )
        monkeypatch.setattr(
            search_page.st,
            "caption",
            Mock(),
        )
        monkeypatch.setattr(
            search_page.st,
            "text_input",
            text_input_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "number_input",
            Mock(return_value=10),
        )
        monkeypatch.setattr(
            search_page.st,
            "form_submit_button",
            Mock(return_value=False),
        )

        search_page.render_search_form()

        text_input_mock.assert_called_once_with(
            label="Search query",
            placeholder="e.g. machine learning",
            value="",
            max_chars=500,
        )

    def test_configures_result_count_input(
        self,
        search_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Configure the number-of-results input."""
        number_input_mock = Mock(return_value=10)

        monkeypatch.setattr(
            search_page.st,
            "container",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            search_page.st,
            "form",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            search_page.st,
            "columns",
            Mock(
                return_value=[
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                ]
            ),
        )
        monkeypatch.setattr(
            search_page.st,
            "markdown",
            Mock(),
        )
        monkeypatch.setattr(
            search_page.st,
            "caption",
            Mock(),
        )
        monkeypatch.setattr(
            search_page.st,
            "text_input",
            Mock(return_value="query"),
        )
        monkeypatch.setattr(
            search_page.st,
            "number_input",
            number_input_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "form_submit_button",
            Mock(return_value=False),
        )

        search_page.render_search_form()

        number_input_mock.assert_called_once_with(
            label="Number of results",
            min_value=1,
            max_value=100,
            value=10,
            step=5,
        )

    def test_configures_submit_button(
        self,
        search_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Configure the primary search form button."""
        submit_mock = Mock(return_value=False)

        monkeypatch.setattr(
            search_page.st,
            "container",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            search_page.st,
            "form",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            search_page.st,
            "columns",
            Mock(
                return_value=[
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                ]
            ),
        )
        monkeypatch.setattr(
            search_page.st,
            "markdown",
            Mock(),
        )
        monkeypatch.setattr(
            search_page.st,
            "caption",
            Mock(),
        )
        monkeypatch.setattr(
            search_page.st,
            "text_input",
            Mock(return_value="query"),
        )
        monkeypatch.setattr(
            search_page.st,
            "number_input",
            Mock(return_value=10),
        )
        monkeypatch.setattr(
            search_page.st,
            "form_submit_button",
            submit_mock,
        )

        search_page.render_search_form()

        submit_mock.assert_called_once_with(
            label="Search papers",
            type="primary",
            key="search-submit-button",
            icon=":material/search:",
            width="stretch",
        )


class TestPerformSearch:
    """Tests for performing searches."""

    def test_stores_successful_search_response(
        self,
        search_page: ModuleType,
        search_response: SearchResponse,
        monkeypatch: pytest.MonkeyPatch,
        context_manager: MagicMock,
    ) -> None:
        """Store the response and query after a successful search."""
        search_mock = Mock(return_value=search_response)

        monkeypatch.setattr(
            search_page,
            "search_papers",
            search_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "spinner",
            Mock(return_value=context_manager),
        )

        search_page.perform_search(
            query="  finite volume methods  ",
            result_k=10,
        )

        search_mock.assert_called_once_with(
            "  finite volume methods  ",
            10,
        )

        assert (
            search_page.st.session_state[search_page.SEARCH_RESPONSE_KEY]
            is search_response
        )
        assert (
            search_page.st.session_state[search_page.LAST_SEARCH_QUERY_KEY]
            == "finite volume methods"
        )
        assert (
            search_page.st.session_state[search_page.SELECTED_PAPER_ID_KEY]
            == "2401.12345"
        )

    def test_selects_no_paper_when_response_is_empty(
        self,
        search_page: ModuleType,
        empty_search_response: SearchResponse,
        monkeypatch: pytest.MonkeyPatch,
        context_manager: MagicMock,
    ) -> None:
        """Store no selected paper when the search has no results."""
        monkeypatch.setattr(
            search_page,
            "search_papers",
            Mock(return_value=empty_search_response),
        )
        monkeypatch.setattr(
            search_page.st,
            "spinner",
            Mock(return_value=context_manager),
        )

        search_page.perform_search(
            query="unknown topic",
            result_k=10,
        )

        assert (
            search_page.st.session_state[search_page.SELECTED_PAPER_ID_KEY]
            is None
        )

    def test_renders_search_spinner(
        self,
        search_page: ModuleType,
        search_response: SearchResponse,
        monkeypatch: pytest.MonkeyPatch,
        context_manager: MagicMock,
    ) -> None:
        """Show a spinner while retrieving and ranking papers."""
        spinner_mock = Mock(return_value=context_manager)

        monkeypatch.setattr(
            search_page,
            "search_papers",
            Mock(return_value=search_response),
        )
        monkeypatch.setattr(
            search_page.st,
            "spinner",
            spinner_mock,
        )

        search_page.perform_search(
            query="semantic search",
            result_k=5,
        )

        spinner_mock.assert_called_once_with("Searching and ranking papers...")
        context_manager.__enter__.assert_called_once_with()
        context_manager.__exit__.assert_called_once()

    def test_handles_http_status_error(
        self,
        search_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        context_manager: MagicMock,
    ) -> None:
        """Display the backend status code for an HTTP error."""

        class TestHTTPStatusError(Exception):
            """HTTP status error used by the test."""

            def __init__(self, status_code: int) -> None:
                super().__init__("HTTP status error")
                self.response = Mock(status_code=status_code)

        logger_mock = Mock()
        error_mock = Mock()

        monkeypatch.setattr(
            search_page.httpx2,
            "HTTPStatusError",
            TestHTTPStatusError,
        )
        monkeypatch.setattr(
            search_page,
            "search_papers",
            Mock(side_effect=TestHTTPStatusError(503)),
        )
        monkeypatch.setattr(
            search_page,
            "LOGGER",
            logger_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "spinner",
            Mock(return_value=context_manager),
        )
        monkeypatch.setattr(
            search_page.st,
            "error",
            error_mock,
        )

        search_page.perform_search(
            query="semantic search",
            result_k=5,
        )

        logger_mock.exception.assert_called_once_with(
            "Search API returned an HTTP error."
        )
        error_mock.assert_called_once_with(
            "The search service returned an error (503)"
        )

    def test_handles_request_error(
        self,
        search_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        context_manager: MagicMock,
    ) -> None:
        """Display an error when the backend cannot be reached."""

        class TestRequestError(Exception):
            """Request error used by the test."""

        logger_mock = Mock()
        error_mock = Mock()

        monkeypatch.setattr(
            search_page.httpx2,
            "RequestError",
            TestRequestError,
        )
        monkeypatch.setattr(
            search_page,
            "search_papers",
            Mock(side_effect=TestRequestError()),
        )
        monkeypatch.setattr(
            search_page,
            "LOGGER",
            logger_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "spinner",
            Mock(return_value=context_manager),
        )
        monkeypatch.setattr(
            search_page.st,
            "error",
            error_mock,
        )

        search_page.perform_search(
            query="semantic search",
            result_k=5,
        )

        logger_mock.exception.assert_called_once_with(
            "Could not connect to the search API."
        )
        error_mock.assert_called_once_with(
            "Could not connect to the FastAPI backend. "
            "Make sure the API is running and try again."
        )

    def test_handles_invalid_response_data(
        self,
        search_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        context_manager: MagicMock,
    ) -> None:
        """Display an error when API response validation fails."""

        class TestValidationError(Exception):
            """Validation error used by the test."""

        logger_mock = Mock()
        error_mock = Mock()

        monkeypatch.setattr(
            search_page,
            "ValidationError",
            TestValidationError,
        )
        monkeypatch.setattr(
            search_page,
            "search_papers",
            Mock(side_effect=TestValidationError()),
        )
        monkeypatch.setattr(
            search_page,
            "LOGGER",
            logger_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "spinner",
            Mock(return_value=context_manager),
        )
        monkeypatch.setattr(
            search_page.st,
            "error",
            error_mock,
        )

        search_page.perform_search(
            query="semantic search",
            result_k=5,
        )

        logger_mock.exception.assert_called_once_with(
            "The search API returned invalid data."
        )
        error_mock.assert_called_once_with(
            "The backend returned data in an unexpected format."
        )


class TestBuildArxivUrl:
    """Tests for arXiv URL construction."""

    @pytest.mark.parametrize(
        ("paper_id", "expected_url"),
        [
            (
                "0901.4761v1",
                "https://arxiv.org/abs/0901.4761v1",
            ),
            (
                "cs/9308101v1",
                "https://arxiv.org/abs/cs/9308101v1",
            ),
            (
                "2401.12345",
                "https://arxiv.org/abs/2401.12345",
            ),
        ],
    )
    def test_builds_arxiv_abstract_url(
        self,
        search_page: ModuleType,
        paper_id: str,
        expected_url: str,
    ) -> None:
        """Build the abstract-page URL for a normalized identifier."""
        assert search_page.build_arxiv_url(paper_id) == expected_url


class TestSelectPaper:
    """Tests for selecting a paper."""

    def test_stores_selected_paper_id(
        self,
        search_page: ModuleType,
    ) -> None:
        """Store the selected paper ID in session state."""
        search_page.select_paper("2401.12345")

        assert (
            search_page.st.session_state[search_page.SELECTED_PAPER_ID_KEY]
            == "2401.12345"
        )


class TestGetSelectedPaper:
    """Tests for retrieving the selected paper."""

    def test_returns_selected_paper(
        self,
        search_page: ModuleType,
        search_response: SearchResponse,
        search_results: list[SearchResultItem],
    ) -> None:
        """Return the result matching the selected paper ID."""
        search_page.st.session_state[search_page.SELECTED_PAPER_ID_KEY] = (
            "2402.67890"
        )

        result = search_page.get_selected_paper(search_response)

        assert result is search_results[1]

    def test_returns_none_for_unknown_selected_id(
        self,
        search_page: ModuleType,
        search_response: SearchResponse,
    ) -> None:
        """Return None when the selected ID is not in the response."""
        search_page.st.session_state[search_page.SELECTED_PAPER_ID_KEY] = (
            "unknown-paper"
        )

        result = search_page.get_selected_paper(search_response)

        assert result is None

    def test_returns_none_when_response_is_empty(
        self,
        search_page: ModuleType,
        empty_search_response: SearchResponse,
    ) -> None:
        """Return None when no search results are available."""
        result = search_page.get_selected_paper(empty_search_response)

        assert result is None


class TestRenderScoreSummary:
    """Tests for rendering paper-ranking scores."""

    def test_renders_all_scores_to_three_decimal_places(
        self,
        search_page: ModuleType,
        search_results: list[SearchResultItem],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render all ranking scores with consistent formatting."""
        first_row = [Mock(), Mock(), Mock()]
        second_row = [Mock(), Mock()]
        columns_mock = Mock(side_effect=[first_row, second_row])

        monkeypatch.setattr(
            search_page.st,
            "columns",
            columns_mock,
        )

        search_page.render_score_summary(search_results[0])

        assert columns_mock.call_args_list == [
            call(3),
            call(2),
        ]
        first_row[0].metric.assert_called_once_with(
            label="Hybrid score",
            value="0.840",
        )
        first_row[1].metric.assert_called_once_with(
            label="Semantic",
            value="0.910",
        )
        first_row[2].metric.assert_called_once_with(
            label="TF-IDF",
            value="0.720",
        )
        second_row[0].metric.assert_called_once_with(
            label="Keyword overlap",
            value="0.650",
        )
        second_row[1].metric.assert_called_once_with(
            label="Recency",
            value="0.800",
        )


class TestRenderPaperDetails:
    """Tests for rendering selected-paper details."""

    def test_renders_selection_message_when_result_is_none(
        self,
        search_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ask the user to select a paper when none is selected."""
        subheader_mock = Mock()
        info_mock = Mock()
        container_mock = Mock()

        monkeypatch.setattr(
            search_page.st,
            "subheader",
            subheader_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "info",
            info_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "container",
            container_mock,
        )

        search_page.render_paper_details(None)

        subheader_mock.assert_called_once_with("Paper details")
        info_mock.assert_called_once_with(
            "Select a paper to view its details."
        )
        container_mock.assert_not_called()

    def test_renders_selected_paper_details(
        self,
        search_page: ModuleType,
        search_results: list[SearchResultItem],
        monkeypatch: pytest.MonkeyPatch,
        context_manager: MagicMock,
    ) -> None:
        """Render the selected paper's metadata and content."""
        result = search_results[0]
        markdown_mock = Mock()
        caption_mock = Mock()
        write_mock = Mock()
        link_button_mock = Mock()
        divider_mock = Mock()
        score_summary_mock = Mock()
        container_mock = Mock(return_value=context_manager)

        monkeypatch.setattr(
            search_page.st,
            "subheader",
            Mock(),
        )
        monkeypatch.setattr(
            search_page.st,
            "container",
            container_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "markdown",
            markdown_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "caption",
            caption_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "write",
            write_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "link_button",
            link_button_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "divider",
            divider_mock,
        )
        monkeypatch.setattr(
            search_page,
            "render_score_summary",
            score_summary_mock,
        )

        search_page.render_paper_details(result)

        container_mock.assert_called_once_with(
            key="result-details-card-2401.12345",
            height=760,
        )
        markdown_mock.assert_any_call("### Finite volume methods")
        markdown_mock.assert_any_call("#### Authors")
        markdown_mock.assert_any_call("#### Abstract")
        markdown_mock.assert_any_call("#### Why this paper matched")
        markdown_mock.assert_any_call("#### Ranking scores")

        caption_mock.assert_called_once_with(
            "Rank 1  ·  Computational Engineering  ·  "
            "Published 15 January 2025"
        )

        assert write_mock.call_args_list == [
            call("Author One, Author Two"),
            call("An abstract about finite volume methods."),
            call("Strong semantic similarity."),
        ]

        link_button_mock.assert_called_once_with(
            label="Open on arXiv",
            url="https://arxiv.org/abs/2401.12345",
            width="stretch",
        )
        assert divider_mock.call_count == 2
        score_summary_mock.assert_called_once_with(result)


class TestRenderSearchWorkspace:
    """Tests for rendering the result workspace."""

    def test_renders_empty_result_message(
        self,
        search_page: ModuleType,
        empty_search_response: SearchResponse,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Display guidance when no papers match the search."""
        info_mock = Mock()
        columns_mock = Mock()

        monkeypatch.setattr(
            search_page.st,
            "info",
            info_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "columns",
            columns_mock,
        )

        search_page.render_search_workspace(empty_search_response)

        info_mock.assert_called_once_with(
            "No papers matched this query. Try broader terminology "
            "or fewer constraints."
        )
        columns_mock.assert_not_called()

    def test_renders_results_and_selected_paper(
        self,
        search_page: ModuleType,
        search_response: SearchResponse,
        search_results: list[SearchResultItem],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render the result list and selected-paper details."""
        columns = [MagicMock(), MagicMock()]
        result_list_mock = Mock()
        selected_paper_mock = Mock(return_value=search_results[0])
        paper_details_mock = Mock()

        monkeypatch.setattr(
            search_page.st,
            "columns",
            Mock(return_value=columns),
        )
        monkeypatch.setattr(
            search_page,
            "render_result_list_card",
            result_list_mock,
        )
        monkeypatch.setattr(
            search_page,
            "get_selected_paper",
            selected_paper_mock,
        )
        monkeypatch.setattr(
            search_page,
            "render_paper_details",
            paper_details_mock,
        )

        search_page.render_search_workspace(search_response)

        search_page.st.columns.assert_called_once_with(
            [1, 1],
            gap="small",
        )
        result_list_mock.assert_called_once_with(
            search_response,
            selected_paper_id_key=(search_page.SELECTED_PAPER_ID_KEY),
            on_click=search_page.select_paper,
        )
        selected_paper_mock.assert_called_once_with(search_response)
        paper_details_mock.assert_called_once_with(search_results[0])

        for column in columns:
            column.__enter__.assert_called_once_with()
            column.__exit__.assert_called_once()


class TestRenderSearchPage:
    """Tests for rendering the complete search page."""

    def test_initializes_state_and_renders_page_sections(
        self,
        search_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Initialize state and render the hero and search form."""
        initialize_mock = Mock()
        hero_mock = Mock()
        form_mock = Mock(return_value=(False, "", 10))
        workspace_mock = Mock()

        search_page.st.session_state[search_page.SEARCH_RESPONSE_KEY] = None

        monkeypatch.setattr(
            search_page,
            "initialize_search_state",
            initialize_mock,
        )
        monkeypatch.setattr(
            search_page,
            "render_hero",
            hero_mock,
        )
        monkeypatch.setattr(
            search_page,
            "render_search_form",
            form_mock,
        )
        monkeypatch.setattr(
            search_page,
            "render_search_workspace",
            workspace_mock,
        )

        search_page.render_search_page()

        initialize_mock.assert_called_once_with()
        hero_mock.assert_called_once_with()
        form_mock.assert_called_once_with()
        workspace_mock.assert_not_called()

    def test_returns_without_searching_for_empty_query(
        self,
        search_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Do not search when the submitted query is empty."""
        perform_search_mock = Mock()
        workspace_mock = Mock()

        monkeypatch.setattr(
            search_page,
            "initialize_search_state",
            Mock(),
        )
        monkeypatch.setattr(
            search_page,
            "render_hero",
            Mock(),
        )
        monkeypatch.setattr(
            search_page,
            "render_search_form",
            Mock(return_value=(True, "", 10)),
        )
        monkeypatch.setattr(
            search_page,
            "perform_search",
            perform_search_mock,
        )
        monkeypatch.setattr(
            search_page,
            "render_search_workspace",
            workspace_mock,
        )

        search_page.render_search_page()

        perform_search_mock.assert_not_called()
        workspace_mock.assert_not_called()

    def test_performs_submitted_search(
        self,
        search_page: ModuleType,
        search_response: SearchResponse,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Perform a search using the submitted query and result count."""
        perform_search_mock = Mock()
        workspace_mock = Mock()
        caption_mock = Mock()

        search_page.st.session_state.update(
            {
                search_page.SEARCH_RESPONSE_KEY: search_response,
                search_page.LAST_SEARCH_QUERY_KEY: ("finite volume methods"),
            }
        )

        monkeypatch.setattr(
            search_page,
            "initialize_search_state",
            Mock(),
        )
        monkeypatch.setattr(
            search_page,
            "render_hero",
            Mock(),
        )
        monkeypatch.setattr(
            search_page,
            "render_search_form",
            Mock(
                return_value=(
                    True,
                    "finite volume methods",
                    20,
                )
            ),
        )
        monkeypatch.setattr(
            search_page,
            "perform_search",
            perform_search_mock,
        )
        monkeypatch.setattr(
            search_page,
            "render_search_workspace",
            workspace_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "caption",
            caption_mock,
        )

        search_page.render_search_page()

        perform_search_mock.assert_called_once_with(
            query="finite volume methods",
            result_k=20,
        )
        caption_mock.assert_called_once_with(
            'Showing results for: "finite volume methods"'
        )
        workspace_mock.assert_called_once_with(search_response)

    def test_renders_existing_search_response(
        self,
        search_page: ModuleType,
        search_response: SearchResponse,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render a previously stored search response."""
        workspace_mock = Mock()
        caption_mock = Mock()

        search_page.st.session_state.update(
            {
                search_page.SEARCH_RESPONSE_KEY: search_response,
                search_page.LAST_SEARCH_QUERY_KEY: "semantic search",
            }
        )

        monkeypatch.setattr(
            search_page,
            "initialize_search_state",
            Mock(),
        )
        monkeypatch.setattr(
            search_page,
            "render_hero",
            Mock(),
        )
        monkeypatch.setattr(
            search_page,
            "render_search_form",
            Mock(return_value=(False, "", 10)),
        )
        monkeypatch.setattr(
            search_page,
            "render_search_workspace",
            workspace_mock,
        )
        monkeypatch.setattr(
            search_page.st,
            "caption",
            caption_mock,
        )

        search_page.render_search_page()

        caption_mock.assert_called_once_with(
            'Showing results for: "semantic search"'
        )
        workspace_mock.assert_called_once_with(search_response)
