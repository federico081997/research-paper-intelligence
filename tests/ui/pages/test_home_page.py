"""Tests for the Research Paper Intelligence home page."""

import importlib
import sys
from collections.abc import Generator
from types import ModuleType
from unittest.mock import MagicMock, Mock, call

import pytest
import streamlit as st

from research_paper_intelligence.api.schemas.system import (
    SystemInfoResponse,
)

HOME_PAGE_MODULE = "research_paper_intelligence.ui.pages.home"
SYSTEM_CLIENT_MODULE = (
    "research_paper_intelligence.ui.api_clients.system_client"
)
CARDS_MODULE = "research_paper_intelligence.ui.components.cards"
NAVIGATION_MODULE = "research_paper_intelligence.ui.navigation"


@pytest.fixture
def system_info() -> SystemInfoResponse:
    """Create representative information about the loaded system."""
    return SystemInfoResponse(
        status="ready",
        paper_count=130_000,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        retrieval_strategy="Hybrid",
        ranking_components=[
            "Semantic similarity",
            "TF-IDF similarity",
            "Keyword overlap",
            "Publication recency",
        ],
        faiss_index_type="IndexFlatIP",
        faiss_index_size=130_000,
        tfidf_document_count=130_000,
        tfidf_vocabulary_size=50_000,
        api_version="1.0.0",
    )


@pytest.fixture
def context_manager() -> MagicMock:
    """Create a mocked context manager."""
    context = MagicMock()
    context.__enter__.return_value = context
    context.__exit__.return_value = False

    return context


@pytest.fixture
def home_page(
    monkeypatch: pytest.MonkeyPatch,
    system_info: SystemInfoResponse,
) -> Generator[ModuleType, None, None]:
    """Import the home page with its import-time dependencies mocked."""
    get_system_info_mock = Mock(return_value=system_info)
    render_feature_card_mock = Mock()

    search_page = Mock(name="search_page")
    assistant_page = Mock(name="assistant_page")

    system_client_module = ModuleType(SYSTEM_CLIENT_MODULE)
    system_client_module.get_system_info = get_system_info_mock

    cards_module = ModuleType(CARDS_MODULE)
    cards_module.render_feature_card = render_feature_card_mock

    navigation_module = ModuleType(NAVIGATION_MODULE)
    navigation_module.SEARCH_PAGE = search_page
    navigation_module.ASSISTANT_PAGE = assistant_page

    monkeypatch.setitem(
        sys.modules,
        SYSTEM_CLIENT_MODULE,
        system_client_module,
    )
    monkeypatch.setitem(
        sys.modules,
        CARDS_MODULE,
        cards_module,
    )
    monkeypatch.setitem(
        sys.modules,
        NAVIGATION_MODULE,
        navigation_module,
    )

    metric_columns = [
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]
    feature_columns = [
        MagicMock(),
        MagicMock(),
    ]

    def columns_side_effect(
        specification: object,
        *args: object,
        **kwargs: object,
    ) -> list[MagicMock]:
        """Return suitable columns for the requested layout."""
        del args, kwargs

        if specification == [1, 1, 1, 1]:
            return metric_columns

        return feature_columns

    monkeypatch.setattr(st, "title", Mock())
    monkeypatch.setattr(st, "markdown", Mock())
    monkeypatch.setattr(st, "subheader", Mock())
    monkeypatch.setattr(st, "divider", Mock())
    monkeypatch.setattr(st, "metric", Mock())
    monkeypatch.setattr(st, "success", Mock())
    monkeypatch.setattr(st, "warning", Mock())
    monkeypatch.setattr(st, "error", Mock())
    monkeypatch.setattr(
        st,
        "columns",
        Mock(side_effect=columns_side_effect),
    )
    monkeypatch.setattr(
        st,
        "expander",
        Mock(return_value=MagicMock()),
    )

    sys.modules.pop(HOME_PAGE_MODULE, None)

    module = importlib.import_module(HOME_PAGE_MODULE)

    # Ignore calls caused by render_home_page() at module import.
    for function_name in (
        "title",
        "markdown",
        "subheader",
        "divider",
        "metric",
        "success",
        "warning",
        "error",
        "columns",
        "expander",
    ):
        mocked_function = getattr(st, function_name)
        mocked_function.reset_mock()

    get_system_info_mock.reset_mock()
    render_feature_card_mock.reset_mock()

    yield module

    sys.modules.pop(HOME_PAGE_MODULE, None)


class TestRenderHero:
    """Tests for rendering the home-page hero."""

    def test_renders_application_title_and_description(
        self,
        home_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render the application name and introductory description."""
        title_mock = Mock()
        markdown_mock = Mock()

        monkeypatch.setattr(
            home_page.st,
            "title",
            title_mock,
        )
        monkeypatch.setattr(
            home_page.st,
            "markdown",
            markdown_mock,
        )

        home_page.render_hero()

        title_mock.assert_called_once_with("Research Paper Intelligence")

        description = markdown_mock.call_args.args[0]

        assert "Discover and rank research papers" in description
        assert "hybrid retrieval" in description
        assert "evidence-grounded AI workflows" in description


class TestRenderSystemMetrics:
    """Tests for rendering the primary system metrics."""

    def test_creates_four_equal_metric_columns(
        self,
        home_page: ModuleType,
        system_info: SystemInfoResponse,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Create four equally sized columns for the metrics."""
        columns = [MagicMock() for _ in range(4)]
        columns_mock = Mock(return_value=columns)

        monkeypatch.setattr(
            home_page.st,
            "columns",
            columns_mock,
        )
        monkeypatch.setattr(
            home_page.st,
            "metric",
            Mock(),
        )

        home_page.render_system_metrics(system_info)

        columns_mock.assert_called_once_with([1, 1, 1, 1])

        for column in columns:
            column.__enter__.assert_called_once_with()
            column.__exit__.assert_called_once()

    def test_renders_formatted_system_metrics(
        self,
        home_page: ModuleType,
        system_info: SystemInfoResponse,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render paper, retrieval, index, and API values."""
        metric_mock = Mock()

        monkeypatch.setattr(
            home_page.st,
            "columns",
            Mock(
                return_value=[
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                ]
            ),
        )
        monkeypatch.setattr(
            home_page.st,
            "metric",
            metric_mock,
        )

        home_page.render_system_metrics(system_info)

        assert metric_mock.call_args_list == [
            call(
                label="Available papers",
                value="130,000",
            ),
            call(
                label="Retrieval method",
                value="Hybrid",
            ),
            call(
                label="Indexed papers",
                value="130,000",
            ),
            call(
                label="API version",
                value="1.0.0",
            ),
        ]


class TestRenderRetrievalDetails:
    """Tests for rendering retrieval configuration details."""

    def test_renders_details_inside_expander(
        self,
        home_page: ModuleType,
        system_info: SystemInfoResponse,
        monkeypatch: pytest.MonkeyPatch,
        context_manager: MagicMock,
    ) -> None:
        """Render retrieval details inside a labelled expander."""
        expander_mock = Mock(return_value=context_manager)

        monkeypatch.setattr(
            home_page.st,
            "expander",
            expander_mock,
        )
        monkeypatch.setattr(
            home_page.st,
            "columns",
            Mock(
                return_value=[
                    MagicMock(),
                    MagicMock(),
                ]
            ),
        )
        monkeypatch.setattr(
            home_page.st,
            "markdown",
            Mock(),
        )

        home_page.render_retrieval_details(system_info)

        expander_mock.assert_called_once_with("**Retrieval configuration**")
        context_manager.__enter__.assert_called_once_with()
        context_manager.__exit__.assert_called_once()

    def test_creates_component_and_resource_columns(
        self,
        home_page: ModuleType,
        system_info: SystemInfoResponse,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Create separate columns for ranking and resource details."""
        columns = [MagicMock(), MagicMock()]
        columns_mock = Mock(return_value=columns)

        monkeypatch.setattr(
            home_page.st,
            "expander",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            home_page.st,
            "columns",
            columns_mock,
        )
        monkeypatch.setattr(
            home_page.st,
            "markdown",
            Mock(),
        )

        home_page.render_retrieval_details(system_info)

        columns_mock.assert_called_once_with(2)

        for column in columns:
            column.__enter__.assert_called_once_with()
            column.__exit__.assert_called_once()

    def test_renders_all_ranking_components(
        self,
        home_page: ModuleType,
        system_info: SystemInfoResponse,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render every hybrid-ranking component."""
        markdown_mock = Mock()

        monkeypatch.setattr(
            home_page.st,
            "expander",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            home_page.st,
            "columns",
            Mock(
                return_value=[
                    MagicMock(),
                    MagicMock(),
                ]
            ),
        )
        monkeypatch.setattr(
            home_page.st,
            "markdown",
            markdown_mock,
        )

        home_page.render_retrieval_details(system_info)

        assert markdown_mock.call_args_list[:5] == [
            call("#### Hybrid ranking components"),
            call("- Semantic similarity"),
            call("- TF-IDF similarity"),
            call("- Keyword overlap"),
            call("- Publication recency"),
        ]

    def test_renders_loaded_resource_information(
        self,
        home_page: ModuleType,
        system_info: SystemInfoResponse,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render the loaded embedding, FAISS, and TF-IDF resources."""
        markdown_mock = Mock()

        monkeypatch.setattr(
            home_page.st,
            "expander",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            home_page.st,
            "columns",
            Mock(
                return_value=[
                    MagicMock(),
                    MagicMock(),
                ]
            ),
        )
        monkeypatch.setattr(
            home_page.st,
            "markdown",
            markdown_mock,
        )

        home_page.render_retrieval_details(system_info)

        markdown_mock.assert_any_call("#### Loaded resources")

        resources_markdown = markdown_mock.call_args.args[0]

        assert "sentence-transformers/all-MiniLM-L6-v2" in resources_markdown
        assert "IndexFlatIP" in resources_markdown
        assert "130,000" in resources_markdown
        assert "50,000" in resources_markdown


class TestRenderSystemStatus:
    """Tests for loading and rendering the backend status."""

    def test_renders_ready_system_status(
        self,
        home_page: ModuleType,
        system_info: SystemInfoResponse,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render a success message when all resources are ready."""
        success_mock = Mock()
        metrics_mock = Mock()
        details_mock = Mock()

        monkeypatch.setattr(
            home_page,
            "get_system_info",
            Mock(return_value=system_info),
        )
        monkeypatch.setattr(
            home_page.st,
            "subheader",
            Mock(),
        )
        monkeypatch.setattr(
            home_page.st,
            "success",
            success_mock,
        )
        monkeypatch.setattr(
            home_page,
            "render_system_metrics",
            metrics_mock,
        )
        monkeypatch.setattr(
            home_page,
            "render_retrieval_details",
            details_mock,
        )

        home_page.render_system_status()

        success_mock.assert_called_once_with(
            "The paper data, embedding model, FAISS index, "
            "and TF-IDF resources are ready."
        )
        metrics_mock.assert_called_once_with(system_info)
        details_mock.assert_called_once_with(system_info)

    def test_renders_warning_when_resources_are_not_ready(
        self,
        home_page: ModuleType,
        system_info: SystemInfoResponse,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render a warning when one or more resources are unavailable."""
        unavailable_info = system_info.model_copy(update={"status": "loading"})
        warning_mock = Mock()

        monkeypatch.setattr(
            home_page,
            "get_system_info",
            Mock(return_value=unavailable_info),
        )
        monkeypatch.setattr(
            home_page.st,
            "subheader",
            Mock(),
        )
        monkeypatch.setattr(
            home_page.st,
            "warning",
            warning_mock,
        )
        monkeypatch.setattr(
            home_page,
            "render_system_metrics",
            Mock(),
        )
        monkeypatch.setattr(
            home_page,
            "render_retrieval_details",
            Mock(),
        )

        home_page.render_system_status()

        warning_mock.assert_called_once_with(
            "The backend is running, but one or more search "
            "resources are not ready."
        )

    def test_handles_backend_connection_error(
        self,
        home_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render an error when the backend cannot be reached."""

        class TestConnectError(Exception):
            """Test connection error."""

        error_mock = Mock()
        metrics_mock = Mock()
        details_mock = Mock()

        monkeypatch.setattr(
            home_page.httpx2,
            "ConnectError",
            TestConnectError,
        )
        monkeypatch.setattr(
            home_page,
            "get_system_info",
            Mock(side_effect=TestConnectError()),
        )
        monkeypatch.setattr(
            home_page.st,
            "subheader",
            Mock(),
        )
        monkeypatch.setattr(
            home_page.st,
            "error",
            error_mock,
        )
        monkeypatch.setattr(
            home_page,
            "render_system_metrics",
            metrics_mock,
        )
        monkeypatch.setattr(
            home_page,
            "render_retrieval_details",
            details_mock,
        )

        home_page.render_system_status()

        error_mock.assert_called_once_with(
            "The FastAPI backend is unavailable. "
            "Start the API server and refresh this page."
        )
        metrics_mock.assert_not_called()
        details_mock.assert_not_called()

    def test_handles_backend_timeout(
        self,
        home_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render a warning when the backend request times out."""

        class TestTimeoutError(Exception):
            """Test timeout error."""

        warning_mock = Mock()
        metrics_mock = Mock()
        details_mock = Mock()

        monkeypatch.setattr(
            home_page.httpx2,
            "TimeoutException",
            TestTimeoutError,
        )
        monkeypatch.setattr(
            home_page,
            "get_system_info",
            Mock(side_effect=TestTimeoutError()),
        )
        monkeypatch.setattr(
            home_page.st,
            "subheader",
            Mock(),
        )
        monkeypatch.setattr(
            home_page.st,
            "warning",
            warning_mock,
        )
        monkeypatch.setattr(
            home_page,
            "render_system_metrics",
            metrics_mock,
        )
        monkeypatch.setattr(
            home_page,
            "render_retrieval_details",
            details_mock,
        )

        home_page.render_system_status()

        warning_mock.assert_called_once_with(
            "The backend did not respond in time. "
            "It may still be loading the search resources."
        )
        metrics_mock.assert_not_called()
        details_mock.assert_not_called()

    def test_handles_http_status_error(
        self,
        home_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render the status code returned by the backend."""

        class TestHTTPStatusError(Exception):
            """Test HTTP status error."""

            def __init__(self, status_code: int) -> None:
                super().__init__("HTTP status error")
                self.response = Mock(status_code=status_code)

        error_mock = Mock()
        metrics_mock = Mock()
        details_mock = Mock()

        monkeypatch.setattr(
            home_page.httpx2,
            "HTTPStatusError",
            TestHTTPStatusError,
        )
        monkeypatch.setattr(
            home_page,
            "get_system_info",
            Mock(side_effect=TestHTTPStatusError(503)),
        )
        monkeypatch.setattr(
            home_page.st,
            "subheader",
            Mock(),
        )
        monkeypatch.setattr(
            home_page.st,
            "error",
            error_mock,
        )
        monkeypatch.setattr(
            home_page,
            "render_system_metrics",
            metrics_mock,
        )
        monkeypatch.setattr(
            home_page,
            "render_retrieval_details",
            details_mock,
        )

        home_page.render_system_status()

        error_mock.assert_called_once_with(
            "The backend returned an error with status 503."
        )
        metrics_mock.assert_not_called()
        details_mock.assert_not_called()

    def test_handles_invalid_system_information(
        self,
        home_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render an error when the response fails validation."""

        class TestValidationError(Exception):
            """Test response-validation error."""

        error_mock = Mock()
        metrics_mock = Mock()
        details_mock = Mock()

        monkeypatch.setattr(
            home_page,
            "ValidationError",
            TestValidationError,
        )
        monkeypatch.setattr(
            home_page,
            "get_system_info",
            Mock(side_effect=TestValidationError()),
        )
        monkeypatch.setattr(
            home_page.st,
            "subheader",
            Mock(),
        )
        monkeypatch.setattr(
            home_page.st,
            "error",
            error_mock,
        )
        monkeypatch.setattr(
            home_page,
            "render_system_metrics",
            metrics_mock,
        )
        monkeypatch.setattr(
            home_page,
            "render_retrieval_details",
            details_mock,
        )

        home_page.render_system_status()

        error_mock.assert_called_once_with(
            "The backend returned system information in an unexpected format."
        )
        metrics_mock.assert_not_called()
        details_mock.assert_not_called()

    def test_always_renders_system_status_heading(
        self,
        home_page: ModuleType,
        system_info: SystemInfoResponse,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render the section heading before loading backend data."""
        subheader_mock = Mock()

        monkeypatch.setattr(
            home_page,
            "get_system_info",
            Mock(return_value=system_info),
        )
        monkeypatch.setattr(
            home_page.st,
            "subheader",
            subheader_mock,
        )
        monkeypatch.setattr(
            home_page.st,
            "success",
            Mock(),
        )
        monkeypatch.setattr(
            home_page,
            "render_system_metrics",
            Mock(),
        )
        monkeypatch.setattr(
            home_page,
            "render_retrieval_details",
            Mock(),
        )

        home_page.render_system_status()

        subheader_mock.assert_called_once_with("System status")


class TestRenderCapabilities:
    """Tests for rendering application capabilities."""

    def test_renders_capabilities_heading_and_columns(
        self,
        home_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render the capability section using two columns."""
        columns = [MagicMock(), MagicMock()]
        subheader_mock = Mock()
        columns_mock = Mock(return_value=columns)

        monkeypatch.setattr(
            home_page.st,
            "subheader",
            subheader_mock,
        )
        monkeypatch.setattr(
            home_page.st,
            "columns",
            columns_mock,
        )
        monkeypatch.setattr(
            home_page,
            "render_feature_card",
            Mock(),
        )

        home_page.render_capabilities()

        subheader_mock.assert_called_once_with("What you can do")
        columns_mock.assert_called_once_with(2)

        for column in columns:
            column.__enter__.assert_called_once_with()
            column.__exit__.assert_called_once()

    def test_renders_search_and_assistant_feature_cards(
        self,
        home_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render cards for search and assistant capabilities."""
        feature_card_mock = Mock()

        monkeypatch.setattr(
            home_page.st,
            "subheader",
            Mock(),
        )
        monkeypatch.setattr(
            home_page.st,
            "columns",
            Mock(
                return_value=[
                    MagicMock(),
                    MagicMock(),
                ]
            ),
        )
        monkeypatch.setattr(
            home_page,
            "render_feature_card",
            feature_card_mock,
        )

        home_page.render_capabilities()

        assert feature_card_mock.call_args_list == [
            call(
                key="hybrid-search",
                title="Paper search",
                icon=":material/search:",
                description=(
                    "Retrieve and rank papers using semantic similarity, "
                    "TF-IDF, keyword overlap, and publication recency."
                ),
                status="Available",
                page=home_page.SEARCH_PAGE,
                button_label="Search papers",
            ),
            call(
                key="research-assistant",
                title="Research assistant",
                icon=":material/smart_toy:",
                description=(
                    "Ask research questions and receive answers grounded "
                    "in retrieved papers with traceable citations."
                ),
                status="Planned",
                page=home_page.ASSISTANT_PAGE,
                button_label="Ask questions",
            ),
        ]


class TestRenderScopeAndLimitations:
    """Tests for rendering scope and limitation information."""

    def test_renders_limitations_inside_expander(
        self,
        home_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        context_manager: MagicMock,
    ) -> None:
        """Render application limitations in an expander."""
        expander_mock = Mock(return_value=context_manager)
        markdown_mock = Mock()

        monkeypatch.setattr(
            home_page.st,
            "expander",
            expander_mock,
        )
        monkeypatch.setattr(
            home_page.st,
            "markdown",
            markdown_mock,
        )

        home_page.render_scope_and_limitations()

        expander_mock.assert_called_once_with("**Scope and limitations**")

        limitations = markdown_mock.call_args.args[0]

        assert "limited to papers contained" in limitations
        assert "not scientific correctness" in limitations
        assert "depends on the search query" in limitations
        assert "verified against" in limitations

        context_manager.__enter__.assert_called_once_with()
        context_manager.__exit__.assert_called_once()


class TestRenderHomePage:
    """Tests for rendering the complete home page."""

    def test_renders_all_sections_with_dividers(
        self,
        home_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render all home-page sections in the expected order."""
        workflow = Mock()

        hero_mock = Mock()
        system_status_mock = Mock()
        capabilities_mock = Mock()
        limitations_mock = Mock()
        divider_mock = Mock()

        workflow.attach_mock(hero_mock, "hero")
        workflow.attach_mock(divider_mock, "divider")
        workflow.attach_mock(
            system_status_mock,
            "system_status",
        )
        workflow.attach_mock(
            capabilities_mock,
            "capabilities",
        )
        workflow.attach_mock(
            limitations_mock,
            "limitations",
        )

        monkeypatch.setattr(
            home_page,
            "render_hero",
            hero_mock,
        )
        monkeypatch.setattr(
            home_page,
            "render_system_status",
            system_status_mock,
        )
        monkeypatch.setattr(
            home_page,
            "render_capabilities",
            capabilities_mock,
        )
        monkeypatch.setattr(
            home_page,
            "render_scope_and_limitations",
            limitations_mock,
        )
        monkeypatch.setattr(
            home_page.st,
            "divider",
            divider_mock,
        )

        home_page.render_home_page()

        assert workflow.mock_calls == [
            call.hero(),
            call.divider(),
            call.system_status(),
            call.divider(),
            call.capabilities(),
            call.divider(),
            call.limitations(),
        ]
