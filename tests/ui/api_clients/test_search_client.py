"""Tests for the paper-search API client."""

from types import ModuleType
from unittest.mock import Mock, call

import pytest
from pydantic import ValidationError

from research_paper_intelligence.api.schemas.search import SearchResponse
from research_paper_intelligence.ui.api_clients import search_client
from research_paper_intelligence.ui.api_clients.search_client import (
    search_papers,
)


@pytest.fixture
def client_module() -> ModuleType:
    """Return the search-client module under test."""
    return search_client


@pytest.fixture
def response_payload() -> dict[str, object]:
    """Create a valid search-response payload."""
    return {
        "total": 0,
        "time_elapsed": 0.125,
        "results": [],
    }


@pytest.fixture
def search_client_settings() -> Mock:
    """Create settings required by the search API client."""
    settings = Mock()
    settings.api_host = "127.0.0.1"
    settings.api_port = 8000
    settings.api_timeout_seconds = 30.0

    return settings


@pytest.fixture
def mock_settings(
    monkeypatch: pytest.MonkeyPatch,
    search_client_settings: Mock,
) -> Mock:
    """Replace the settings factory with a mock."""
    get_settings_mock = Mock(
        return_value=search_client_settings,
    )

    monkeypatch.setattr(
        search_client,
        "get_settings",
        get_settings_mock,
    )

    return get_settings_mock


class TestSearchPapers:
    """Tests for the search_papers function."""

    def test_returns_validated_search_response(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
    ) -> None:
        """Return the JSON payload as a SearchResponse model."""
        result = search_papers(
            query="finite volume methods",
            result_k=5,
        )

        assert isinstance(result, SearchResponse)
        assert result.total == 0
        assert result.time_elapsed == pytest.approx(0.125)
        assert result.results == []

    def test_loads_application_settings(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
    ) -> None:
        """Load application settings once for the request."""
        search_papers(
            query="finite volume methods",
            result_k=5,
        )

        mock_settings.assert_called_once_with()

    def test_creates_client_with_configured_api_values(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
    ) -> None:
        """Create the HTTP client with the configured URL and timeout."""
        search_papers(
            query="semantic search",
            result_k=10,
        )

        http_client_constructor.assert_called_once_with(
            base_url="http://127.0.0.1:8000",
            timeout=30.0,
        )

    @pytest.mark.parametrize(
        ("host", "port", "expected_base_url"),
        [
            (
                "127.0.0.1",
                8000,
                "http://127.0.0.1:8000",
            ),
            (
                "localhost",
                8080,
                "http://localhost:8080",
            ),
            (
                "api",
                9000,
                "http://api:9000",
            ),
        ],
    )
    def test_builds_base_url_from_settings(
        self,
        search_client_settings: Mock,
        mock_settings: Mock,
        http_client_constructor: Mock,
        host: str,
        port: int,
        expected_base_url: str,
    ) -> None:
        """Build the API base URL from the configured host and port."""
        search_client_settings.api_host = host
        search_client_settings.api_port = port

        search_papers(
            query="neural operators",
            result_k=5,
        )

        assert (
            http_client_constructor.call_args.kwargs["base_url"]
            == expected_base_url
        )

    def test_uses_configured_request_timeout(
        self,
        search_client_settings: Mock,
        mock_settings: Mock,
        http_client_constructor: Mock,
    ) -> None:
        """Pass the configured timeout to the HTTP client."""
        search_client_settings.api_timeout_seconds = 45.0

        search_papers(
            query="machine learning",
            result_k=5,
        )

        assert http_client_constructor.call_args.kwargs["timeout"] == 45.0

    def test_sends_query_and_result_count_to_search_endpoint(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_client: Mock,
    ) -> None:
        """Send the query and result count to the search endpoint."""
        search_papers(
            query="finite volume elastoplasticity",
            result_k=7,
        )

        http_client.get.assert_called_once_with(
            "api/v1/search/?query=finite volume elastoplasticity&result_k=7"
        )

    @pytest.mark.parametrize(
        ("query", "result_k", "expected_url"),
        [
            (
                "finite volume methods",
                1,
                "api/v1/search/?query=finite volume methods&result_k=1",
            ),
            (
                "semantic search",
                5,
                "api/v1/search/?query=semantic search&result_k=5",
            ),
            (
                "neural operators",
                10,
                "api/v1/search/?query=neural operators&result_k=10",
            ),
        ],
    )
    def test_builds_search_request_url(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_client: Mock,
        query: str,
        result_k: int,
        expected_url: str,
    ) -> None:
        """Build the request URL from the supplied search arguments."""
        search_papers(
            query=query,
            result_k=result_k,
        )

        http_client.get.assert_called_once_with(expected_url)

    def test_checks_status_before_reading_response_json(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_response: Mock,
    ) -> None:
        """Check the response status before reading the JSON payload."""
        workflow = Mock()

        workflow.attach_mock(
            http_response.raise_for_status,
            "raise_for_status",
        )
        workflow.attach_mock(
            http_response.json,
            "json",
        )

        search_papers(
            query="machine learning",
            result_k=5,
        )

        assert workflow.mock_calls == [
            call.raise_for_status(),
            call.json(),
        ]

    def test_uses_http_client_as_context_manager(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_client_context: Mock,
    ) -> None:
        """Enter and exit the HTTP client context manager."""
        search_papers(
            query="semantic retrieval",
            result_k=5,
        )

        http_client_context.__enter__.assert_called_once_with()
        http_client_context.__exit__.assert_called_once()

    def test_propagates_request_error(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_client: Mock,
    ) -> None:
        """Propagate an error raised while sending the request."""
        http_client.get.side_effect = RuntimeError(
            "The search API is unavailable."
        )

        with pytest.raises(
            RuntimeError,
            match="The search API is unavailable",
        ):
            search_papers(
                query="neural operators",
                result_k=5,
            )

    def test_propagates_status_check_error(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_response: Mock,
    ) -> None:
        """Propagate an error raised for an unsuccessful response."""
        http_response.raise_for_status.side_effect = RuntimeError(
            "Search request failed."
        )

        with pytest.raises(
            RuntimeError,
            match="Search request failed",
        ):
            search_papers(
                query="finite volume methods",
                result_k=5,
            )

        http_response.json.assert_not_called()

    def test_rejects_invalid_response_payload(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_response: Mock,
    ) -> None:
        """Raise a validation error for an invalid response payload."""
        http_response.json.return_value = {
            "total": "invalid-total",
            "time_elapsed": 0.125,
            "results": [],
        }

        with pytest.raises(ValidationError):
            search_papers(
                query="semantic search",
                result_k=5,
            )

    def test_propagates_json_decoding_error(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_response: Mock,
    ) -> None:
        """Propagate an error raised while decoding response JSON."""
        http_response.json.side_effect = ValueError("Invalid JSON response.")

        with pytest.raises(
            ValueError,
            match="Invalid JSON response",
        ):
            search_papers(
                query="semantic search",
                result_k=5,
            )

    def test_propagates_settings_loading_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Propagate an error raised while loading application settings."""
        client_constructor = Mock()

        monkeypatch.setattr(
            search_client,
            "get_settings",
            Mock(side_effect=RuntimeError("Settings could not be loaded.")),
        )
        monkeypatch.setattr(
            search_client.httpx2,
            "Client",
            client_constructor,
        )

        with pytest.raises(
            RuntimeError,
            match="Settings could not be loaded",
        ):
            search_papers(
                query="semantic search",
                result_k=5,
            )

        client_constructor.assert_not_called()
