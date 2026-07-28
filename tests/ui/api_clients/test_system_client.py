"""Tests for the system-information API client."""

from collections.abc import Generator
from types import ModuleType
from unittest.mock import Mock, call

import pytest
from pydantic import ValidationError

from research_paper_intelligence.api.schemas.system import (
    SystemInfoResponse,
)
from research_paper_intelligence.ui.api_clients import system_client
from research_paper_intelligence.ui.api_clients.system_client import (
    get_system_info,
)


@pytest.fixture(autouse=True)
def clear_streamlit_cache() -> Generator[None, None, None]:
    """Clear cached system information before and after each test."""
    system_client.st.cache_data.clear()

    yield

    system_client.st.cache_data.clear()


@pytest.fixture
def client_module() -> ModuleType:
    """Return the system-client module under test."""
    return system_client


@pytest.fixture
def response_payload() -> dict[str, object]:
    """Create a valid system-information response payload."""
    return {
        "status": "ready",
        "paper_count": 130_000,
        "embedding_model": ("sentence-transformers/all-MiniLM-L6-v2"),
        "retrieval_strategy": "Hybrid",
        "ranking_components": [
            "Semantic similarity",
            "TF-IDF similarity",
            "Keyword overlap",
            "Publication recency",
        ],
        "faiss_index_type": "IndexFlatIP",
        "faiss_index_size": 130_000,
        "tfidf_document_count": 130_000,
        "tfidf_vocabulary_size": 50_000,
        "api_version": "1.0.0",
    }


@pytest.fixture
def system_client_settings() -> Mock:
    """Create settings required by the system-information client."""
    settings = Mock()
    settings.api_host = "127.0.0.1"
    settings.api_port = 8000
    settings.api_timeout_seconds = 30.0

    return settings


@pytest.fixture
def mock_settings(
    monkeypatch: pytest.MonkeyPatch,
    system_client_settings: Mock,
) -> Mock:
    """Replace the settings factory with a mock."""
    get_settings_mock = Mock(
        return_value=system_client_settings,
    )

    monkeypatch.setattr(
        system_client,
        "get_settings",
        get_settings_mock,
    )

    return get_settings_mock


class TestGetSystemInfo:
    """Tests for the get_system_info function."""

    def test_returns_validated_system_information(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
    ) -> None:
        """Return the API response as a SystemInfoResponse model."""
        result = get_system_info()

        assert isinstance(result, SystemInfoResponse)
        assert result.status == "ready"
        assert result.paper_count == 130_000
        assert result.embedding_model == (
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        assert result.retrieval_strategy == "Hybrid"
        assert result.ranking_components == [
            "Semantic similarity",
            "TF-IDF similarity",
            "Keyword overlap",
            "Publication recency",
        ]
        assert result.faiss_index_type == "IndexFlatIP"
        assert result.faiss_index_size == 130_000
        assert result.tfidf_document_count == 130_000
        assert result.tfidf_vocabulary_size == 50_000
        assert result.api_version == "1.0.0"

    def test_loads_application_settings(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
    ) -> None:
        """Load application settings when requesting system information."""
        get_system_info()

        mock_settings.assert_called_once_with()

    def test_creates_client_with_configured_api_values(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
    ) -> None:
        """Create the HTTP client with the configured URL and timeout."""
        get_system_info()

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
        system_client_settings: Mock,
        mock_settings: Mock,
        http_client_constructor: Mock,
        host: str,
        port: int,
        expected_base_url: str,
    ) -> None:
        """Build the API base URL from the configured host and port."""
        system_client_settings.api_host = host
        system_client_settings.api_port = port

        get_system_info()

        assert (
            http_client_constructor.call_args.kwargs["base_url"]
            == expected_base_url
        )

    def test_uses_configured_request_timeout(
        self,
        system_client_settings: Mock,
        mock_settings: Mock,
        http_client_constructor: Mock,
    ) -> None:
        """Pass the configured timeout to the HTTP client."""
        system_client_settings.api_timeout_seconds = 45.0

        get_system_info()

        assert http_client_constructor.call_args.kwargs["timeout"] == 45.0

    def test_requests_system_information_endpoint(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_client: Mock,
    ) -> None:
        """Send a GET request to the system-information endpoint."""
        get_system_info()

        http_client.get.assert_called_once_with("/api/v1/system/")

    def test_checks_status_before_reading_response_json(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_response: Mock,
    ) -> None:
        """Check the HTTP status before reading the JSON payload."""
        workflow = Mock()

        workflow.attach_mock(
            http_response.raise_for_status,
            "raise_for_status",
        )
        workflow.attach_mock(
            http_response.json,
            "json",
        )

        get_system_info()

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
        get_system_info()

        http_client_context.__enter__.assert_called_once_with()
        http_client_context.__exit__.assert_called_once()

    def test_caches_system_information(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_client: Mock,
    ) -> None:
        """Reuse the cached response instead of repeating the API call."""
        first_result = get_system_info()
        second_result = get_system_info()

        assert first_result == second_result
        mock_settings.assert_called_once_with()
        http_client_constructor.assert_called_once()
        http_client.get.assert_called_once_with("/api/v1/system/")

    def test_requests_information_again_after_cache_clear(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_client: Mock,
    ) -> None:
        """Repeat the API request after clearing Streamlit's cache."""
        get_system_info()

        system_client.st.cache_data.clear()

        get_system_info()

        assert mock_settings.call_count == 2
        assert http_client_constructor.call_count == 2
        assert http_client.get.call_count == 2

    def test_propagates_request_error(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_client: Mock,
    ) -> None:
        """Propagate errors raised while sending the HTTP request."""
        http_client.get.side_effect = RuntimeError(
            "The system API is unavailable."
        )

        with pytest.raises(
            RuntimeError,
            match="The system API is unavailable",
        ):
            get_system_info()

    def test_propagates_status_check_error(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_response: Mock,
    ) -> None:
        """Propagate errors raised for an unsuccessful HTTP response."""
        http_response.raise_for_status.side_effect = RuntimeError(
            "System-information request failed."
        )

        with pytest.raises(
            RuntimeError,
            match="System-information request failed",
        ):
            get_system_info()

        http_response.json.assert_not_called()

    def test_rejects_invalid_response_payload(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_response: Mock,
    ) -> None:
        """Raise a validation error for an invalid response payload."""
        http_response.json.return_value = {
            "status": "ready",
            "paper_count": "invalid-count",
            "embedding_model": "example-model",
            "retrieval_strategy": "Hybrid",
            "ranking_components": [],
            "faiss_index_type": "IndexFlatIP",
            "faiss_index_size": 100,
            "tfidf_document_count": 100,
            "tfidf_vocabulary_size": 50,
            "api_version": "1.0.0",
        }

        with pytest.raises(ValidationError):
            get_system_info()

    def test_propagates_json_decoding_error(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_response: Mock,
    ) -> None:
        """Propagate errors raised while decoding response JSON."""
        http_response.json.side_effect = ValueError("Invalid JSON response.")

        with pytest.raises(
            ValueError,
            match="Invalid JSON response",
        ):
            get_system_info()

    def test_propagates_settings_loading_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Propagate errors raised while loading application settings."""
        client_constructor = Mock()

        monkeypatch.setattr(
            system_client,
            "get_settings",
            Mock(side_effect=RuntimeError("Settings could not be loaded.")),
        )
        monkeypatch.setattr(
            system_client.httpx2,
            "Client",
            client_constructor,
        )

        with pytest.raises(
            RuntimeError,
            match="Settings could not be loaded",
        ):
            get_system_info()

        client_constructor.assert_not_called()
