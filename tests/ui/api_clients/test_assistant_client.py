"""Tests for the assistant API client."""

from types import ModuleType
from unittest.mock import Mock, call
from uuid import uuid4

import pytest
from pydantic import ValidationError

from research_paper_intelligence.api.schemas.assistant import (
    AssistantResponse,
)
from research_paper_intelligence.ui.api_clients import assistant_client
from research_paper_intelligence.ui.api_clients.assistant_client import chat


@pytest.fixture
def client_module() -> ModuleType:
    """Return the assistant-client module under test."""
    return assistant_client


@pytest.fixture
def response_payload() -> dict[str, object]:
    """Create a valid assistant-response payload."""
    return {
        "response": "Finite volume methods conserve fluxes.",
        "thread_id": str(uuid4()),
    }


@pytest.fixture
def assistant_client_settings() -> Mock:
    """Create settings required by the assistant API client."""
    settings = Mock()
    settings.api_host = "127.0.0.1"
    settings.api_port = 8000
    settings.api_assistant_timeout_seconds = 60.0

    return settings


@pytest.fixture
def mock_settings(
    monkeypatch: pytest.MonkeyPatch,
    assistant_client_settings: Mock,
) -> Mock:
    """Replace the settings factory with a mock."""
    get_settings_mock = Mock(
        return_value=assistant_client_settings,
    )

    monkeypatch.setattr(
        assistant_client,
        "get_settings",
        get_settings_mock,
    )

    return get_settings_mock


class TestChat:
    """Tests for the chat function."""

    def test_returns_validated_assistant_response(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_response: Mock,
    ) -> None:
        """Return the JSON payload as an AssistantResponse model."""
        thread_id = uuid4()
        http_response.json.return_value = {
            "response": "Finite volume methods conserve fluxes.",
            "thread_id": str(thread_id),
        }

        result = chat(
            user_query="Explain finite volume methods.",
            thread_id=str(thread_id),
        )

        assert isinstance(result, AssistantResponse)
        assert result.response == ("Finite volume methods conserve fluxes.")
        assert result.thread_id == thread_id

    def test_loads_application_settings(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
    ) -> None:
        """Load application settings once for the request."""
        chat(
            user_query="Explain semantic search.",
            thread_id=str(uuid4()),
        )

        mock_settings.assert_called_once_with()

    def test_creates_client_with_configured_api_values(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
    ) -> None:
        """Create the HTTP client with the configured URL and timeout."""
        chat(
            user_query="Explain semantic search.",
            thread_id=str(uuid4()),
        )

        http_client_constructor.assert_called_once_with(
            base_url="http://127.0.0.1:8000",
            timeout=60.0,
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
        assistant_client_settings: Mock,
        mock_settings: Mock,
        http_client_constructor: Mock,
        host: str,
        port: int,
        expected_base_url: str,
    ) -> None:
        """Build the API base URL from the configured host and port."""
        assistant_client_settings.api_host = host
        assistant_client_settings.api_port = port

        chat(
            user_query="Explain neural operators.",
            thread_id=str(uuid4()),
        )

        assert (
            http_client_constructor.call_args.kwargs["base_url"]
            == expected_base_url
        )

    def test_uses_configured_assistant_timeout(
        self,
        assistant_client_settings: Mock,
        mock_settings: Mock,
        http_client_constructor: Mock,
    ) -> None:
        """Pass the configured assistant timeout to the HTTP client."""
        assistant_client_settings.api_assistant_timeout_seconds = 90.0

        chat(
            user_query="Explain neural operators.",
            thread_id=str(uuid4()),
        )

        assert http_client_constructor.call_args.kwargs["timeout"] == 90.0

    def test_posts_query_and_thread_id_to_assistant_endpoint(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_client: Mock,
    ) -> None:
        """Send the user query and thread ID to the assistant endpoint."""
        thread_id = str(uuid4())

        chat(
            user_query="Find papers about elastoplasticity.",
            thread_id=thread_id,
        )

        http_client.post.assert_called_once_with(
            "/api/v1/assistant/chat",
            json={
                "user_query": ("Find papers about elastoplasticity."),
                "thread_id": thread_id,
            },
        )

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

        chat(
            user_query="Explain machine learning.",
            thread_id=str(uuid4()),
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
        chat(
            user_query="Explain semantic retrieval.",
            thread_id=str(uuid4()),
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
        http_client.post.side_effect = RuntimeError(
            "The assistant API is unavailable."
        )

        with pytest.raises(
            RuntimeError,
            match="The assistant API is unavailable",
        ):
            chat(
                user_query="Explain neural operators.",
                thread_id=str(uuid4()),
            )

    def test_propagates_status_check_error(
        self,
        mock_settings: Mock,
        http_client_constructor: Mock,
        http_response: Mock,
    ) -> None:
        """Propagate an error raised for an unsuccessful response."""
        http_response.raise_for_status.side_effect = RuntimeError(
            "Assistant request failed."
        )

        with pytest.raises(
            RuntimeError,
            match="Assistant request failed",
        ):
            chat(
                user_query="Explain finite volume methods.",
                thread_id=str(uuid4()),
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
            "response": "Generated response.",
            "thread_id": "not-a-valid-uuid",
        }

        with pytest.raises(ValidationError):
            chat(
                user_query="Explain semantic search.",
                thread_id=str(uuid4()),
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
            chat(
                user_query="Explain semantic search.",
                thread_id=str(uuid4()),
            )

    def test_propagates_settings_loading_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Propagate an error raised while loading application settings."""
        client_constructor = Mock()

        monkeypatch.setattr(
            assistant_client,
            "get_settings",
            Mock(side_effect=RuntimeError("Settings could not be loaded.")),
        )
        monkeypatch.setattr(
            assistant_client.httpx2,
            "Client",
            client_constructor,
        )

        with pytest.raises(
            RuntimeError,
            match="Settings could not be loaded",
        ):
            chat(
                user_query="Explain semantic search.",
                thread_id=str(uuid4()),
            )

        client_constructor.assert_not_called()
