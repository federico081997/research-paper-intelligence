"""Tests for the assistant API router."""

from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from research_paper_intelligence.api.dependencies import (
    get_assistant_service,
)
from research_paper_intelligence.api.routers.assistant_router import (
    assistant_router,
)


@pytest.fixture
def assistant_app(assistant_service: Mock) -> FastAPI:
    """Create a FastAPI application containing the assistant router."""
    app = FastAPI()
    app.include_router(assistant_router)

    app.dependency_overrides[get_assistant_service] = lambda: assistant_service

    return app


@pytest.fixture
def assistant_client(assistant_app: FastAPI) -> TestClient:
    """Create a test client for the assistant router."""
    return TestClient(assistant_app)


class TestChatWithAssistant:
    """Tests for the assistant chat endpoint."""

    def test_returns_assistant_response(
        self,
        assistant_client: TestClient,
        assistant_service: Mock,
    ) -> None:
        """Return the generated answer and the supplied thread ID."""
        thread_id = uuid4()
        assistant_service.chat.return_value = (
            "Finite volume methods discretize conservation laws."
        )

        response = assistant_client.post(
            "/assistant/chat",
            json={
                "user_query": "Explain finite volume methods.",
                "thread_id": str(thread_id),
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "response": (
                "Finite volume methods discretize conservation laws."
            ),
            "thread_id": str(thread_id),
        }

    def test_passes_request_values_to_assistant_service(
        self,
        assistant_client: TestClient,
        assistant_service: Mock,
    ) -> None:
        """Pass the query and string thread ID to the assistant service."""
        thread_id = uuid4()
        assistant_service.chat.return_value = "Test response."

        assistant_client.post(
            "/assistant/chat",
            json={
                "user_query": "Find papers about elastoplasticity.",
                "thread_id": str(thread_id),
            },
        )

        assistant_service.chat.assert_called_once_with(
            user_query="Find papers about elastoplasticity.",
            thread_id=str(thread_id),
        )

    def test_returns_valid_uuid_in_response(
        self,
        assistant_client: TestClient,
        assistant_service: Mock,
    ) -> None:
        """Return the thread ID as a valid UUID value."""
        thread_id = uuid4()
        assistant_service.chat.return_value = "Test response."

        response = assistant_client.post(
            "/assistant/chat",
            json={
                "user_query": "Explain semantic search.",
                "thread_id": str(thread_id),
            },
        )

        response_thread_id = UUID(response.json()["thread_id"])

        assert response_thread_id == thread_id

    def test_rejects_empty_user_query(
        self,
        assistant_client: TestClient,
        assistant_service: Mock,
    ) -> None:
        """Reject a request containing an empty user query."""
        response = assistant_client.post(
            "/assistant/chat",
            json={
                "user_query": "",
                "thread_id": str(uuid4()),
            },
        )

        assert response.status_code == 422
        assistant_service.chat.assert_not_called()

    @pytest.mark.parametrize(
        "invalid_thread_id",
        [
            "not-a-uuid",
            "",
            "12345",
        ],
    )
    def test_rejects_invalid_thread_id(
        self,
        assistant_client: TestClient,
        assistant_service: Mock,
        invalid_thread_id: str,
    ) -> None:
        """Reject requests containing an invalid thread ID."""
        response = assistant_client.post(
            "/assistant/chat",
            json={
                "user_query": "Explain machine learning.",
                "thread_id": invalid_thread_id,
            },
        )

        assert response.status_code == 422
        assistant_service.chat.assert_not_called()

    @pytest.mark.parametrize(
        "missing_field",
        [
            "user_query",
            "thread_id",
        ],
    )
    def test_rejects_missing_required_fields(
        self,
        assistant_client: TestClient,
        assistant_service: Mock,
        missing_field: str,
    ) -> None:
        """Reject requests that omit a required request field."""
        payload = {
            "user_query": "Explain neural networks.",
            "thread_id": str(uuid4()),
        }
        del payload[missing_field]

        response = assistant_client.post(
            "/assistant/chat",
            json=payload,
        )

        assert response.status_code == 422
        assistant_service.chat.assert_not_called()
