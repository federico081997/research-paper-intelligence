"""Client for retrieving AI assistant response."""

import httpx2

from research_paper_intelligence.api.schemas.assistant import AssistantResponse
from research_paper_intelligence.config import get_settings


def chat(
    user_query: str,
    thread_id: str,
) -> AssistantResponse:
    """Retrieve the assistant response for a user query.

    Args:
        user_query: The user query to be sent to the assistant.
        thread_id: The conversation's thread ID (must be a valid UUID4 string).

    Returns:
        The structured assistant response.
    """
    settings = get_settings()

    payload = {
        "user_query": user_query,
        "thread_id": thread_id,
    }

    with httpx2.Client(
        base_url="http://" + settings.api_host + ":" + str(settings.api_port),
        timeout=settings.api_assistant_timeout_seconds,
    ) as client:
        response = client.post(
            "/api/v1/assistant/chat",
            json=payload,
        )
        response.raise_for_status()

        return AssistantResponse.model_validate(response.json())
