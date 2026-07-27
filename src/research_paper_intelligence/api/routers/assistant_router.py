"""Contains the assistant router for the API."""

from fastapi import APIRouter

from research_paper_intelligence.api.dependencies import (
    AssistantServiceDependency,
)
from research_paper_intelligence.api.schemas.assistant import (
    AssistantRequest,
    AssistantResponse,
)

assistant_router = APIRouter(prefix="/assistant", tags=["Assistant"])


@assistant_router.post("/chat", response_model=AssistantResponse)
def chat_with_assistant(
    request: AssistantRequest,
    assistant: AssistantServiceDependency,
) -> AssistantResponse:
    """Handle incoming chat messages.

    Args:
        request: The incoming user request.
        assistant: The assistant service dependency.
    """
    answer = assistant.chat(
        user_query=request.user_query, thread_id=str(request.thread_id)
    )

    return AssistantResponse(response=answer, thread_id=request.thread_id)
