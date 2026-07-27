"""API schemas for the assistant endpoint."""

from pydantic import UUID4, BaseModel, ConfigDict, Field


class AssistantRequest(BaseModel):
    """Defines the request schema for the assistant endpoint."""

    model_config = ConfigDict(str_strip_whitespace=True)

    user_query: str = Field(
        min_length=1,
        description="The user query for the research assistant.",
    )

    thread_id: UUID4 = Field(
        description="The unique UUID4 identifier for the conversation thread.",
    )


class AssistantResponse(BaseModel):
    """Defines the response schema for the assistant endpoint."""

    response: str
    thread_id: UUID4
