"""Utilities to build the LLM model."""

from typing import TypeVar, cast

from langchain_core.language_models import (
    BaseChatModel,
    LanguageModelInput,
)
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from research_paper_intelligence.config import Settings

StructuredOutputT = TypeVar(
    "StructuredOutputT",
    bound=BaseModel,
)


def create_chat_model(settings: Settings) -> ChatGoogleGenerativeAI:
    """Create the base chat model used by assistant nodes.

    Args:
        settings: the settings used to create the chat model.

    Returns:
        The base chat model.
    """
    return ChatGoogleGenerativeAI(
        model=settings.model_name,
        api_key=settings.google_api_key.get_secret_value(),
    )


def create_structured_model[StructuredOutputT: BaseModel](
    chat_model: BaseChatModel,
    output_schema: type[StructuredOutputT],
) -> Runnable[LanguageModelInput, StructuredOutputT]:
    """Bind a chat model to a Pydantic structured-output schema.

    Args:
        chat_model: the chat model to bind.
        output_schema: the Pydantic structured-output schema.

    Returns:
        The structured model bound to the chat.
    """
    structured_model = chat_model.with_structured_output(output_schema)

    return cast(
        Runnable[LanguageModelInput, StructuredOutputT],
        structured_model,
    )
