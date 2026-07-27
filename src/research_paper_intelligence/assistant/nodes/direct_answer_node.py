"""LangGraph direct answer node class."""

from typing import TypedDict

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable

from research_paper_intelligence.assistant.models import DirectAnswer
from research_paper_intelligence.assistant.prompts import (
    DIRECT_ANSWER_SYSTEM_PROMPT,
)
from research_paper_intelligence.assistant.state import AssistantState


class DirectAnswerNodeUpdate(TypedDict):
    """State fields updated by the direct answer node."""

    final_answer: str


class GenerateDirectAnswerNode:
    """Generate an answer without retrieving research papers."""

    def __init__(
        self,
        direct_answer_model: Runnable[
            LanguageModelInput,
            DirectAnswer,
        ],
    ) -> None:
        """Initialize the direct-answer node.

        Args:
            direct_answer_model: Chat model configured to return a
                DirectAnswer.
        """
        self._direct_answer_model = direct_answer_model

    def __call__(
        self,
        state: AssistantState,
    ) -> DirectAnswerNodeUpdate:
        """Generate a direct answer and store it in graph state.

        Args:
            state: State shared by the assistant.

        Returns:
            State update for the final_answer field.
        """
        messages = [
            SystemMessage(content=DIRECT_ANSWER_SYSTEM_PROMPT),
            *state["messages"],
        ]
        result = self._direct_answer_model.invoke(messages)

        return {"final_answer": result.final_answer}
