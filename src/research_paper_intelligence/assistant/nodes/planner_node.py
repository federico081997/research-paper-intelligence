"""LangGraph planner node class."""

from typing import TypedDict

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable

from research_paper_intelligence.assistant.models import PlanRequest
from research_paper_intelligence.assistant.prompts import (
    PLAN_REQUEST_SYSTEM_PROMPT,
)
from research_paper_intelligence.assistant.state import AssistantState


class PlannerNodeUpdate(TypedDict):
    """State fields updated by the planner node."""

    request_type: str
    search_query: str
    result_k: int


class PlannerNode:
    """Plan how the assistant should process the user's request."""

    def __init__(
        self, planner_model: Runnable[LanguageModelInput, PlanRequest]
    ) -> None:
        """Initialize the planner node.

        Args:
            planner_model: Chat model configured to return a PlanRequest.
        """
        self._planner_model = planner_model

    def __call__(self, state: AssistantState) -> PlannerNodeUpdate:
        """Create an execution plan for the current user request.

        Args:
            state: State shared by the assistant.

        Returns:
            State update for request_type, search_query and result_k fields.
        """
        messages = [
            SystemMessage(content=PLAN_REQUEST_SYSTEM_PROMPT),
            *state["messages"],
        ]

        plan = self._planner_model.invoke(messages)

        return {
            "request_type": plan.request_type,
            "search_query": plan.search_query,
            "result_k": plan.result_k,
        }
