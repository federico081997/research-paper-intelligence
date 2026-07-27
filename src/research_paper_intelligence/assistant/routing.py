"""Routing components for the research assistant graph."""

from typing import Literal

from research_paper_intelligence.assistant.state import AssistantState
from research_paper_intelligence.config import get_settings

type PlanningRoute = Literal[
    "generate_direct_answer",
    "retrieve_papers",
]

type RetrievalRoute = Literal[
    "rewrite_query",
    "generate_grounded_answer",
    "generate_limited_answer",
]


def route_after_planning(state: AssistantState) -> PlanningRoute:
    """Route the request according to the planner decision.

    Args:
        state: assistant state shared by the graph nodes.

    Returns:
        The registered name of the next graph node.
    """
    if state["request_type"] == "direct":
        return "generate_direct_answer"

    return "retrieve_papers"


def route_after_retrieval(state: AssistantState) -> RetrievalRoute:
    """Create the router used after retrieval grading.

    Args:
        state: assistant state shared by the graph nodes.

    Returns:
        The registered name of the next graph node.
    """
    max_query_rewrites = get_settings().max_query_rewrites

    if state["retrieval_sufficient"]:
        return "generate_grounded_answer"

    if state.get("rewrite_count", 0) < max_query_rewrites:
        return "rewrite_query"

    return "generate_limited_answer"
