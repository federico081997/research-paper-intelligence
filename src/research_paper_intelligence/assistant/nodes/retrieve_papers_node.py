"""LangGraph the retrieve papers node class."""

from typing import TypedDict

from research_paper_intelligence.assistant.models import RetrievedPaper
from research_paper_intelligence.assistant.retrieval import AssistantRetriever
from research_paper_intelligence.assistant.state import AssistantState


class RetrievePapersUpdate(TypedDict):
    """State fields updated by the retrieve papers node."""

    retrieved_papers: list[RetrievedPaper]
    retrieval_attempts: int


class RetrievePapersNode:
    """Retrieve relevant papers."""

    def __init__(self, search_service: AssistantRetriever) -> None:
        """Initialize retrieved paper node.

        Args:
            search_service: the search service used to retrieve papers.
        """
        self._search_service = search_service

    def __call__(self, state: AssistantState) -> RetrievePapersUpdate:
        """Retrieve papers relevant to a question.

        Args:
            state: AssistantState object

        Returns:
            State update for retrieved_papers and retrieval_attempts
                fields.
        """
        papers = self._search_service.retrieve(
            query=state["search_query"],
            result_k=state["result_k"],
        )

        retrieval_attempts = state.get("retrieval_attempts", 0) + 1

        return {
            "retrieved_papers": papers,
            "retrieval_attempts": retrieval_attempts,
        }
