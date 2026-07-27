"""LangGraph rewrite query node class."""

from typing import TypedDict

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from research_paper_intelligence.assistant.models import (
    RewrittenQuery,
)
from research_paper_intelligence.assistant.prompts import (
    REWRITE_QUERY_SYSTEM_PROMPT,
)
from research_paper_intelligence.assistant.state import AssistantState


class RewriteQueryUpdate(TypedDict):
    """State fields updated by the rewrite-query node."""

    search_query: str
    rewrite_count: int


class RewriteQueryNode:
    """Rewrite an unsuccessful scientific-paper search query."""

    def __init__(
        self,
        query_rewriter_model: Runnable[
            LanguageModelInput,
            RewrittenQuery,
        ],
    ) -> None:
        """Initialize the rewrite-query node.

        Args:
            query_rewriter_model: Structured model configured to produce a
                ``RewrittenQuery``.
        """
        self._query_rewriter_model = query_rewriter_model

    def __call__(
        self,
        state: AssistantState,
    ) -> RewriteQueryUpdate:
        """Rewrite the search query using retrieval feedback.

        Args:
            state: Current state of the assistant.

        Returns:
            State update for search_query and rewrite_count fields.
        """
        original_query = state["original_query"]
        search_query = state["search_query"]
        retrieval_feedback = state["retrieval_feedback"]

        rewrite_request = (
            f"Original user request:\n"
            f"{original_query}\n\n"
            f"Previous search query:\n"
            f"{search_query}\n\n"
            f"Retrieval feedback:\n"
            f"{retrieval_feedback}"
        )

        result = self._query_rewriter_model.invoke(
            [
                SystemMessage(content=REWRITE_QUERY_SYSTEM_PROMPT),
                HumanMessage(content=rewrite_request),
            ]
        )

        rewritten_query = result.search_query.strip()

        return {
            "search_query": rewritten_query,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }
