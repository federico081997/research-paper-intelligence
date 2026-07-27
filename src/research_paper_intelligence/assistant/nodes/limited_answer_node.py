"""LangGraph limited evidence answer node class."""

from typing import TypedDict

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from research_paper_intelligence.assistant.models import (
    ResearchAnswer,
    RetrievedPaper,
)
from research_paper_intelligence.assistant.prompts import (
    LIMITED_EVIDENCE_ANSWER_SYSTEM_PROMPT,
)
from research_paper_intelligence.assistant.state import AssistantState


class FinalAnswerUpdate(TypedDict):
    """State fields updated by the limited-evidence answer node."""

    final_answer: str


class GenerateLimitedAnswerNode:
    """Generate a qualified answer from partially sufficient evidence."""

    def __init__(
        self,
        limited_answer_model: Runnable[
            LanguageModelInput,
            ResearchAnswer,
        ],
    ) -> None:
        """Initialize the limited-evidence answer node.

        Args:
            limited_answer_model: Structured model that produces a
                qualified answer.
        """
        self._limited_answer_model = limited_answer_model

    def __call__(
        self,
        state: AssistantState,
    ) -> FinalAnswerUpdate:
        """Generate the best supported answer from retrieved papers.

        Args:
            state: Current state of the assistant.

        Returns:
            State update for the final_answer field.
        """
        retrieved_papers = state.get("retrieved_papers", [])

        request = (
            f"Original user request:\n"
            f"{state['original_query']}\n\n"
            f"Most recent search query:\n"
            f"{state['search_query']}\n\n"
            f"Retrieval feedback:\n"
            f"{state['retrieval_feedback']}\n\n"
            f"Most recently retrieved papers:\n"
            f"{self._format_papers(retrieved_papers)}"
        )

        result = self._limited_answer_model.invoke(
            [
                SystemMessage(
                    content=LIMITED_EVIDENCE_ANSWER_SYSTEM_PROMPT,
                ),
                HumanMessage(content=request),
            ]
        )

        return {
            "final_answer": result.final_answer,
        }

    @staticmethod
    def _format_papers(
        papers: list[RetrievedPaper],
    ) -> str:
        """Format retrieved papers with stable citation identifiers.

        Args:
            papers: List of retrieved papers.

        Returns:
            Formatted string of retrieved papers.
        """
        formatted_papers: list[str] = []

        for index, paper in enumerate(papers, start=1):
            formatted_papers.append(
                "\n".join(
                    [
                        f"Citation: [{index}]",
                        f"Paper ID: {paper.paper_id}",
                        f"Title: {paper.title}",
                        f"Authors: {paper.authors}",
                        f"Published: {paper.published_date}",
                        f"Abstract: {paper.abstract}",
                    ]
                )
            )

        return "\n\n".join(formatted_papers)
