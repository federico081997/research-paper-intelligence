"""LangGraph grounded answer node class."""

from typing import TypedDict

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from research_paper_intelligence.assistant.models import (
    ResearchAnswer,
    RetrievedPaper,
)
from research_paper_intelligence.assistant.prompts import (
    GROUNDED_ANSWER_SYSTEM_PROMPT,
)
from research_paper_intelligence.assistant.state import AssistantState


class FinalAnswerUpdate(TypedDict):
    """State fields updated by the grounded-answer-generation node."""

    final_answer: str


class GenerateGroundedAnswerNode:
    """Generate a complete answer grounded in retrieved papers."""

    def __init__(
        self,
        answer_model: Runnable[
            LanguageModelInput,
            ResearchAnswer,
        ],
    ) -> None:
        """Initialize the grounded-answer node.

        Args:
            answer_model: Structured model that produces a research answer.
        """
        self._answer_model = answer_model

    def __call__(
        self,
        state: AssistantState,
    ) -> FinalAnswerUpdate:
        """Generate the final answer from sufficient retrieved evidence.

        Args:
            state: Current state of the assistant.

        Returns:
            State update for the final_answer field.
        """
        request = (
            f"User request:\n"
            f"{state['original_query']}\n\n"
            f"Retrieved papers:\n"
            f"{self._format_papers(state['retrieved_papers'])}"
        )

        result = self._answer_model.invoke(
            [
                SystemMessage(content=GROUNDED_ANSWER_SYSTEM_PROMPT),
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
