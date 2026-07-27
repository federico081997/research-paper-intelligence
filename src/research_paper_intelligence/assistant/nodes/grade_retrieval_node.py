"""LangGraph grade retrieval node class."""

from typing import TypedDict

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from research_paper_intelligence.assistant.models import (
    RetrievalGrader,
    RetrievedPaper,
)
from research_paper_intelligence.assistant.prompts import (
    GRADE_RETRIEVAL_SYSTEM_PROMPT,
)
from research_paper_intelligence.assistant.state import AssistantState


class GradeRetrievalUpdate(TypedDict):
    """State fields updated by the grade retrieval node."""

    retrieval_sufficient: bool
    retrieval_feedback: str


class GradeRetrievalNode:
    """Grade whether the retrieved papers are sufficiently relevant."""

    def __init__(
        self,
        retrieval_grader_model: Runnable[LanguageModelInput, RetrievalGrader],
    ) -> None:
        """Initialize the GradeRetrieval node.

        Args:
            retrieval_grader_model: Chat model configured to return a
                RetrievalGrader instance.
        """
        self._grade_retrieval_model = retrieval_grader_model

    def __call__(
        self,
        state: AssistantState,
    ) -> GradeRetrievalUpdate:
        """Evaluate the relevance and sufficiency of retrieved papers.

        Args:
            state: State shared by the assistant.

        Returns:
            State update for retrieval_sufficient and retrieval_feedback
                fields.
        """
        retrieved_papers = state.get("retrieved_papers", [])

        if not retrieved_papers:
            return {
                "retrieval_sufficient": False,
                "retrieval_feedback": (
                    "No papers were retrieved. Rewrite the search query using "
                    "more specific scientific terminology."
                ),
            }

        evidence = self._format_retrieved_papers(retrieved_papers)

        grading_request = (
            f"Original user request:\n"
            f"{state['original_query']}\n\n"
            f"Search query used:\n"
            f"{state['search_query']}\n\n"
            f"Retrieved papers:\n"
            f"{evidence}"
        )

        grade = self._grade_retrieval_model.invoke(
            [
                SystemMessage(content=GRADE_RETRIEVAL_SYSTEM_PROMPT),
                HumanMessage(content=grading_request),
            ]
        )

        return {
            "retrieval_sufficient": grade.retrieval_sufficient,
            "retrieval_feedback": grade.retrieval_feedback,
        }

    @staticmethod
    def _format_retrieved_papers(
        papers: list[RetrievedPaper],
    ) -> str:
        """Format retrieved papers for evaluation by the grading model.

        Args:
            papers: Papers retrieved by the search_service.

        Returns:
            Formatted retrieved papers for evaluation by the grading model.
        """
        formatted_papers = []

        for index, paper in enumerate(papers, start=1):
            formatted_papers.append(
                "\n".join(
                    [
                        f"Paper {index}",
                        f"ID: {paper.paper_id}",
                        f"Title: {paper.title}",
                        f"Abstract: {paper.abstract}",
                    ]
                )
            )

        return "\n\n".join(formatted_papers)
