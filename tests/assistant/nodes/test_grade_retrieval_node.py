"""Tests for the grade-retrieval LangGraph node."""

from datetime import date
from typing import cast
from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from research_paper_intelligence.assistant.models import (
    RetrievalGrader,
    RetrievedPaper,
)
from research_paper_intelligence.assistant.nodes.grade_retrieval_node import (
    GradeRetrievalNode,
)
from research_paper_intelligence.assistant.prompts import (
    GRADE_RETRIEVAL_SYSTEM_PROMPT,
)
from research_paper_intelligence.assistant.state import AssistantState


@pytest.fixture
def retrieval_grader_model() -> Mock:
    """Create a mocked structured retrieval-grading model."""
    return Mock(spec=Runnable)


@pytest.fixture
def grade_retrieval_node(
    retrieval_grader_model: Mock,
) -> GradeRetrievalNode:
    """Create a grade-retrieval node using a mocked model."""
    return GradeRetrievalNode(
        retrieval_grader_model=retrieval_grader_model,
    )


class TestGradeRetrievalNode:
    """Tests for the GradeRetrievalNode class."""

    def test_returns_sufficient_retrieval_grade(
        self,
        grade_retrieval_node: GradeRetrievalNode,
        retrieval_grader_model: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Return a sufficient retrieval decision from the grading model."""
        retrieval_grader_model.invoke.return_value = RetrievalGrader(
            retrieval_sufficient=True,
            retrieval_feedback=(
                "The papers directly address finite volume elastoplasticity."
            ),
        )
        state = cast(
            AssistantState,
            {
                "original_query": (
                    "Find papers about finite volume elastoplasticity."
                ),
                "search_query": (
                    "finite volume elastoplasticity solid mechanics"
                ),
                "retrieved_papers": retrieved_papers,
            },
        )

        result = grade_retrieval_node(state)

        assert result == {
            "retrieval_sufficient": True,
            "retrieval_feedback": (
                "The papers directly address finite volume elastoplasticity."
            ),
        }

    def test_returns_insufficient_retrieval_grade(
        self,
        grade_retrieval_node: GradeRetrievalNode,
        retrieval_grader_model: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Return an insufficient decision and the model feedback."""
        retrieval_grader_model.invoke.return_value = RetrievalGrader(
            retrieval_sufficient=False,
            retrieval_feedback=(
                "The papers do not compare the requested numerical methods."
            ),
        )
        state = cast(
            AssistantState,
            {
                "original_query": (
                    "Compare finite volume and finite element methods."
                ),
                "search_query": "finite volume methods",
                "retrieved_papers": retrieved_papers,
            },
        )

        result = grade_retrieval_node(state)

        assert result == {
            "retrieval_sufficient": False,
            "retrieval_feedback": (
                "The papers do not compare the requested numerical methods."
            ),
        }

    @pytest.mark.parametrize(
        "state",
        [
            {
                "original_query": "Find papers about machine learning.",
                "search_query": "machine learning",
            },
            {
                "original_query": "Find papers about machine learning.",
                "search_query": "machine learning",
                "retrieved_papers": [],
            },
        ],
    )
    def test_returns_insufficient_grade_when_no_papers_are_retrieved(
        self,
        grade_retrieval_node: GradeRetrievalNode,
        retrieval_grader_model: Mock,
        state: dict[str, object],
    ) -> None:
        """Return fixed feedback when no retrieved papers are available."""
        assistant_state = cast(AssistantState, state)

        result = grade_retrieval_node(assistant_state)

        assert result == {
            "retrieval_sufficient": False,
            "retrieval_feedback": (
                "No papers were retrieved. Rewrite the search query using "
                "more specific scientific terminology."
            ),
        }
        retrieval_grader_model.invoke.assert_not_called()

    def test_invokes_grader_with_system_and_human_messages(
        self,
        grade_retrieval_node: GradeRetrievalNode,
        retrieval_grader_model: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Invoke the grading model with the expected message sequence."""
        retrieval_grader_model.invoke.return_value = RetrievalGrader(
            retrieval_sufficient=True,
            retrieval_feedback="The evidence is sufficient.",
        )
        state = cast(
            AssistantState,
            {
                "original_query": (
                    "Explain finite volume methods for elastoplasticity."
                ),
                "search_query": (
                    "finite volume elastoplasticity numerical method"
                ),
                "retrieved_papers": retrieved_papers,
            },
        )

        grade_retrieval_node(state)

        retrieval_grader_model.invoke.assert_called_once()

        messages = retrieval_grader_model.invoke.call_args.args[0]

        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert messages[0].content == GRADE_RETRIEVAL_SYSTEM_PROMPT

    def test_includes_request_query_and_papers_in_grading_request(
        self,
        grade_retrieval_node: GradeRetrievalNode,
        retrieval_grader_model: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Include the request, search query, and evidence in the prompt."""
        original_query = (
            "Find research on block-coupled finite volume methods."
        )
        search_query = (
            "block-coupled finite volume computational solid mechanics"
        )
        retrieval_grader_model.invoke.return_value = RetrievalGrader(
            retrieval_sufficient=True,
            retrieval_feedback="The evidence is sufficient.",
        )
        state = cast(
            AssistantState,
            {
                "original_query": original_query,
                "search_query": search_query,
                "retrieved_papers": retrieved_papers,
            },
        )

        grade_retrieval_node(state)

        messages = retrieval_grader_model.invoke.call_args.args[0]
        grading_request = messages[1].content

        assert isinstance(grading_request, str)
        assert f"Original user request:\n{original_query}" in grading_request
        assert f"Search query used:\n{search_query}" in grading_request
        assert "Retrieved papers:" in grading_request
        assert "Paper 1" in grading_request
        assert "ID: 2401.12345" in grading_request
        assert (
            "Title: Finite volume methods for solid mechanics"
            in grading_request
        )
        assert (
            "Abstract: A block-coupled finite volume method is developed "
            "for computational solid mechanics." in grading_request
        )
        assert "Paper 2" in grading_request
        assert "ID: 2402.67890" in grading_request

    def test_returns_only_retrieval_grade_fields(
        self,
        grade_retrieval_node: GradeRetrievalNode,
        retrieval_grader_model: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Return only fields managed by the grade-retrieval node."""
        retrieval_grader_model.invoke.return_value = RetrievalGrader(
            retrieval_sufficient=True,
            retrieval_feedback="The evidence is sufficient.",
        )
        state = cast(
            AssistantState,
            {
                "original_query": "Explain finite volume methods.",
                "search_query": "finite volume methods",
                "retrieved_papers": retrieved_papers,
            },
        )

        result = grade_retrieval_node(state)

        assert set(result) == {
            "retrieval_sufficient",
            "retrieval_feedback",
        }


class TestFormatRetrievedPapers:
    """Tests for retrieved-paper evidence formatting."""

    def test_formats_single_paper(self) -> None:
        """Format one retrieved paper with its identifying evidence."""
        paper = RetrievedPaper(
            paper_id="2501.00001",
            title="Semantic search for scientific literature",
            abstract="A study of embedding-based scientific search.",
            authors="Example Author",
            category="Machine Learning",
            published_date=date(2025, 1, 1),
            rank=1,
            hybrid_score=0.92,
        )

        result = GradeRetrievalNode._format_retrieved_papers([paper])

        assert result == (
            "Paper 1\n"
            "ID: 2501.00001\n"
            "Title: Semantic search for scientific literature\n"
            "Abstract: A study of embedding-based scientific search."
        )

    def test_formats_multiple_papers_with_sequential_numbers(
        self,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Separate papers and number them according to list order."""
        result = GradeRetrievalNode._format_retrieved_papers(retrieved_papers)

        expected = (
            "Paper 1\n"
            "ID: 2401.12345\n"
            "Title: Finite volume methods for solid mechanics\n"
            "Abstract: A block-coupled finite volume method is developed "
            "for computational solid mechanics.\n\n"
            "Paper 2\n"
            "ID: 2402.67890\n"
            "Title: Elastoplasticity using finite volume discretisation\n"
            "Abstract: This paper studies elastoplastic constitutive models "
            "within a finite volume framework."
        )

        assert result == expected

    def test_returns_empty_string_for_empty_paper_list(self) -> None:
        """Return an empty evidence string when the paper list is empty."""
        result = GradeRetrievalNode._format_retrieved_papers([])

        assert result == ""
