"""Tests for the grounded-answer LangGraph node."""

from datetime import date
from typing import cast
from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from research_paper_intelligence.assistant.models import (
    ResearchAnswer,
    RetrievedPaper,
)
from research_paper_intelligence.assistant.nodes.grounded_answer_node import (
    GenerateGroundedAnswerNode,
)
from research_paper_intelligence.assistant.prompts import (
    GROUNDED_ANSWER_SYSTEM_PROMPT,
)
from research_paper_intelligence.assistant.state import AssistantState


@pytest.fixture
def answer_model() -> Mock:
    """Create a mocked structured research-answer model."""
    return Mock(spec=Runnable)


@pytest.fixture
def grounded_answer_node(
    answer_model: Mock,
) -> GenerateGroundedAnswerNode:
    """Create a grounded-answer node using a mocked model."""
    return GenerateGroundedAnswerNode(answer_model=answer_model)


class TestGenerateGroundedAnswerNode:
    """Tests for the GenerateGroundedAnswerNode class."""

    def test_returns_generated_final_answer(
        self,
        grounded_answer_node: GenerateGroundedAnswerNode,
        answer_model: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Return the research answer as a final-answer state update."""
        final_answer = (
            "## Summary\n\n"
            "Finite volume methods can be applied to solid mechanics [1]."
        )
        answer_model.invoke.return_value = ResearchAnswer(
            final_answer=final_answer
        )
        state = cast(
            AssistantState,
            {
                "original_query": (
                    "Explain finite volume methods for solid mechanics."
                ),
                "retrieved_papers": retrieved_papers,
            },
        )

        result = grounded_answer_node(state)

        assert result == {"final_answer": final_answer}

    def test_invokes_answer_model_once(
        self,
        grounded_answer_node: GenerateGroundedAnswerNode,
        answer_model: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Invoke the configured answer model exactly once."""
        answer_model.invoke.return_value = ResearchAnswer(
            final_answer="Grounded answer."
        )
        state = cast(
            AssistantState,
            {
                "original_query": "Explain finite volume methods.",
                "retrieved_papers": retrieved_papers,
            },
        )

        grounded_answer_node(state)

        answer_model.invoke.assert_called_once()

    def test_invokes_model_with_system_and_human_messages(
        self,
        grounded_answer_node: GenerateGroundedAnswerNode,
        answer_model: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Invoke the model with the expected message sequence."""
        answer_model.invoke.return_value = ResearchAnswer(
            final_answer="Grounded answer."
        )
        state = cast(
            AssistantState,
            {
                "original_query": "Explain finite volume methods.",
                "retrieved_papers": retrieved_papers,
            },
        )

        grounded_answer_node(state)

        messages = answer_model.invoke.call_args.args[0]

        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert messages[0].content == GROUNDED_ANSWER_SYSTEM_PROMPT

    def test_includes_user_request_and_retrieved_papers(
        self,
        grounded_answer_node: GenerateGroundedAnswerNode,
        answer_model: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Include the original request and paper evidence in the prompt."""
        original_query = (
            "How are finite volume methods used for elastoplasticity?"
        )
        answer_model.invoke.return_value = ResearchAnswer(
            final_answer="Grounded answer."
        )
        state = cast(
            AssistantState,
            {
                "original_query": original_query,
                "retrieved_papers": retrieved_papers,
            },
        )

        grounded_answer_node(state)

        messages = answer_model.invoke.call_args.args[0]
        request = messages[1].content

        assert isinstance(request, str)
        assert f"User request:\n{original_query}" in request
        assert "Retrieved papers:" in request

        assert "Citation: [1]" in request
        assert "Paper ID: 2401.12345" in request
        assert "Title: Finite volume methods for solid mechanics" in request
        assert "Authors: Author One, Author Two" in request
        assert "Published: 2025-01-15" in request
        assert (
            "Abstract: A block-coupled finite volume method is developed "
            "for computational solid mechanics." in request
        )

        assert "Citation: [2]" in request
        assert "Paper ID: 2402.67890" in request
        assert (
            "Title: Elastoplasticity using finite volume discretisation"
            in request
        )

    def test_returns_only_final_answer_field(
        self,
        grounded_answer_node: GenerateGroundedAnswerNode,
        answer_model: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Return only the state field managed by this node."""
        answer_model.invoke.return_value = ResearchAnswer(
            final_answer="Grounded answer."
        )
        state = cast(
            AssistantState,
            {
                "original_query": "Explain finite volume methods.",
                "retrieved_papers": retrieved_papers,
            },
        )

        result = grounded_answer_node(state)

        assert set(result) == {"final_answer"}

    def test_preserves_markdown_in_generated_answer(
        self,
        grounded_answer_node: GenerateGroundedAnswerNode,
        answer_model: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Preserve Markdown returned by the structured answer model."""
        final_answer = (
            "## Main findings\n\n"
            "- Finite volume methods conserve fluxes [1].\n"
            "- Block coupling improves equation coupling [1].\n\n"
            "## References\n\n"
            "[1] Author One and Author Two."
        )
        answer_model.invoke.return_value = ResearchAnswer(
            final_answer=final_answer
        )
        state = cast(
            AssistantState,
            {
                "original_query": "Explain the main findings.",
                "retrieved_papers": retrieved_papers,
            },
        )

        result = grounded_answer_node(state)

        assert result["final_answer"] == final_answer


class TestFormatPapers:
    """Tests for retrieved-paper citation formatting."""

    def test_formats_single_paper(self) -> None:
        """Format one paper with a stable citation identifier."""
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

        result = GenerateGroundedAnswerNode._format_papers([paper])

        assert result == (
            "Citation: [1]\n"
            "Paper ID: 2501.00001\n"
            "Title: Semantic search for scientific literature\n"
            "Authors: Example Author\n"
            "Published: 2025-01-01\n"
            "Abstract: A study of embedding-based scientific search."
        )

    def test_formats_multiple_papers(
        self,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Separate papers and assign sequential citation identifiers."""
        result = GenerateGroundedAnswerNode._format_papers(retrieved_papers)

        expected = (
            "Citation: [1]\n"
            "Paper ID: 2401.12345\n"
            "Title: Finite volume methods for solid mechanics\n"
            "Authors: Author One, Author Two\n"
            "Published: 2025-01-15\n"
            "Abstract: A block-coupled finite volume method is developed "
            "for computational solid mechanics.\n\n"
            "Citation: [2]\n"
            "Paper ID: 2402.67890\n"
            "Title: Elastoplasticity using finite volume discretisation\n"
            "Authors: Author Three\n"
            "Published: 2024-06-10\n"
            "Abstract: This paper studies elastoplastic constitutive models "
            "within a finite volume framework."
        )

        assert result == expected

    def test_uses_list_order_for_citation_numbers(self) -> None:
        """Assign citations by list order rather than stored paper rank."""
        paper = RetrievedPaper(
            paper_id="2501.00001",
            title="Example paper",
            abstract="Example abstract.",
            authors="Example Author",
            category="Machine Learning",
            published_date=date(2025, 1, 1),
            rank=7,
            hybrid_score=0.50,
        )

        result = GenerateGroundedAnswerNode._format_papers([paper])

        assert result.startswith("Citation: [1]")
        assert "Citation: [7]" not in result

    def test_returns_empty_string_for_empty_paper_list(self) -> None:
        """Return an empty evidence string when no papers are supplied."""
        result = GenerateGroundedAnswerNode._format_papers([])

        assert result == ""
