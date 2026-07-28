"""Tests for the limited-evidence answer LangGraph node."""

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
from research_paper_intelligence.assistant.nodes.limited_answer_node import (
    GenerateLimitedAnswerNode,
)
from research_paper_intelligence.assistant.prompts import (
    LIMITED_EVIDENCE_ANSWER_SYSTEM_PROMPT,
)
from research_paper_intelligence.assistant.state import AssistantState


@pytest.fixture
def limited_answer_model() -> Mock:
    """Create a mocked structured limited-answer model."""
    return Mock(spec=Runnable)


@pytest.fixture
def limited_answer_node(
    limited_answer_model: Mock,
) -> GenerateLimitedAnswerNode:
    """Create a limited-answer node using a mocked model."""
    return GenerateLimitedAnswerNode(
        limited_answer_model=limited_answer_model,
    )


class TestGenerateLimitedAnswerNode:
    """Tests for the GenerateLimitedAnswerNode class."""

    def test_returns_generated_final_answer(
        self,
        limited_answer_node: GenerateLimitedAnswerNode,
        limited_answer_model: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Return the qualified answer as a final-answer state update."""
        final_answer = (
            "## Limited findings\n\n"
            "The retrieved papers provide partial evidence for the "
            "requested comparison [1]."
        )
        limited_answer_model.invoke.return_value = ResearchAnswer(
            final_answer=final_answer,
        )
        state = cast(
            AssistantState,
            {
                "original_query": (
                    "Compare finite volume and finite element methods."
                ),
                "search_query": "finite volume solid mechanics",
                "retrieval_feedback": (
                    "The papers discuss finite volume methods but do not "
                    "provide a direct finite-element comparison."
                ),
                "retrieved_papers": retrieved_papers,
            },
        )

        result = limited_answer_node(state)

        assert result == {"final_answer": final_answer}

    def test_invokes_limited_answer_model_once(
        self,
        limited_answer_node: GenerateLimitedAnswerNode,
        limited_answer_model: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Invoke the configured limited-answer model exactly once."""
        limited_answer_model.invoke.return_value = ResearchAnswer(
            final_answer="Limited-evidence answer.",
        )
        state = cast(
            AssistantState,
            {
                "original_query": "Compare the numerical methods.",
                "search_query": "finite volume numerical methods",
                "retrieval_feedback": (
                    "The retrieved evidence is incomplete."
                ),
                "retrieved_papers": retrieved_papers,
            },
        )

        limited_answer_node(state)

        limited_answer_model.invoke.assert_called_once()

    def test_invokes_model_with_system_and_human_messages(
        self,
        limited_answer_node: GenerateLimitedAnswerNode,
        limited_answer_model: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Invoke the model with the expected message sequence."""
        limited_answer_model.invoke.return_value = ResearchAnswer(
            final_answer="Limited-evidence answer.",
        )
        state = cast(
            AssistantState,
            {
                "original_query": "Explain the available evidence.",
                "search_query": "finite volume evidence",
                "retrieval_feedback": (
                    "Only partially relevant papers were found."
                ),
                "retrieved_papers": retrieved_papers,
            },
        )

        limited_answer_node(state)

        messages = limited_answer_model.invoke.call_args.args[0]

        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert messages[0].content == LIMITED_EVIDENCE_ANSWER_SYSTEM_PROMPT

    def test_includes_retrieval_context_in_request(
        self,
        limited_answer_node: GenerateLimitedAnswerNode,
        limited_answer_model: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Include the request, query, feedback, and papers in the prompt."""
        original_query = (
            "Compare finite volume and finite element elastoplasticity."
        )
        search_query = "finite volume elastoplasticity"
        retrieval_feedback = (
            "The retrieved papers cover finite volume formulations but "
            "not the requested finite-element comparison."
        )
        limited_answer_model.invoke.return_value = ResearchAnswer(
            final_answer="Limited-evidence answer.",
        )
        state = cast(
            AssistantState,
            {
                "original_query": original_query,
                "search_query": search_query,
                "retrieval_feedback": retrieval_feedback,
                "retrieved_papers": retrieved_papers,
            },
        )

        limited_answer_node(state)

        messages = limited_answer_model.invoke.call_args.args[0]
        request = messages[1].content

        assert isinstance(request, str)
        assert f"Original user request:\n{original_query}" in request
        assert f"Most recent search query:\n{search_query}" in request
        assert f"Retrieval feedback:\n{retrieval_feedback}" in request
        assert "Most recently retrieved papers:" in request

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

    def test_handles_missing_retrieved_papers(
        self,
        limited_answer_node: GenerateLimitedAnswerNode,
        limited_answer_model: Mock,
    ) -> None:
        """Generate a qualified answer when no papers are in the state."""
        final_answer = (
            "No sufficiently relevant papers were found, so the requested "
            "question cannot be answered reliably from the collection."
        )
        limited_answer_model.invoke.return_value = ResearchAnswer(
            final_answer=final_answer,
        )
        state = cast(
            AssistantState,
            {
                "original_query": "Explain the requested method.",
                "search_query": "specific scientific method",
                "retrieval_feedback": "No relevant papers were retrieved.",
            },
        )

        result = limited_answer_node(state)

        messages = limited_answer_model.invoke.call_args.args[0]
        request = messages[1].content

        assert isinstance(request, str)
        assert request.endswith("Most recently retrieved papers:\n")
        assert result == {"final_answer": final_answer}

    def test_handles_empty_retrieved_paper_list(
        self,
        limited_answer_node: GenerateLimitedAnswerNode,
        limited_answer_model: Mock,
    ) -> None:
        """Generate an answer when the retrieved-paper list is empty."""
        limited_answer_model.invoke.return_value = ResearchAnswer(
            final_answer="No relevant evidence was available.",
        )
        state = cast(
            AssistantState,
            {
                "original_query": "Find evidence for this method.",
                "search_query": "scientific method evidence",
                "retrieval_feedback": "No papers were retrieved.",
                "retrieved_papers": [],
            },
        )

        result = limited_answer_node(state)

        assert result == {
            "final_answer": "No relevant evidence was available.",
        }
        limited_answer_model.invoke.assert_called_once()

    def test_returns_only_final_answer_field(
        self,
        limited_answer_node: GenerateLimitedAnswerNode,
        limited_answer_model: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Return only the state field managed by this node."""
        limited_answer_model.invoke.return_value = ResearchAnswer(
            final_answer="Limited-evidence answer.",
        )
        state = cast(
            AssistantState,
            {
                "original_query": "Explain the available evidence.",
                "search_query": "available scientific evidence",
                "retrieval_feedback": (
                    "The retrieved evidence is incomplete."
                ),
                "retrieved_papers": retrieved_papers,
            },
        )

        result = limited_answer_node(state)

        assert set(result) == {"final_answer"}

    def test_preserves_markdown_in_generated_answer(
        self,
        limited_answer_node: GenerateLimitedAnswerNode,
        limited_answer_model: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Preserve Markdown returned by the structured answer model."""
        final_answer = (
            "## Evidence limitations\n\n"
            "The retrieved evidence only partially addresses the request.\n\n"
            "## Available findings\n\n"
            "- The method is described in one retrieved paper [1].\n"
            "- A direct comparison was not found.\n\n"
            "## References\n\n"
            "[1] Author One and Author Two."
        )
        limited_answer_model.invoke.return_value = ResearchAnswer(
            final_answer=final_answer,
        )
        state = cast(
            AssistantState,
            {
                "original_query": "Compare the available methods.",
                "search_query": "comparison of numerical methods",
                "retrieval_feedback": (
                    "A complete comparison was not retrieved."
                ),
                "retrieved_papers": retrieved_papers,
            },
        )

        result = limited_answer_node(state)

        assert result["final_answer"] == final_answer


class TestFormatPapers:
    """Tests for limited-evidence paper formatting."""

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

        result = GenerateLimitedAnswerNode._format_papers([paper])

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
        result = GenerateLimitedAnswerNode._format_papers(
            retrieved_papers,
        )

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
            rank=8,
            hybrid_score=0.50,
        )

        result = GenerateLimitedAnswerNode._format_papers([paper])

        assert result.startswith("Citation: [1]")
        assert "Citation: [8]" not in result

    def test_returns_empty_string_for_empty_paper_list(self) -> None:
        """Return an empty string when no papers are supplied."""
        result = GenerateLimitedAnswerNode._format_papers([])

        assert result == ""
