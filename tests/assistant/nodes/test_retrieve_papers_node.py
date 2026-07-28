"""Tests for the retrieve-papers LangGraph node."""

from typing import cast
from unittest.mock import Mock

import pytest

from research_paper_intelligence.assistant.models import RetrievedPaper
from research_paper_intelligence.assistant.nodes.retrieve_papers_node import (
    RetrievePapersNode,
)
from research_paper_intelligence.assistant.retrieval import AssistantRetriever
from research_paper_intelligence.assistant.state import AssistantState


@pytest.fixture
def assistant_retriever() -> Mock:
    """Create a mocked assistant paper retriever."""
    return Mock(spec=AssistantRetriever)


@pytest.fixture
def retrieve_papers_node(
    assistant_retriever: Mock,
) -> RetrievePapersNode:
    """Create a retrieve-papers node using a mocked retriever."""
    return RetrievePapersNode(search_service=assistant_retriever)


class TestRetrievePapersNode:
    """Tests for the RetrievePapersNode class."""

    def test_returns_retrieved_papers(
        self,
        retrieve_papers_node: RetrievePapersNode,
        assistant_retriever: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Return papers produced by the assistant retriever."""
        assistant_retriever.retrieve.return_value = retrieved_papers
        state = cast(
            AssistantState,
            {
                "search_query": "finite volume elastoplasticity",
                "result_k": 2,
            },
        )

        result = retrieve_papers_node(state)

        assert result["retrieved_papers"] is retrieved_papers

    def test_passes_query_and_result_count_to_retriever(
        self,
        retrieve_papers_node: RetrievePapersNode,
        assistant_retriever: Mock,
    ) -> None:
        """Pass the planned search query and result count to retrieval."""
        assistant_retriever.retrieve.return_value = []
        state = cast(
            AssistantState,
            {
                "search_query": (
                    "block-coupled finite volume solid mechanics"
                ),
                "result_k": 7,
            },
        )

        retrieve_papers_node(state)

        assistant_retriever.retrieve.assert_called_once_with(
            query="block-coupled finite volume solid mechanics",
            result_k=7,
        )

    def test_sets_first_retrieval_attempt_to_one(
        self,
        retrieve_papers_node: RetrievePapersNode,
        assistant_retriever: Mock,
    ) -> None:
        """Set retrieval attempts to one when no previous count exists."""
        assistant_retriever.retrieve.return_value = []
        state = cast(
            AssistantState,
            {
                "search_query": "neural operators",
                "result_k": 5,
            },
        )

        result = retrieve_papers_node(state)

        assert result["retrieval_attempts"] == 1

    @pytest.mark.parametrize(
        ("current_attempts", "expected_attempts"),
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (5, 6),
        ],
    )
    def test_increments_existing_retrieval_attempts(
        self,
        retrieve_papers_node: RetrievePapersNode,
        assistant_retriever: Mock,
        current_attempts: int,
        expected_attempts: int,
    ) -> None:
        """Increment the number of completed retrieval attempts."""
        assistant_retriever.retrieve.return_value = []
        state = cast(
            AssistantState,
            {
                "search_query": "scientific machine learning",
                "result_k": 5,
                "retrieval_attempts": current_attempts,
            },
        )

        result = retrieve_papers_node(state)

        assert result["retrieval_attempts"] == expected_attempts

    def test_returns_empty_paper_list_when_no_results_are_found(
        self,
        retrieve_papers_node: RetrievePapersNode,
        assistant_retriever: Mock,
    ) -> None:
        """Return an empty list when the retriever finds no papers."""
        assistant_retriever.retrieve.return_value = []
        state = cast(
            AssistantState,
            {
                "search_query": "highly specific unavailable topic",
                "result_k": 5,
            },
        )

        result = retrieve_papers_node(state)

        assert result == {
            "retrieved_papers": [],
            "retrieval_attempts": 1,
        }

    def test_returns_only_retrieval_state_fields(
        self,
        retrieve_papers_node: RetrievePapersNode,
        assistant_retriever: Mock,
        retrieved_papers: list[RetrievedPaper],
    ) -> None:
        """Return only fields managed by the retrieval node."""
        assistant_retriever.retrieve.return_value = retrieved_papers
        state = cast(
            AssistantState,
            {
                "search_query": "finite volume methods",
                "result_k": 2,
            },
        )

        result = retrieve_papers_node(state)

        assert set(result) == {
            "retrieved_papers",
            "retrieval_attempts",
        }

    def test_invokes_retriever_once(
        self,
        retrieve_papers_node: RetrievePapersNode,
        assistant_retriever: Mock,
    ) -> None:
        """Invoke the configured retriever exactly once."""
        assistant_retriever.retrieve.return_value = []
        state = cast(
            AssistantState,
            {
                "search_query": "semantic search",
                "result_k": 3,
            },
        )

        retrieve_papers_node(state)

        assistant_retriever.retrieve.assert_called_once()

    def test_does_not_modify_existing_state(
        self,
        retrieve_papers_node: RetrievePapersNode,
        assistant_retriever: Mock,
    ) -> None:
        """Leave the input graph state unchanged."""
        assistant_retriever.retrieve.return_value = []
        state = cast(
            AssistantState,
            {
                "search_query": "finite volume methods",
                "result_k": 5,
                "retrieval_attempts": 2,
            },
        )

        retrieve_papers_node(state)

        assert state["search_query"] == "finite volume methods"
        assert state["result_k"] == 5
        assert state["retrieval_attempts"] == 2
        assert "retrieved_papers" not in state

    def test_propagates_retrieval_errors(
        self,
        retrieve_papers_node: RetrievePapersNode,
        assistant_retriever: Mock,
    ) -> None:
        """Propagate errors raised by the underlying retriever."""
        assistant_retriever.retrieve.side_effect = RuntimeError(
            "Paper retrieval failed."
        )
        state = cast(
            AssistantState,
            {
                "search_query": "finite volume methods",
                "result_k": 5,
            },
        )

        with pytest.raises(
            RuntimeError,
            match="Paper retrieval failed",
        ):
            retrieve_papers_node(state)
