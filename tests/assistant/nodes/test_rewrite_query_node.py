"""Tests for the rewrite-query LangGraph node."""

from typing import cast
from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from research_paper_intelligence.assistant.models import RewrittenQuery
from research_paper_intelligence.assistant.nodes.rewrite_query_node import (
    RewriteQueryNode,
)
from research_paper_intelligence.assistant.prompts import (
    REWRITE_QUERY_SYSTEM_PROMPT,
)
from research_paper_intelligence.assistant.state import AssistantState


@pytest.fixture
def query_rewriter_model() -> Mock:
    """Create a mocked structured query-rewriting model."""
    return Mock(spec=Runnable)


@pytest.fixture
def rewrite_query_node(
    query_rewriter_model: Mock,
) -> RewriteQueryNode:
    """Create a rewrite-query node using a mocked model."""
    return RewriteQueryNode(
        query_rewriter_model=query_rewriter_model,
    )


class TestRewriteQueryNode:
    """Tests for the RewriteQueryNode class."""

    def test_returns_rewritten_search_query(
        self,
        rewrite_query_node: RewriteQueryNode,
        query_rewriter_model: Mock,
    ) -> None:
        """Return the rewritten query produced by the model."""
        query_rewriter_model.invoke.return_value = RewrittenQuery(
            search_query=(
                "block-coupled finite volume elastoplasticity solid mechanics"
            ),
        )
        state = cast(
            AssistantState,
            {
                "original_query": (
                    "Find papers about finite volume elastoplasticity."
                ),
                "search_query": "finite volume methods",
                "retrieval_feedback": (
                    "The retrieved papers were too broad and did not "
                    "focus on elastoplastic solid mechanics."
                ),
            },
        )

        result = rewrite_query_node(state)

        assert result["search_query"] == (
            "block-coupled finite volume elastoplasticity solid mechanics"
        )

    def test_strips_whitespace_from_rewritten_query(
        self,
        rewrite_query_node: RewriteQueryNode,
        query_rewriter_model: Mock,
    ) -> None:
        """Remove leading and trailing whitespace from the rewritten query."""
        query_rewriter_model.invoke.return_value = RewrittenQuery(
            search_query="  neural operators partial differential equations  ",
        )
        state = cast(
            AssistantState,
            {
                "original_query": "Find papers about neural operators.",
                "search_query": "neural networks",
                "retrieval_feedback": (
                    "The query retrieved generic neural-network papers."
                ),
            },
        )

        result = rewrite_query_node(state)

        assert result["search_query"] == (
            "neural operators partial differential equations"
        )

    def test_sets_first_rewrite_count_to_one(
        self,
        rewrite_query_node: RewriteQueryNode,
        query_rewriter_model: Mock,
    ) -> None:
        """Set rewrite count to one when no previous count exists."""
        query_rewriter_model.invoke.return_value = RewrittenQuery(
            search_query="scientific machine learning differential equations",
        )
        state = cast(
            AssistantState,
            {
                "original_query": (
                    "Find papers about machine learning for PDEs."
                ),
                "search_query": "machine learning",
                "retrieval_feedback": (
                    "The retrieved papers were not specific to PDEs."
                ),
            },
        )

        result = rewrite_query_node(state)

        assert result["rewrite_count"] == 1

    @pytest.mark.parametrize(
        ("current_count", "expected_count"),
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (5, 6),
        ],
    )
    def test_increments_existing_rewrite_count(
        self,
        rewrite_query_node: RewriteQueryNode,
        query_rewriter_model: Mock,
        current_count: int,
        expected_count: int,
    ) -> None:
        """Increment the number of completed query rewrites."""
        query_rewriter_model.invoke.return_value = RewrittenQuery(
            search_query="finite volume computational solid mechanics",
        )
        state = cast(
            AssistantState,
            {
                "original_query": (
                    "Find papers about finite volume solid mechanics."
                ),
                "search_query": "finite volume methods",
                "retrieval_feedback": (
                    "The previous query retrieved unrelated fluid papers."
                ),
                "rewrite_count": current_count,
            },
        )

        result = rewrite_query_node(state)

        assert result["rewrite_count"] == expected_count

    def test_invokes_model_with_system_and_human_messages(
        self,
        rewrite_query_node: RewriteQueryNode,
        query_rewriter_model: Mock,
    ) -> None:
        """Invoke the model with the expected message sequence."""
        query_rewriter_model.invoke.return_value = RewrittenQuery(
            search_query="finite volume elastoplasticity",
        )
        state = cast(
            AssistantState,
            {
                "original_query": (
                    "Find research about finite volume elastoplasticity."
                ),
                "search_query": "finite volume methods",
                "retrieval_feedback": (
                    "The results did not address elastoplasticity."
                ),
            },
        )

        rewrite_query_node(state)

        messages = query_rewriter_model.invoke.call_args.args[0]

        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert messages[0].content == REWRITE_QUERY_SYSTEM_PROMPT

    def test_includes_retrieval_context_in_rewrite_request(
        self,
        rewrite_query_node: RewriteQueryNode,
        query_rewriter_model: Mock,
    ) -> None:
        """Include the original request, previous query, and feedback."""
        original_query = "Compare finite volume and finite element methods."
        previous_query = "finite volume methods"
        retrieval_feedback = (
            "The results discuss finite volume methods but do not include "
            "finite element comparisons."
        )
        query_rewriter_model.invoke.return_value = RewrittenQuery(
            search_query=(
                "finite volume versus finite element numerical methods"
            ),
        )
        state = cast(
            AssistantState,
            {
                "original_query": original_query,
                "search_query": previous_query,
                "retrieval_feedback": retrieval_feedback,
            },
        )

        rewrite_query_node(state)

        messages = query_rewriter_model.invoke.call_args.args[0]
        rewrite_request = messages[1].content

        assert isinstance(rewrite_request, str)
        assert f"Original user request:\n{original_query}" in rewrite_request
        assert f"Previous search query:\n{previous_query}" in rewrite_request
        assert f"Retrieval feedback:\n{retrieval_feedback}" in rewrite_request

    def test_invokes_query_rewriter_model_once(
        self,
        rewrite_query_node: RewriteQueryNode,
        query_rewriter_model: Mock,
    ) -> None:
        """Invoke the configured query-rewriting model exactly once."""
        query_rewriter_model.invoke.return_value = RewrittenQuery(
            search_query="semantic search scientific literature",
        )
        state = cast(
            AssistantState,
            {
                "original_query": (
                    "Find papers about semantic scientific search."
                ),
                "search_query": "semantic search",
                "retrieval_feedback": (
                    "The results were not specific to scientific literature."
                ),
            },
        )

        rewrite_query_node(state)

        query_rewriter_model.invoke.assert_called_once()

    def test_returns_only_rewrite_state_fields(
        self,
        rewrite_query_node: RewriteQueryNode,
        query_rewriter_model: Mock,
    ) -> None:
        """Return only fields managed by the rewrite-query node."""
        query_rewriter_model.invoke.return_value = RewrittenQuery(
            search_query="physics-informed neural networks",
        )
        state = cast(
            AssistantState,
            {
                "original_query": (
                    "Find papers about physics-informed learning."
                ),
                "search_query": "machine learning physics",
                "retrieval_feedback": (
                    "The results were broader than the user request."
                ),
            },
        )

        result = rewrite_query_node(state)

        assert set(result) == {
            "search_query",
            "rewrite_count",
        }

    def test_does_not_modify_existing_state(
        self,
        rewrite_query_node: RewriteQueryNode,
        query_rewriter_model: Mock,
    ) -> None:
        """Leave the input graph state unchanged."""
        query_rewriter_model.invoke.return_value = RewrittenQuery(
            search_query="rewritten scientific query",
        )
        state = cast(
            AssistantState,
            {
                "original_query": "Find relevant scientific papers.",
                "search_query": "original search query",
                "retrieval_feedback": "The results were insufficient.",
                "rewrite_count": 2,
            },
        )

        rewrite_query_node(state)

        assert state["search_query"] == "original search query"
        assert state["rewrite_count"] == 2
        assert state["retrieval_feedback"] == (
            "The results were insufficient."
        )

    def test_propagates_model_errors(
        self,
        rewrite_query_node: RewriteQueryNode,
        query_rewriter_model: Mock,
    ) -> None:
        """Propagate errors raised by the query-rewriting model."""
        query_rewriter_model.invoke.side_effect = RuntimeError(
            "Query rewriting failed."
        )
        state = cast(
            AssistantState,
            {
                "original_query": "Find papers about neural operators.",
                "search_query": "neural networks",
                "retrieval_feedback": (
                    "The results did not discuss neural operators."
                ),
                "rewrite_count": 1,
            },
        )

        with pytest.raises(
            RuntimeError,
            match="Query rewriting failed",
        ):
            rewrite_query_node(state)
