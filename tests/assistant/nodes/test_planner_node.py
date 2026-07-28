"""Tests for the planner LangGraph node."""

from typing import cast
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from research_paper_intelligence.assistant.models import PlanRequest
from research_paper_intelligence.assistant.nodes.planner_node import (
    PlannerNode,
)
from research_paper_intelligence.assistant.prompts import (
    PLAN_REQUEST_SYSTEM_PROMPT,
)
from research_paper_intelligence.assistant.state import AssistantState


@pytest.fixture
def planner_model() -> Mock:
    """Create a mocked structured planner model."""
    return Mock(spec=Runnable)


@pytest.fixture
def planner_node(
    planner_model: Mock,
) -> PlannerNode:
    """Create a planner node using a mocked model."""
    return PlannerNode(planner_model=planner_model)


class TestPlannerNode:
    """Tests for the PlannerNode class."""

    def test_returns_direct_request_plan(
        self,
        planner_node: PlannerNode,
        planner_model: Mock,
    ) -> None:
        """Return a direct execution plan from the planner model."""
        planner_model.invoke.return_value = PlanRequest(
            request_type="direct",
            search_query="",
            result_k=0,
        )
        state = cast(
            AssistantState,
            {
                "messages": [
                    HumanMessage(content="Hello."),
                ],
            },
        )

        result = planner_node(state)

        assert result == {
            "request_type": "direct",
            "search_query": "",
            "result_k": 0,
        }

    @pytest.mark.parametrize("result_k", [1, 5, 10])
    def test_returns_retrieval_request_plan(
        self,
        planner_node: PlannerNode,
        planner_model: Mock,
        result_k: int,
    ) -> None:
        """Return a retrieval plan containing query and result count."""
        planner_model.invoke.return_value = PlanRequest(
            request_type="retrieval",
            search_query="finite volume elastoplasticity",
            result_k=result_k,
        )
        state = cast(
            AssistantState,
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Find papers about finite volume elastoplasticity."
                        )
                    ),
                ],
            },
        )

        result = planner_node(state)

        assert result == {
            "request_type": "retrieval",
            "search_query": "finite volume elastoplasticity",
            "result_k": result_k,
        }

    def test_prepends_planner_system_prompt(
        self,
        planner_node: PlannerNode,
        planner_model: Mock,
    ) -> None:
        """Prepend the planner system prompt to the conversation."""
        conversation_messages = [
            HumanMessage(content="Find papers about finite volume methods."),
        ]
        planner_model.invoke.return_value = PlanRequest(
            request_type="retrieval",
            search_query="finite volume methods",
            result_k=5,
        )
        state = cast(
            AssistantState,
            {
                "messages": conversation_messages,
            },
        )

        planner_node(state)

        messages = planner_model.invoke.call_args.args[0]

        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == PLAN_REQUEST_SYSTEM_PROMPT
        assert messages[1:] == conversation_messages

    def test_preserves_conversation_message_order(
        self,
        planner_node: PlannerNode,
        planner_model: Mock,
    ) -> None:
        """Preserve the order of existing conversation messages."""
        conversation_messages = [
            HumanMessage(content="Find papers about finite volume methods."),
            AIMessage(content="Which aspect would you like to investigate?"),
            HumanMessage(content="Their application to elastoplasticity."),
        ]
        planner_model.invoke.return_value = PlanRequest(
            request_type="retrieval",
            search_query=(
                "finite volume methods elastoplasticity applications"
            ),
            result_k=5,
        )
        state = cast(
            AssistantState,
            {
                "messages": conversation_messages,
            },
        )

        planner_node(state)

        messages = planner_model.invoke.call_args.args[0]

        assert messages[1:] == conversation_messages

    def test_invokes_planner_model_once(
        self,
        planner_node: PlannerNode,
        planner_model: Mock,
    ) -> None:
        """Invoke the configured planner model exactly once."""
        planner_model.invoke.return_value = PlanRequest(
            request_type="direct",
            search_query="",
            result_k=0,
        )
        state = cast(
            AssistantState,
            {
                "messages": [
                    HumanMessage(content="What can you do?"),
                ],
            },
        )

        planner_node(state)

        planner_model.invoke.assert_called_once()

    def test_handles_empty_message_history(
        self,
        planner_node: PlannerNode,
        planner_model: Mock,
    ) -> None:
        """Invoke the planner with the system prompt when history is empty."""
        planner_model.invoke.return_value = PlanRequest(
            request_type="direct",
            search_query="",
            result_k=0,
        )
        state = cast(
            AssistantState,
            {
                "messages": [],
            },
        )

        result = planner_node(state)

        messages = planner_model.invoke.call_args.args[0]

        assert len(messages) == 1
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == PLAN_REQUEST_SYSTEM_PROMPT
        assert result == {
            "request_type": "direct",
            "search_query": "",
            "result_k": 0,
        }

    def test_returns_only_planner_state_fields(
        self,
        planner_node: PlannerNode,
        planner_model: Mock,
    ) -> None:
        """Return only fields managed by the planner node."""
        planner_model.invoke.return_value = PlanRequest(
            request_type="retrieval",
            search_query="semantic search scientific papers",
            result_k=7,
        )
        state = cast(
            AssistantState,
            {
                "messages": [
                    HumanMessage(
                        content="Find seven papers about semantic search."
                    ),
                ],
            },
        )

        result = planner_node(state)

        assert set(result) == {
            "request_type",
            "search_query",
            "result_k",
        }

    def test_does_not_modify_existing_message_history(
        self,
        planner_node: PlannerNode,
        planner_model: Mock,
    ) -> None:
        """Leave the conversation messages stored in state unchanged."""
        conversation_messages = [
            HumanMessage(content="Find papers about neural operators."),
        ]
        original_messages = list(conversation_messages)
        state = cast(
            AssistantState,
            {
                "messages": conversation_messages,
            },
        )
        planner_model.invoke.return_value = PlanRequest(
            request_type="retrieval",
            search_query="neural operators",
            result_k=5,
        )

        planner_node(state)

        assert state["messages"] is conversation_messages
        assert state["messages"] == original_messages
