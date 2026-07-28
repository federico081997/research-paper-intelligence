"""Tests for the research-assistant application service."""

from typing import cast
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from research_paper_intelligence.services.assistant_service import (
    ResearchAssistant,
)


@pytest.fixture
def compiled_graph() -> Mock:
    """Create a mocked compiled LangGraph workflow."""
    return Mock(spec=CompiledStateGraph)


@pytest.fixture
def research_assistant(
    compiled_graph: Mock,
) -> ResearchAssistant:
    """Create a research assistant using a mocked graph."""
    return ResearchAssistant(
        graph=cast(CompiledStateGraph, compiled_graph),
    )


class TestResearchAssistantChat:
    """Tests for the ResearchAssistant.chat method."""

    def test_returns_final_graph_message(
        self,
        research_assistant: ResearchAssistant,
        compiled_graph: Mock,
    ) -> None:
        """Return the content of the final message produced by the graph."""
        compiled_graph.invoke.return_value = {
            "messages": [
                HumanMessage(content="Explain finite volume methods."),
                AIMessage(
                    content=(
                        "Finite volume methods discretize conservation laws."
                    )
                ),
            ]
        }

        result = research_assistant.chat(
            user_query="Explain finite volume methods.",
            thread_id="thread-001",
        )

        assert result == (
            "Finite volume methods discretize conservation laws."
        )

    def test_invokes_graph_with_initial_conversation_state(
        self,
        research_assistant: ResearchAssistant,
        compiled_graph: Mock,
    ) -> None:
        """Invoke the graph with the user message and reset counters."""
        compiled_graph.invoke.return_value = {
            "messages": [
                AIMessage(content="Generated answer."),
            ]
        }

        research_assistant.chat(
            user_query="Find papers about neural operators.",
            thread_id="thread-001",
        )

        compiled_graph.invoke.assert_called_once()

        invocation = compiled_graph.invoke.call_args
        graph_input = invocation.kwargs["input"]

        assert graph_input["original_query"] == (
            "Find papers about neural operators."
        )
        assert graph_input["retrieval_attempts"] == 0
        assert graph_input["rewrite_count"] == 0

        assert len(graph_input["messages"]) == 1
        assert isinstance(graph_input["messages"][0], HumanMessage)
        assert graph_input["messages"][0].content == (
            "Find papers about neural operators."
        )

    def test_passes_thread_id_in_runnable_configuration(
        self,
        research_assistant: ResearchAssistant,
        compiled_graph: Mock,
    ) -> None:
        """Pass the conversation thread ID to LangGraph configuration."""
        compiled_graph.invoke.return_value = {
            "messages": [
                AIMessage(content="Generated answer."),
            ]
        }

        research_assistant.chat(
            user_query="Explain semantic search.",
            thread_id="conversation-123",
        )

        config = compiled_graph.invoke.call_args.kwargs["config"]

        assert config == {
            "configurable": {
                "thread_id": "conversation-123",
            }
        }

    def test_uses_last_message_when_graph_returns_history(
        self,
        research_assistant: ResearchAssistant,
        compiled_graph: Mock,
    ) -> None:
        """Return the last message rather than an earlier graph message."""
        compiled_graph.invoke.return_value = {
            "messages": [
                HumanMessage(content="Original request."),
                AIMessage(content="Intermediate answer."),
                HumanMessage(content="Follow-up request."),
                AIMessage(content="Final answer."),
            ]
        }

        result = research_assistant.chat(
            user_query="Follow-up request.",
            thread_id="thread-001",
        )

        assert result == "Final answer."

    def test_resets_workflow_counters_for_each_message(
        self,
        research_assistant: ResearchAssistant,
        compiled_graph: Mock,
    ) -> None:
        """Reset retrieval and rewrite counters for every graph execution."""
        compiled_graph.invoke.side_effect = [
            {
                "messages": [
                    AIMessage(content="First answer."),
                ]
            },
            {
                "messages": [
                    AIMessage(content="Second answer."),
                ]
            },
        ]

        research_assistant.chat(
            user_query="First request.",
            thread_id="thread-001",
        )
        research_assistant.chat(
            user_query="Second request.",
            thread_id="thread-001",
        )

        assert compiled_graph.invoke.call_count == 2

        first_input = compiled_graph.invoke.call_args_list[0].kwargs["input"]
        second_input = compiled_graph.invoke.call_args_list[1].kwargs["input"]

        assert first_input["retrieval_attempts"] == 0
        assert first_input["rewrite_count"] == 0
        assert second_input["retrieval_attempts"] == 0
        assert second_input["rewrite_count"] == 0

    def test_uses_supplied_query_for_message_and_original_query(
        self,
        research_assistant: ResearchAssistant,
        compiled_graph: Mock,
    ) -> None:
        """Use the same query as the message and original workflow request."""
        user_query = "Find five papers about finite volume elastoplasticity."
        compiled_graph.invoke.return_value = {
            "messages": [
                AIMessage(content="Generated answer."),
            ]
        }

        research_assistant.chat(
            user_query=user_query,
            thread_id="thread-001",
        )

        graph_input = compiled_graph.invoke.call_args.kwargs["input"]

        assert graph_input["original_query"] == user_query
        assert graph_input["messages"][0].content == user_query

    def test_creates_new_human_message_for_each_invocation(
        self,
        research_assistant: ResearchAssistant,
        compiled_graph: Mock,
    ) -> None:
        """Create a separate HumanMessage for each user request."""
        compiled_graph.invoke.side_effect = [
            {
                "messages": [
                    AIMessage(content="First answer."),
                ]
            },
            {
                "messages": [
                    AIMessage(content="Second answer."),
                ]
            },
        ]

        research_assistant.chat(
            user_query="First query.",
            thread_id="thread-001",
        )
        research_assistant.chat(
            user_query="Second query.",
            thread_id="thread-001",
        )

        first_message = compiled_graph.invoke.call_args_list[0].kwargs[
            "input"
        ]["messages"][0]
        second_message = compiled_graph.invoke.call_args_list[1].kwargs[
            "input"
        ]["messages"][0]

        assert first_message is not second_message
        assert first_message.content == "First query."
        assert second_message.content == "Second query."

    def test_invokes_graph_once_per_chat_request(
        self,
        research_assistant: ResearchAssistant,
        compiled_graph: Mock,
    ) -> None:
        """Invoke the compiled graph exactly once for one chat request."""
        compiled_graph.invoke.return_value = {
            "messages": [
                AIMessage(content="Generated answer."),
            ]
        }

        research_assistant.chat(
            user_query="Explain machine learning.",
            thread_id="thread-001",
        )

        compiled_graph.invoke.assert_called_once()

    def test_propagates_graph_execution_error(
        self,
        research_assistant: ResearchAssistant,
        compiled_graph: Mock,
    ) -> None:
        """Propagate errors raised while executing the assistant graph."""
        compiled_graph.invoke.side_effect = RuntimeError(
            "Assistant graph execution failed."
        )

        with pytest.raises(
            RuntimeError,
            match="Assistant graph execution failed",
        ):
            research_assistant.chat(
                user_query="Explain neural operators.",
                thread_id="thread-001",
            )

    def test_raises_key_error_when_graph_returns_no_messages_field(
        self,
        research_assistant: ResearchAssistant,
        compiled_graph: Mock,
    ) -> None:
        """Raise a KeyError when the graph result has no message history."""
        compiled_graph.invoke.return_value = {
            "final_answer": "Generated answer.",
        }

        with pytest.raises(KeyError, match="messages"):
            research_assistant.chat(
                user_query="Explain semantic retrieval.",
                thread_id="thread-001",
            )

    def test_raises_index_error_when_graph_returns_empty_messages(
        self,
        research_assistant: ResearchAssistant,
        compiled_graph: Mock,
    ) -> None:
        """Raise an IndexError when the graph returns an empty history."""
        compiled_graph.invoke.return_value = {
            "messages": [],
        }

        with pytest.raises(IndexError):
            research_assistant.chat(
                user_query="Explain semantic retrieval.",
                thread_id="thread-001",
            )
