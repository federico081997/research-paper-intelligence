"""Tests for the final-response LangGraph node."""

from typing import cast

from langchain_core.messages import AIMessage, HumanMessage

from research_paper_intelligence.assistant.nodes import (
    FinalizeResponseNode,
)
from research_paper_intelligence.assistant.state import AssistantState


class TestFinalizeResponseNode:
    """Tests for the FinalizeResponseNode class."""

    def test_returns_final_answer_as_ai_message(self) -> None:
        """Return the final answer wrapped in an AI message."""
        node = FinalizeResponseNode()
        state = cast(
            AssistantState,
            {
                "final_answer": "This is the final response.",
            },
        )

        result = node(state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "This is the final response."

    def test_returns_only_messages_state_update(self) -> None:
        """Return only the messages field in the state update."""
        node = FinalizeResponseNode()
        state = cast(
            AssistantState,
            {
                "final_answer": "Final answer.",
            },
        )

        result = node(state)

        assert set(result) == {"messages"}

    def test_creates_new_ai_message_for_each_invocation(self) -> None:
        """Create a distinct AI message for every node invocation."""
        node = FinalizeResponseNode()
        state = cast(
            AssistantState,
            {
                "final_answer": "Final answer.",
            },
        )

        first_result = node(state)
        second_result = node(state)

        assert first_result["messages"][0] is not second_result["messages"][0]
        assert first_result["messages"] == second_result["messages"]

    def test_does_not_modify_existing_message_history(self) -> None:
        """Leave the existing conversation history unchanged."""
        node = FinalizeResponseNode()
        existing_messages = [
            HumanMessage(content="Explain finite volume methods."),
        ]
        state = cast(
            AssistantState,
            {
                "messages": existing_messages,
                "final_answer": "Finite volume methods conserve fluxes.",
            },
        )

        node(state)

        assert state["messages"] is existing_messages
        assert state["messages"] == [
            HumanMessage(
                content="Explain finite volume methods.",
            )
        ]

    def test_preserves_markdown_content(self) -> None:
        """Preserve Markdown formatting in the final answer."""
        node = FinalizeResponseNode()
        final_answer = (
            "## Findings\n\n"
            "- Finite volume methods conserve fluxes.\n"
            "- They operate on control volumes.\n\n"
            "## References\n\n"
            "[1] Example paper."
        )
        state = cast(
            AssistantState,
            {
                "final_answer": final_answer,
            },
        )

        result = node(state)

        assert result["messages"][0].content == final_answer
