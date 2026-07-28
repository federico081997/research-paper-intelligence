"""Tests for the direct-answer LangGraph node."""

from typing import cast
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from research_paper_intelligence.assistant.models import DirectAnswer
from research_paper_intelligence.assistant.nodes.direct_answer_node import (
    GenerateDirectAnswerNode,
)
from research_paper_intelligence.assistant.prompts import (
    DIRECT_ANSWER_SYSTEM_PROMPT,
)
from research_paper_intelligence.assistant.state import AssistantState


@pytest.fixture
def direct_answer_model() -> Mock:
    """Create a mocked structured direct-answer model."""
    return Mock(spec=Runnable)


@pytest.fixture
def direct_answer_node(
    direct_answer_model: Mock,
) -> GenerateDirectAnswerNode:
    """Create a direct-answer node using a mocked model."""
    return GenerateDirectAnswerNode(
        direct_answer_model=direct_answer_model,
    )


class TestGenerateDirectAnswerNode:
    """Tests for the GenerateDirectAnswerNode class."""

    def test_returns_generated_final_answer(
        self,
        direct_answer_node: GenerateDirectAnswerNode,
        direct_answer_model: Mock,
    ) -> None:
        """Return the model answer as a final-answer state update."""
        direct_answer_model.invoke.return_value = DirectAnswer(
            final_answer="Hello. How can I help you?"
        )
        state = cast(
            AssistantState,
            {
                "messages": [
                    HumanMessage(content="Hello."),
                ],
            },
        )

        result = direct_answer_node(state)

        assert result == {
            "final_answer": "Hello. How can I help you?",
        }

    def test_prepends_direct_answer_system_prompt(
        self,
        direct_answer_node: GenerateDirectAnswerNode,
        direct_answer_model: Mock,
    ) -> None:
        """Prepend the direct-answer system prompt to the conversation."""
        conversation_messages = [
            HumanMessage(content="What can you do?"),
            AIMessage(content="I can help you search research papers."),
            HumanMessage(content="Thanks."),
        ]
        state = cast(
            AssistantState,
            {
                "messages": conversation_messages,
            },
        )
        direct_answer_model.invoke.return_value = DirectAnswer(
            final_answer="You're welcome."
        )

        direct_answer_node(state)

        invocation_messages = direct_answer_model.invoke.call_args.args[0]

        assert isinstance(invocation_messages[0], SystemMessage)
        assert invocation_messages[0].content == DIRECT_ANSWER_SYSTEM_PROMPT
        assert invocation_messages[1:] == conversation_messages

    def test_invokes_direct_answer_model_once(
        self,
        direct_answer_node: GenerateDirectAnswerNode,
        direct_answer_model: Mock,
    ) -> None:
        """Invoke the configured direct-answer model exactly once."""
        state = cast(
            AssistantState,
            {
                "messages": [
                    HumanMessage(content="Hello."),
                ],
            },
        )
        direct_answer_model.invoke.return_value = DirectAnswer(
            final_answer="Hello."
        )

        direct_answer_node(state)

        direct_answer_model.invoke.assert_called_once()

    def test_handles_empty_message_history(
        self,
        direct_answer_node: GenerateDirectAnswerNode,
        direct_answer_model: Mock,
    ) -> None:
        """Invoke the model with the system prompt when history is empty."""
        state = cast(
            AssistantState,
            {
                "messages": [],
            },
        )
        direct_answer_model.invoke.return_value = DirectAnswer(
            final_answer="How can I help you?"
        )

        result = direct_answer_node(state)

        invocation_messages = direct_answer_model.invoke.call_args.args[0]

        assert len(invocation_messages) == 1
        assert isinstance(invocation_messages[0], SystemMessage)
        assert invocation_messages[0].content == DIRECT_ANSWER_SYSTEM_PROMPT
        assert result == {
            "final_answer": "How can I help you?",
        }

    def test_does_not_modify_existing_message_history(
        self,
        direct_answer_node: GenerateDirectAnswerNode,
        direct_answer_model: Mock,
    ) -> None:
        """Leave the message collection stored in state unchanged."""
        conversation_messages = [
            HumanMessage(content="Explain what this assistant does."),
        ]
        original_messages = list(conversation_messages)
        state = cast(
            AssistantState,
            {
                "messages": conversation_messages,
            },
        )
        direct_answer_model.invoke.return_value = DirectAnswer(
            final_answer="I help users discover research papers."
        )

        direct_answer_node(state)

        assert state["messages"] == original_messages
        assert state["messages"] is conversation_messages
