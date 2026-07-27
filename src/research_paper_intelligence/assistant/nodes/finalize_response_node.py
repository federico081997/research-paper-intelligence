"""LangGraph finalize response node class."""

from typing import TypedDict

from langchain_core.messages import AIMessage

from research_paper_intelligence.assistant.state import AssistantState


class FinalizeResponseUpdate(TypedDict):
    """State fields updated by the final-response node."""

    messages: list[AIMessage]


class FinalizeResponseNode:
    """Add the final answer to conversational message history."""

    def __call__(self, state: AssistantState) -> FinalizeResponseUpdate:
        """Add the final answer to the conversation history in the state.

        Args:
            state: State shared by the assistant.

        Returns:
            State update for messages field.
        """
        return {"messages": [AIMessage(content=state["final_answer"])]}
