"""Research assistant service of the application."""

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph


class ResearchAssistant:
    """Application service for executing assistant conversations."""

    def __init__(self, graph: CompiledStateGraph) -> None:
        """Initialise ResearchAssistant object.

        Args:
            graph: The compiled graph object.
        """
        self._graph = graph

    def chat(self, user_query: str, thread_id: str) -> str:
        """Process one user message and return the final answer.

        Args:
            user_query: The user query.
            thread_id: The conversation thread id.

        Returns:
            The final answer returned to the user.
        """
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        result = self._graph.invoke(
            input={
                "messages": [HumanMessage(content=user_query)],
                "original_query": user_query,
                "retrieval_attempts": 0,
                "rewrite_count": 0,
            },
            config=config,
        )

        return str(result["messages"][-1].content)
