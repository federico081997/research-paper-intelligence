"""State models for the research-assistant graph."""

from typing import Literal

from langgraph.graph import MessagesState

from research_paper_intelligence.assistant.retrieval import RetrievedPaper


class AssistantState(MessagesState, total=False):
    """State shared between research-assistant graph nodes."""

    # Initial request
    original_query: str

    # Retrieval planning
    search_query: str
    request_type: Literal["direct", "retrieval"]
    result_k: int

    # Retrieved evidence
    retrieved_papers: list[RetrievedPaper]
    retrieval_attempts: int

    # Retrieval evaluation
    retrieval_sufficient: bool
    retrieval_feedback: str

    # Retry control
    rewrite_count: int

    # Final output
    final_answer: str
