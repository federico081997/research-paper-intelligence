"""Construction of the research-assistant graph."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph

from research_paper_intelligence.assistant.llm import (
    create_chat_model,
    create_structured_model,
)
from research_paper_intelligence.assistant.models import (
    DirectAnswer,
    PlanRequest,
    ResearchAnswer,
    RetrievalGrader,
    RewrittenQuery,
)
from research_paper_intelligence.assistant.nodes import (
    FinalizeResponseNode,
    GenerateDirectAnswerNode,
    GenerateGroundedAnswerNode,
    GenerateLimitedAnswerNode,
    GradeRetrievalNode,
    PlannerNode,
    RetrievePapersNode,
    RewriteQueryNode,
)
from research_paper_intelligence.assistant.retrieval import AssistantRetriever
from research_paper_intelligence.assistant.routing import (
    route_after_planning,
    route_after_retrieval,
)
from research_paper_intelligence.assistant.state import AssistantState
from research_paper_intelligence.config import Settings


def create_assistant_graph(
    settings: Settings,
    search_service: AssistantRetriever,
) -> CompiledStateGraph:
    """Create and compile the research-assistant graph.

    Args:
        settings: Settings used to create the research-assistant graph.
        search_service: Search service used to retrieve papers.

    Returns:
        CompiledStateGraph: Compiled research-assistant graph.
    """
    chat_model = create_chat_model(settings)

    # ------------------------------------------------------------------------#
    # Initialize graph nodes -------------------------------------------------
    # ------------------------------------------------------------------------#

    planner_node = PlannerNode(
        planner_model=create_structured_model(
            chat_model=chat_model,
            output_schema=PlanRequest,
        )
    )

    direct_answer_node = GenerateDirectAnswerNode(
        direct_answer_model=create_structured_model(
            chat_model=chat_model,
            output_schema=DirectAnswer,
        )
    )

    retrieve_papers_node = RetrievePapersNode(
        search_service=search_service,
    )

    grade_retrieval_node = GradeRetrievalNode(
        retrieval_grader_model=create_structured_model(
            chat_model=chat_model,
            output_schema=RetrievalGrader,
        )
    )

    rewrite_query_node = RewriteQueryNode(
        query_rewriter_model=create_structured_model(
            chat_model=chat_model,
            output_schema=RewrittenQuery,
        )
    )

    grounded_answer_node = GenerateGroundedAnswerNode(
        answer_model=create_structured_model(
            chat_model=chat_model,
            output_schema=ResearchAnswer,
        )
    )

    limited_answer_node = GenerateLimitedAnswerNode(
        limited_answer_model=create_structured_model(
            chat_model=chat_model,
            output_schema=ResearchAnswer,
        )
    )

    finalize_response_node = FinalizeResponseNode()

    # ------------------------------------------------------------------------#
    # Build the LangChain graph -----------------------------------------------
    # ------------------------------------------------------------------------#

    builder = StateGraph(AssistantState)
    checkpointer = InMemorySaver()

    builder.add_node("plan_request", planner_node)
    builder.add_node("generate_direct_answer", direct_answer_node)
    builder.add_node("retrieve_papers", retrieve_papers_node)
    builder.add_node("grade_retrieval", grade_retrieval_node)
    builder.add_node("rewrite_query", rewrite_query_node)
    builder.add_node("generate_grounded_answer", grounded_answer_node)
    builder.add_node("generate_limited_answer", limited_answer_node)
    builder.add_node("finalize_response", finalize_response_node)

    builder.add_edge(START, "plan_request")
    builder.add_conditional_edges("plan_request", route_after_planning)
    builder.add_edge("retrieve_papers", "grade_retrieval")
    builder.add_conditional_edges("grade_retrieval", route_after_retrieval)
    builder.add_edge("rewrite_query", "retrieve_papers")
    builder.add_edge("generate_direct_answer", "finalize_response")
    builder.add_edge("generate_grounded_answer", "finalize_response")
    builder.add_edge("generate_limited_answer", "finalize_response")
    builder.add_edge("finalize_response", END)

    return builder.compile(checkpointer=checkpointer)
