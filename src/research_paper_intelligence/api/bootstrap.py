"""Defines the bootstrap functions for the API."""

from research_paper_intelligence.assistant.graph import create_assistant_graph
from research_paper_intelligence.assistant.retrieval import AssistantRetriever
from research_paper_intelligence.config import get_settings
from research_paper_intelligence.data.data_loader import load_data
from research_paper_intelligence.embeddings.encoder import get_model
from research_paper_intelligence.repositories.paper_repository import (
    PaperRepository,
)
from research_paper_intelligence.services.assistant_service import (
    ResearchAssistant,
)
from research_paper_intelligence.services.search_service import SearchService
from research_paper_intelligence.storage.faiss_index_io import load_faiss_index
from research_paper_intelligence.storage.tfidf_index_io import load_tfidf_index


def create_search_service() -> SearchService:
    """Create and configure the research-paper search service.

    Load the application settings, paper data, embedding model,
    FAISS index, TF-IDF vectorizer, and TF-IDF matrix. These shared
    resources are then passed to the SearchService constructor.

    This function should normally be called once during API startup.

    Returns:
        The configured search service.
    """
    settings = get_settings()

    paper_data = load_data(settings.processed_papers_path)
    paper_repository = PaperRepository(paper_data)
    model = get_model(settings.embedding_model_name)
    index = load_faiss_index(settings.faiss_index_papers_path)
    vectorizer, tfidf_matrix = load_tfidf_index(
        settings.tfidf_vectorizer_path, settings.tfidf_matrix_path
    )

    return SearchService(
        paper_repository=paper_repository,
        model=model,
        index=index,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        settings=settings,
    )


def create_assistant_service(
    search_service: SearchService,
) -> ResearchAssistant:
    """Create and configure the research-paper assistant service.

    Loads the application settings, builds the LangGraph state machine,
    and initializes the application service wrapper.

    This function should normally be called once during API startup.

    Args:
        search_service: The configured search service for retrieving papers.

    Returns:
        The configured research assistant service.
    """
    settings = get_settings()

    # Build and compile the LangGraph
    compiled_graph = create_assistant_graph(
        settings=settings,
        search_service=AssistantRetriever(search_service),
    )

    # Inject the graph into the application service class
    return ResearchAssistant(graph=compiled_graph)
