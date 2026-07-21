"""Defines the bootstrap functions for the API."""

from research_paper_intelligence.config import get_settings
from research_paper_intelligence.data.data_loader import load_data
from research_paper_intelligence.embeddings.encoder import get_model
from research_paper_intelligence.repositories.paper_repository import (
    PaperRepository,
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
