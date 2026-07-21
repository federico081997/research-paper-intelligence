"""Defines the search service class used by the application."""

import faiss
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from research_paper_intelligence.config import Settings
from research_paper_intelligence.domain.search_result import SearchResult
from research_paper_intelligence.ranking.hybrid_ranking import hybrid_search
from research_paper_intelligence.repositories.paper_repository import (
    PaperRepository,
)


class SearchService:
    """Coordinates paper retrieval and hybrid ranking."""

    RETRIEVAL_STRATEGY = "Hybrid"
    RANKING_COMPONENTS = [
        "Semantic similarity",
        "TF-IDF similarity",
        "Keyword overlap",
        "Publication recency",
    ]

    def __init__(
        self,
        paper_repository: PaperRepository,
        model: SentenceTransformer,
        index: faiss.Index,
        vectorizer: TfidfVectorizer,
        tfidf_matrix: csr_matrix,
        settings: Settings,
    ) -> None:
        """Initializes the search service."""
        self.paper_repository = paper_repository
        self.model = model
        self.index = index
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.settings = settings

    def search(
        self,
        query: str,
        result_k: int,
    ) -> list[SearchResult]:
        """Performs a hybrid search for papers based on the given query."""
        return hybrid_search(
            query=query,
            paper_repository=self.paper_repository,
            model=self.model,
            index=self.index,
            vectorizer=self.vectorizer,
            tfidf_matrix=self.tfidf_matrix,
            candidate_k=self.settings.candidate_top_k,
            result_k=result_k,
            semantic_weight=self.settings.semantic_weight,
            tfidf_weight=self.settings.tfidf_weight,
            keyword_weight=self.settings.keyword_weight,
            recency_weight=self.settings.recency_weight,
            half_life_years=self.settings.half_life_years,
        )

    @property
    def paper_count(self) -> int:
        """Retrieves the total number of papers in the repository."""
        return len(self.paper_repository)

    @property
    def embedding_model(self) -> str:
        """Retrieves the embedding model used for paper embeddings."""
        return self.settings.embedding_model_name

    @property
    def retrieval_strategy(self) -> str:
        """Retrieves the retrieval strategy used for paper search."""
        return self.RETRIEVAL_STRATEGY

    @property
    def ranking_components(self) -> list[str]:
        """Retrieves the ranking components used for paper search."""
        return self.RANKING_COMPONENTS

    @property
    def faiss_index_type(self) -> str:
        """Retrieves the type of FAISS index used for paper search."""
        return type(self.index).__name__

    @property
    def faiss_index_size(self) -> int:
        """Retrieves the size of the FAISS index used for paper search."""
        return int(self.index.ntotal)

    @property
    def tfidf_document_count(self) -> int:
        """Retrieves the number of documents used for TF-IDF vectorization."""
        return int(self.tfidf_matrix.shape[0])

    @property
    def tfidf_vocabulary_size(self) -> int:
        """Retrieves the size of the vocab used for TF-IDF vectorization."""
        return len(self.vectorizer.vocabulary_)
