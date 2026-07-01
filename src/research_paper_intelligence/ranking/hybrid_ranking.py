"""Utilities to perform the hybrid search on the papers."""

import faiss
import numpy as np
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from research_paper_intelligence.domain.search_result import SearchResult
from research_paper_intelligence.ranking.keyword_overlap_scores import (
    calculate_keyword_overlap_scores,
)
from research_paper_intelligence.ranking.recency_scores import (
    calculate_recency_scores,
)
from research_paper_intelligence.ranking.tfidf_scores import (
    calculate_tfidf_scores,
)
from research_paper_intelligence.repositories.paper_repository import (
    PaperRepository,
)
from research_paper_intelligence.retrieval.semantic_search import (
    semantic_search,
)


def build_explanation(
    semantic_score: float,
    keyword_overlap_score: float,
    tfidf_score: float,
    recency_score: float,
) -> str:
    """Generates a human-readable explanation for a recommendation.

    Args:
        semantic_score: Semantic score.
        keyword_overlap_score: Keyword overlap score.
        tfidf_score: TF-IDF score.
        recency_score: Recency score.

    Returns:
        explanation: A string containing the explanation for the hybrid
            ranking scores.
    """
    reasons: list[str] = []

    # Add a reason based on semantic similarity strength.
    if semantic_score >= 0.75:
        reasons.append("very strong semantic similarity")
    elif semantic_score >= 0.65:
        reasons.append("strong topical similarity")
    elif semantic_score >= 0.50:
        reasons.append("moderate semantic similarity")

    # Add a reason based on keyword overlap.
    if keyword_overlap_score >= 0.50:
        reasons.append("clear overlap in technical keywords")
    elif keyword_overlap_score > 0.0:
        reasons.append("some overlap in technical terminology")

    # Add a reason for TF-IDF scores
    if tfidf_score >= 0.75:
        reasons.append("a high TF-IDF score")
    elif tfidf_score >= 0.50:
        reasons.append("a moderate TF-IDF score")

    # Add a reason if the paper is relatively recent.
    if recency_score >= 0.75:
        reasons.append("relatively recent publication")

    # Explanation if no specific reason crosses the thresholds.
    if not reasons:
        return "Recommended based on the overall embedding similarity pattern."

    if len(reasons) == 1:
        return f"Recommended because of {reasons[0]}."

    return (
        "Recommended because of "
        + ", ".join(reasons[:-1])
        + f", and {reasons[-1]}."
    )


def hybrid_search(
    query: str,
    paper_repository: PaperRepository,
    model: SentenceTransformer,
    index: faiss.Index,
    vectorizer: TfidfVectorizer,
    tfidf_matrix: csr_matrix,
    candidate_k: int,
    result_k: int,
    semantic_weight: float,
    tfidf_weight: float,
    keyword_weight: float,
    recency_weight: float,
    half_life_years: float,
) -> list[SearchResult]:
    """Retrieve and rerank papers using a hybrid ranking approach.

    Args:
        query: Search query.
        paper_repository: Repository for accessing paper data.
        model: Sentence transformer model used to encode the papers.
        index: The FAISS index containing the paper embeddings.
        vectorizer: The fitted TF-IDF vectorizer used to transform the query
            and generate the paper TF-IDF matrix.
        tfidf_matrix: Precomputed sparse TF-IDF matrix of shape
            (n_papers, n_features).
        candidate_k: Number of candidate papers to retrieve from the index.
        result_k: Number of results to return.
        semantic_weight: Weight for the semantic score.
        tfidf_weight: Weight for the TF-IDF score.
        keyword_weight: Weight for the keyword overlap score.
        recency_weight: Weight for the recency score.
        half_life_years: Half-life of the recency score in years.

    Returns:
        A list of SearchResult objects containing the necessary information
            about the search results, sorted by descending hybrid scores.

    Raises:
        ValueError: If result_k or the score weights are invalid.
    """
    if result_k <= 0:
        raise ValueError("result_k must be a positive integer.")

    # Fetch the candidate papers using semantic search
    candidate_positions, semantic_scores = semantic_search(
        query=query,
        model=model,
        index=index,
        candidate_top_k=candidate_k,
    )

    # Normalize semantic scores to fall between 0 and 1
    normalized_semantic_scores = np.clip(
        (semantic_scores + 1.0) / 2.0,
        0.0,
        1.0,
    )

    # Extract the candidate Paper objects from the paper repository
    candidate_papers = [
        paper_repository.get_by_position(position)
        for position in candidate_positions
    ]

    # Calculate the keyword overlap score for every candidate paper
    keyword_scores = calculate_keyword_overlap_scores(
        query=query,
        candidate_papers=candidate_papers,
        vectorizer=vectorizer,
    )

    # Calculate the TF-IDF scores for every candidate paper
    tfidf_scores = calculate_tfidf_scores(
        query=query,
        vectorizer=vectorizer,
        matrix=tfidf_matrix,
        candidate_positions=candidate_positions,
    )

    # Calculate the recency scores for every candidate paper
    recency_scores = calculate_recency_scores(
        candidate_papers=candidate_papers,
        half_life_years=half_life_years,
    )

    # Calculate the final hybrid scores
    hybrid_scores = (
        semantic_weight * normalized_semantic_scores
        + keyword_weight * keyword_scores
        + tfidf_weight * tfidf_scores
        + recency_weight * recency_scores
    )

    number_of_results = min(result_k, len(candidate_papers))

    ranked_indices = np.argsort(hybrid_scores, descending=True)

    # Assemble the search results
    results: list[SearchResult] = []

    for i in range(number_of_results):
        paper_index = int(ranked_indices[i])
        semantic_score = float(normalized_semantic_scores[paper_index])
        tfidf_score = float(tfidf_scores[paper_index])
        keyword_overlap_score = float(keyword_scores[paper_index])
        recency_score = float(recency_scores[paper_index])
        hybrid_score = float(hybrid_scores[paper_index])

        # Build the explanation for why the paper was chosen
        explanation = build_explanation(
            semantic_score=semantic_score,
            tfidf_score=tfidf_score,
            keyword_overlap_score=keyword_overlap_score,
            recency_score=recency_score,
        )

        results.append(
            SearchResult(
                paper=candidate_papers[paper_index],
                rank=i + 1,
                semantic_score=semantic_score,
                tfidf_score=tfidf_score,
                keyword_overlap_score=keyword_overlap_score,
                recency_score=recency_score,
                hybrid_score=hybrid_score,
                explanation=explanation,
            )
        )

    return results
