"""Defines the domain model for a search result."""

from dataclasses import dataclass

from research_paper_intelligence.domain.paper import Paper


@dataclass(frozen=True)
class SearchResult:
    """Represents a search result."""

    paper: Paper
    rank: int
    semantic_score: float
    tfidf_score: float
    keyword_overlap_score: float
    recency_score: float
    hybrid_score: float
    explanation: str
