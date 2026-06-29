"""Defines the domain model for a search result."""

from dataclasses import dataclass

from research_paper_intelligence.domain.paper import Paper


@dataclass(frozen=True)
class SearchResult:
    """Represents a search result."""

    paper: Paper
    score: float
    rank: int
