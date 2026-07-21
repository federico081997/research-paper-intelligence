"""API schemas for the search endpoint."""

from collections.abc import Sequence
from datetime import date
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from research_paper_intelligence.domain.search_result import SearchResult


class SearchRequest(BaseModel):
    """Defines the request schema for the search endpoint."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(
        min_length=1,
        max_length=500,
        description="The search query to be used to retrieve research papers.",
    )
    result_k: int = Field(
        default=10,
        gt=0,
        le=100,
        description="The number of papers to be returned to the user.",
    )


class SearchResultItem(BaseModel):
    """Defines the response schema for one search result."""

    paper_id: str
    title: str
    abstract: str
    authors: str
    category: str
    published_date: date
    rank: int
    semantic_score: float
    tfidf_score: float
    keyword_overlap_score: float
    recency_score: float
    hybrid_score: float
    explanation: str

    @classmethod
    def from_search_result(cls, search_result: SearchResult) -> Self:
        """Constructs a SearchResponse from a SearchResult.

        Args:
            search_result: The search result to construct from.
        """
        return cls(
            paper_id=search_result.paper.paper_id,
            title=search_result.paper.title,
            abstract=search_result.paper.abstract,
            authors=search_result.paper.authors,
            category=search_result.paper.category,
            published_date=search_result.paper.published_date,
            rank=search_result.rank,
            semantic_score=search_result.semantic_score,
            tfidf_score=search_result.tfidf_score,
            keyword_overlap_score=search_result.keyword_overlap_score,
            recency_score=search_result.recency_score,
            hybrid_score=search_result.hybrid_score,
            explanation=search_result.explanation,
        )


class SearchResponse(BaseModel):
    """Defines the main response schema for the search endpoint."""

    results: list[SearchResultItem]
    total: int
    time_elapsed: float

    @classmethod
    def from_search_results(
        cls,
        results: Sequence[SearchResult],
        time_elapsed: float,
    ) -> Self:
        """Constructs the complete SearchResponse from the search results.

        Args:
            results: The search results to construct from.
            time_elapsed: The time elapsed during the search.
        """
        return cls(
            results=[
                SearchResultItem.from_search_result(search_result)
                for search_result in results
            ],
            total=len(results),
            time_elapsed=time_elapsed,
        )
