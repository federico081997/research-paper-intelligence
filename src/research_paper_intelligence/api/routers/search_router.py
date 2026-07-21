"""Contains the search router for the API."""

from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Query

from research_paper_intelligence.api.dependencies import (
    SearchServiceDependency,
)
from research_paper_intelligence.api.schemas.search import (
    SearchRequest,
    SearchResponse,
)

search_router = APIRouter(prefix="/search", tags=["Search"])


@search_router.get("/", response_model=SearchResponse)
def search_papers(
    search_request: Annotated[SearchRequest, Query()],
    search_service: SearchServiceDependency,
) -> SearchResponse:
    """Return the top-ranked papers matching the search query.

    Args:
        search_request: The search request containing the query and result
            count.
        search_service: The search service dependency.

    Returns:
        The structured search response.
    """
    start_time = perf_counter()

    # Perform the search using the search service.
    search_response = search_service.search(
        search_request.query, search_request.result_k
    )

    time_elapsed = perf_counter() - start_time

    # Convert the search response to a structured format and return it.
    return SearchResponse.from_search_results(
        results=search_response,
        time_elapsed=time_elapsed,
    )
