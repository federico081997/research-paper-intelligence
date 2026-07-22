"""Client for retrieving the search results."""

import httpx2

from research_paper_intelligence.api.schemas.search import (
    SearchResponse,
)
from research_paper_intelligence.config import get_settings


def search_papers(
    query: str,
    result_k: int,
) -> SearchResponse:
    """Retrieve the search results for a query.

    Args:
        query: search query
        result_k: number of results to return
    """
    settings = get_settings()

    with httpx2.Client(
        base_url="http://" + settings.api_host + ":" + str(settings.api_port),
        timeout=settings.api_timeout_seconds,
    ) as client:
        response = client.get(
            "api/v1/search/?query=" + query + "&result_k=" + str(result_k)
        )
        response.raise_for_status()

        return SearchResponse.model_validate(response.json())
