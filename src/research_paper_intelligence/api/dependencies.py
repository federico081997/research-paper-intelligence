"""Defines dependencies for the FastAPI application."""

from typing import Annotated, cast

from fastapi import Depends, Request

from research_paper_intelligence.services.search_service import SearchService


def get_search_service(request: Request) -> SearchService:
    """Return the search service stored in the application state.

    Args:
        request: The current request.

    Returns:
        The search service.
    """
    return cast(SearchService, request.app.state.search_service)


# Dependency for the search service
SearchServiceDependency = Annotated[SearchService, Depends(get_search_service)]
