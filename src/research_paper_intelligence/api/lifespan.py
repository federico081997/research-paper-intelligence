"""Defines the lifespan events for the FastAPI application."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from research_paper_intelligence.api.bootstrap import (
    create_assistant_service,
    create_search_service,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Defines the lifespan events for the FastAPI application.

    Args:
        app: The FastAPI application.
    """
    app.state.search_service = create_search_service()
    app.state.assistant = create_assistant_service(
        search_service=app.state.search_service
    )

    try:
        yield
    finally:
        app.state.assistant = None
        app.state.search_service = None
