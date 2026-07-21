"""Main entry point for the API."""

from fastapi import FastAPI

from research_paper_intelligence.api.lifespan import lifespan
from research_paper_intelligence.api.routers import (
    search_router,
    system_router,
)


def create_app() -> FastAPI:
    """Creates the FastAPI application."""
    app = FastAPI(
        title="Research Paper Intelligence API",
        description="An API for discovering, ranking and analyzing research "
        "papers using hybrid retrieval and agentic AI workflows.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add the system router
    app.include_router(system_router.system_router, prefix="/api/v1")

    # Add the Search router
    app.include_router(search_router.search_router, prefix="/api/v1")

    return app


# Initialize the FastAPI application
fastapi_app = create_app()
