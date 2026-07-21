"""Command-line entry points for the API."""

import uvicorn

from research_paper_intelligence.config import get_settings


def main() -> None:
    """Script to run the API."""
    settings = get_settings()

    uvicorn.run(
        app="research_paper_intelligence.api.app:fastapi_app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )


if __name__ == "__main__":
    main()
