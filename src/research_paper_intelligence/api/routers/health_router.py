"""Define the health API endpoint."""

from fastapi import APIRouter

health_router = APIRouter(tags=["health"])


@health_router.get("/health")
def health_check() -> dict[str, str]:
    """Return the current API health status."""
    return {"status": "healthy"}
