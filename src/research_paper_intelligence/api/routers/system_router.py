"""API router for system information."""

from fastapi import APIRouter, Request

from research_paper_intelligence.api.dependencies import (
    SearchServiceDependency,
)
from research_paper_intelligence.api.schemas.system import SystemInfoResponse

system_router = APIRouter(prefix="/system", tags=["System"])


@system_router.get("/", response_model=SystemInfoResponse)
def get_system_info(
    request: Request, search_request: SearchServiceDependency
) -> SystemInfoResponse:
    """Retrieves system information."""
    return SystemInfoResponse(
        status="ready",
        paper_count=search_request.paper_count,
        embedding_model=search_request.embedding_model,
        retrieval_strategy=search_request.retrieval_strategy,
        ranking_components=search_request.ranking_components,
        faiss_index_type=search_request.faiss_index_type,
        faiss_index_size=search_request.faiss_index_size,
        tfidf_document_count=search_request.tfidf_document_count,
        tfidf_vocabulary_size=search_request.tfidf_vocabulary_size,
        api_version=request.app.version,
    )
