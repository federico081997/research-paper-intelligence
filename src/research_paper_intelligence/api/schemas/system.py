"""Schemas for the application system information."""

from typing import Literal

from pydantic import BaseModel


class SystemInfoResponse(BaseModel):
    """Description of the system information response."""

    status: Literal["ready", "not_ready"]
    paper_count: int
    embedding_model: str
    retrieval_strategy: str
    ranking_components: list[str]
    faiss_index_type: str
    faiss_index_size: int
    tfidf_document_count: int
    tfidf_vocabulary_size: int
    api_version: str
