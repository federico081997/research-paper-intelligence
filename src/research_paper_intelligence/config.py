"""Application configuration and environment-variable loading."""

from functools import cache
from pathlib import Path
from typing import Literal, Self

import numpy as np
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Main project path
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Validated configuration for the Research Paper Intelligence project."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="RPI_",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    #   Application Settings
    # -------------------------------------------------------------------------

    app_name: str = "Research Paper Intelligence"

    environment: Literal[
        "development",
        "test",
        "production",
    ] = "development"

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    device: Literal["cpu", "cuda", "auto"] = "auto"

    # -------------------------------------------------------------------------
    #   Data and Generated Artifacts
    # -------------------------------------------------------------------------

    raw_papers_path: Path = PROJECT_ROOT / Path("data/raw/arxiv_papers.csv")
    processed_papers_path: Path = PROJECT_ROOT / Path(
        "data/processed/arxiv_cleaned.csv"
    )
    cluster_summary_path: Path = PROJECT_ROOT / Path(
        "data/processed/cluster_summary.csv"
    )
    papers_clustered_path: Path = PROJECT_ROOT / Path(
        "data/processed/papers_clustered.csv"
    )
    faiss_index_papers_path: Path = PROJECT_ROOT / Path(
        "data/artifacts/faiss_paper_index.bin"
    )
    tfidf_vectorizer_path: Path = PROJECT_ROOT / Path(
        "data/artifacts/tfidf_vectorizer.joblib"
    )
    tfidf_matrix_path: Path = PROJECT_ROOT / Path(
        "data/artifacts/tfidf_matrix.npz"
    )
    paper_embeddings_path: Path = PROJECT_ROOT / Path(
        "data/artifacts/paper_embeddings.npy"
    )

    # -------------------------------------------------------------------------
    #   HuggingFace Data and Generated Artifacts
    # -------------------------------------------------------------------------

    hf_repository: str = "Federico081997/research-paper-intelligence-data"
    hf_raw_papers_file: str = "raw_data/arxiv_papers.csv"
    hf_processed_papers_file: str = "processed_data/arxiv_cleaned.csv"
    hf_cluster_summary_file: str = "processed_data/cluster_summary.csv"
    hf_papers_clustered_file: str = "processed_data/papers_clustered.csv"
    hf_faiss_index_papers_file: str = "artifacts/faiss_paper_index.bin"
    hf_tfidf_vectorizer_file: str = "artifacts/tfidf_vectorizer.joblib"
    hf_tfidf_matrix_file: str = "artifacts/tfidf_matrix.npz"
    hf_paper_embeddings_file: str = "artifacts/paper_embeddings.npy"

    # -------------------------------------------------------------------------
    #   Embeddings
    # -------------------------------------------------------------------------

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # -------------------------------------------------------------------------
    #   Hybrid ranking
    # -------------------------------------------------------------------------

    candidate_top_k: int = Field(default=100, gt=1)
    half_life_years: float = Field(default=5.0, gt=0.0)
    semantic_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    tfidf_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    recency_weight: float = Field(default=0.05, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_score_weights(self) -> Self:
        """Score weights must sum to 1.0."""
        total = (
            self.semantic_weight
            + self.tfidf_weight
            + self.keyword_weight
            + self.recency_weight
        )
        if not np.isclose(total, 1.0):
            raise ValueError("Score weights must sum to 1.0.")
        return self

    # -------------------------------------------------------------------------
    #   Ollama
    # -------------------------------------------------------------------------

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: int = Field(
        default=120, gt=0, description=" Ollama timeout in seconds"
    )
    ollama_batch_size: int = Field(
        default=10,
        gt=1,
        description="Number of clusters processed in each Ollama request.",
    )
    ollama_temperature: float | int = Field(
        default=0.1,
        gt=0.0,
        lt=2.0,
        description="Controls randomness in Ollama responses.",
    )


@cache
def get_settings() -> Settings:
    """Returns the cached application settings."""
    return Settings()
