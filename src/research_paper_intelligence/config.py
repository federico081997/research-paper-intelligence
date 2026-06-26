"""Application configuration and environment-variable loading."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
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

    # -------------------------------------------------------------------------
    #   Data and Generated Artifacts
    # -------------------------------------------------------------------------

    raw_papers_path: Path = PROJECT_ROOT / Path("data/raw/arXiv_papers.csv")
    processed_papers_path: Path = PROJECT_ROOT / Path(
        "data/processed/arxiv_cleaned.csv"
    )
    cluster_summary_papers_path: Path = PROJECT_ROOT / Path(
        "data/processed/cluster_summary.csv"
    )
    papers_clustered_path: Path = PROJECT_ROOT / Path(
        "data/processed/papers_clustered.csv"
    )
    faiss_paper_index: Path = PROJECT_ROOT / Path(
        "data/artifacts/faiss_paper_index.bin"
    )
    paper_embeddings_path: Path = PROJECT_ROOT / Path(
        "data/artifacts/paper_embeddings.npy"
    )

    # -------------------------------------------------------------------------
    #   Embeddings
    # -------------------------------------------------------------------------

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

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


@lru_cache
def get_settings() -> Settings:
    """Returns the cached application settings."""
    return Settings()
